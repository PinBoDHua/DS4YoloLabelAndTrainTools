"""主窗口：菜单/工具栏/面板布局、数据集加载、标注保存、撤销重做、快捷键。"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QAction, QActionGroup, QImage, QKeySequence, QPixmap
from PySide6.QtWidgets import (QApplication, QDialog, QDockWidget, QFileDialog,
                               QInputDialog, QLabel, QMainWindow, QMenu, QMessageBox,
                               QProgressDialog, QTabWidget, QToolBar)

from app.config import config
from app.core.dataset_manager import Dataset, create_dataset
from app.core.label_data import DatasetMeta, ImageAnnotation, LabelType
from app.core.label_io import write_yolo_txt
from app.theme import theme
from app.ui.canvas import COCO_SKELETON, LabelCanvas
from app.ui.class_panel import ClassPanel
from app.ui.image_panel import ImagePanel
from app.ui.test_panel import TestPanel
from app.ui.train_panel import TrainPanel
from app.ui.welcome_page import WelcomePage
from app.ui import tools as tools_mod

logger = logging.getLogger(__name__)

TOOL_INFO = [
    ("select", "选择", "V", "选择/编辑标注"),
    ("rect", "矩形框", "W", "绘制检测框"),
    ("polygon", "多边形", "E", "绘制分割多边形"),
    ("pose", "关键点", "R", "绘制关键点（先画框后点关键点）"),
    ("obb", "旋转框", "T", "绘制旋转框（OBB）"),
]
TOOL_KEYS = {key: name for name, _, key, _ in TOOL_INFO}


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLO 标注训练工具")
        self.resize(1440, 900)

        # ---- 数据状态 ----
        self.dataset: Dataset | None = None
        self.images: list[str] = []
        self.current_idx: int = -1
        self.current_ann: ImageAnnotation | None = None
        self._undo_stack: list[dict] = []  # 快照 JSON
        self._redo_stack: list[dict] = []
        self._pose_kpts = 17

        # ---- UI 构建（中央为开始/标注/训练/推理测试四页合一）----
        self.welcome_page = WelcomePage(self)
        self.canvas = LabelCanvas(self)
        self.train_panel = TrainPanel(self)
        self.test_panel = TestPanel(self)
        self.main_tabs = QTabWidget()
        self.main_tabs.addTab(self.welcome_page, "开始")
        self.main_tabs.addTab(self.canvas, "标注")
        self.main_tabs.addTab(self.train_panel, "训练")
        self.main_tabs.addTab(self.test_panel, "推理测试")
        self.setCentralWidget(self.main_tabs)

        self.class_panel = ClassPanel(self)
        self.image_panel = ImagePanel(self)

        self._build_docks()
        self._build_toolbar()
        self._build_menus()
        self._build_statusbar()

        # ---- 连接 ----
        self.canvas.labels_changed.connect(self._on_labels_changed)
        self.canvas.edit_begin.connect(self._on_edit_begin)
        self.canvas.mouse_moved.connect(self._on_mouse_moved)
        self.canvas.shape_selected.connect(self._on_shape_selected)
        self.canvas.tool_auto_changed.connect(self._on_tool_auto_changed)
        self.canvas.zoom_changed.connect(self._on_zoom_changed)
        self.canvas.context_menu_requested.connect(self._on_canvas_context_menu)
        self.canvas.navigate_requested.connect(self._navigate)
        self.canvas.class_key.connect(self._on_class_key)
        self.image_panel.image_selected.connect(self._on_image_clicked)
        self.class_panel.current_class_changed.connect(self._on_class_changed)
        self.class_panel.classes_changed.connect(self._on_classes_changed)
        self.train_panel.running_changed.connect(self._on_train_running)
        self.train_panel.best_model_ready.connect(self.test_panel.set_model)
        self.welcome_page.new_dataset_requested.connect(self._new_dataset)
        self.welcome_page.open_dataset_requested.connect(self._open_dataset_dialog)
        self.welcome_page.open_path_requested.connect(self._open_dataset)

        # 主题恢复
        theme.theme_changed.connect(self._on_theme_changed)
        self._update_theme_ui()

        # 默认选择工具
        self._tool_name = "select"
        self._set_tool("select")

        self._update_actions()

        # 应用退出前兜底清理所有后台线程（覆盖直接 app.quit() 等所有退出路径）
        QApplication.instance().aboutToQuit.connect(self._on_app_about_to_quit)

    def _on_app_about_to_quit(self) -> None:
        """退出前停掉所有后台线程，避免线程运行时被销毁导致 C++ 层崩溃。"""
        try:
            self._save_dataset_state()
        except Exception:
            pass
        try:
            from app.core.device_probe import shutdown_probe
            shutdown_probe()
        except Exception:
            pass
        try:
            self.image_panel._shutdown()
        except Exception:
            pass
        try:
            self.test_panel.shutdown()
        except Exception:
            pass
        try:
            w = getattr(self.train_panel, "_worker", None)
            if w is not None and w.isRunning():
                w.stop()
                w.wait()
        except Exception:
            pass
        try:
            w = getattr(self.train_panel, "_worker_export", None)
            if w is not None and w.isRunning():
                w.wait()
        except Exception:
            pass
        try:
            if hasattr(self, "_bg_tasks"):
                for t in self._bg_tasks:
                    if t.isRunning():
                        t.requestInterruption()
                        t.wait()
        except Exception:
            pass

    # ================= UI 构建 =================
    def _build_docks(self) -> None:
        # 左侧类别
        self.dock_classes = QDockWidget("类别管理", self)
        self.dock_classes.setWidget(self.class_panel)
        self.dock_classes.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_classes)

        # 右侧图片
        self.dock_images = QDockWidget("图片列表", self)
        self.dock_images.setWidget(self.image_panel)
        self.dock_images.setFeatures(
            QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_images)

        self.resizeDocks([self.dock_classes], [180], Qt.Horizontal)
        self.resizeDocks([self.dock_images], [240], Qt.Horizontal)

    def _build_toolbar(self) -> None:
        tb = QToolBar("标注工具")
        tb.setObjectName("main_toolbar")
        tb.setMovable(False)
        tb.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.addToolBar(tb)
        self._tool_actions: dict[str, QAction] = {}
        group = QActionGroup(self)
        group.setExclusive(True)
        for name, label, key, tip in TOOL_INFO:
            act = QAction(label, self)
            act.setShortcut(QKeySequence(key))
            act.setToolTip(f"{tip}（快捷键 {key}）")
            act.setCheckable(True)
            act.triggered.connect(lambda _=False, n=name: self._set_tool(n))
            group.addAction(act)
            tb.addAction(act)
            self._tool_actions[name] = act

        tb.addSeparator()
        self.act_prev = QAction("上一张", self)
        self.act_prev.setShortcut(QKeySequence("A"))
        self.act_prev.setToolTip("上一张（快捷键 A / ←）")
        self.act_prev.triggered.connect(lambda: self._navigate(-1))
        tb.addAction(self.act_prev)

        self.act_next = QAction("下一张", self)
        self.act_next.setShortcut(QKeySequence("D"))
        self.act_next.setToolTip("下一张（快捷键 D / →）")
        self.act_next.triggered.connect(lambda: self._navigate(1))
        tb.addAction(self.act_next)

        # 缩放控件：缩小 / 百分比 / 放大 / 适应窗口
        tb.addSeparator()
        self.act_zoom_out = QAction("缩小", self)
        self.act_zoom_out.setToolTip("缩小（快捷键 -）")
        self.act_zoom_out.triggered.connect(lambda: self.canvas.zoom_by(0.8))
        tb.addAction(self.act_zoom_out)
        self.lb_zoom_tb = QLabel("100%")
        self.lb_zoom_tb.setMinimumWidth(52)
        self.lb_zoom_tb.setAlignment(Qt.AlignCenter)
        self.lb_zoom_tb.setToolTip("当前缩放比例")
        tb.addWidget(self.lb_zoom_tb)
        self.act_zoom_in = QAction("放大", self)
        self.act_zoom_in.setToolTip("放大（快捷键 +）")
        self.act_zoom_in.triggered.connect(lambda: self.canvas.zoom_by(1.25))
        tb.addAction(self.act_zoom_in)
        self.act_fit_tb = QAction("适应窗口", self)
        self.act_fit_tb.setToolTip("适应窗口（快捷键 F）")
        self.act_fit_tb.triggered.connect(self.canvas.reset_zoom)
        tb.addAction(self.act_fit_tb)

    def _build_menus(self) -> None:
        m_file = self.menuBar().addMenu("文件(&F)")
        self._add_action(m_file, "新建数据集...", self._new_dataset, "Ctrl+N")
        self._add_action(m_file, "打开数据集...", self._open_dataset_dialog, "Ctrl+O")
        self._add_menu_recent(m_file)
        m_file.addSeparator()
        self._add_action(m_file, "导入图片...", self._import_images)
        self._add_action(m_file, "导出可视化图...", self._export_visualization)
        self._add_action(m_file, "批量导出可视化图...", self._export_visualization_batch)
        m_file.addSeparator()
        self._add_action(m_file, "退出", self.close, "Alt+F4")

        m_edit = self.menuBar().addMenu("编辑(&E)")
        self.act_undo = self._add_action(m_edit, "撤销", self._undo, "Ctrl+Z")
        self.act_redo = self._add_action(m_edit, "重做", self._redo, "Ctrl+Y")
        m_edit.addSeparator()
        self.act_copy = self._add_action(m_edit, "复制标注", self._copy_labels, "Ctrl+C")
        self.act_paste = self._add_action(m_edit, "粘贴标注", self._paste_labels, "Ctrl+V")
        self.act_duplicate = self._add_action(m_edit, "原位复制选中标注",
                                              self.canvas.duplicate_selected, "Ctrl+D")
        m_edit.addSeparator()
        self._add_action(m_edit, "删除选中标注", self.canvas.delete_selected, "Delete")
        self._add_action(m_edit, "全选标注", self.canvas.select_all, "Ctrl+A")

        m_view = self.menuBar().addMenu("视图(&V)")
        self.act_fit = self._add_action(m_view, "适应窗口", self.canvas.reset_zoom, "F")
        self.act_zoom_in_menu = self._add_action(m_view, "放大", lambda: self.canvas.zoom_by(1.25), "Ctrl+=")
        self.act_zoom_out_menu = self._add_action(m_view, "缩小", lambda: self.canvas.zoom_by(0.8), "Ctrl+-")
        m_view.addSeparator()
        self.act_show_labels = self._add_action(m_view, "显示类别文本", self._toggle_show_labels)
        self.act_show_labels.setCheckable(True)
        self.act_show_labels.setChecked(config.get("show_labels", True))
        self.act_crosshair = self._add_action(m_view, "十字准星", self._toggle_crosshair)
        self.act_crosshair.setCheckable(True)
        self.act_crosshair.setChecked(config.get("crosshair", False))
        self.act_crosshair.setToolTip("在画布上显示跟随鼠标的十字准星，便于对齐标注")
        self.canvas.set_crosshair(self.act_crosshair.isChecked())
        m_view.addSeparator()
        self._add_action(m_view, "设置...", self._open_settings)
        m_theme = m_view.addMenu("主题")
        self._theme_light = m_theme.addAction("浅色主题")
        self._theme_light.setCheckable(True)
        self._theme_dark = m_theme.addAction("深色主题")
        self._theme_dark.setCheckable(True)
        theme_group = QActionGroup(self)
        theme_group.setExclusive(True)
        for a in (self._theme_light, self._theme_dark):
            theme_group.addAction(a)
        self._theme_light.triggered.connect(lambda: self._set_theme("light"))
        self._theme_dark.triggered.connect(lambda: self._set_theme("dark"))

        m_ai = self.menuBar().addMenu("AI 辅助(&A)")
        self._add_action(m_ai, "模型预标注...", self._prelabel)
        self._add_action(m_ai, "SAM 智能分割...", self._sam_segment)

        m_data = self.menuBar().addMenu("数据集(&D)")
        self._add_action(m_data, "数据集统计...", self._dataset_stats)
        self._add_action(m_data, "划分训练/验证/测试集...", self._split_dataset)
        m_data.addSeparator()
        self._add_action(m_data, "VOC XML 转 YOLO...", self._convert_voc)
        self._add_action(m_data, "COCO JSON 转 YOLO...", self._convert_coco)

        m_train = self.menuBar().addMenu("训练(&T)")
        self._add_action(m_train, "训练面板", self._show_train_panel)
        self._add_action(m_train, "推理测试", self._show_test_panel)

        m_help = self.menuBar().addMenu("帮助(&H)")
        self._add_action(m_help, "快捷键说明", self._show_shortcuts)

    def _build_statusbar(self) -> None:
        self.lb_pos = QLabel("坐标: -")
        self.lb_img = QLabel("图片: 0/0")
        self.lb_zoom = QLabel("缩放: 100%")
        self.lb_size = QLabel("尺寸: -")
        self.lb_class = QLabel("类别: -")
        self.lb_count = QLabel("标注: 0")
        self.lb_progress = QLabel("进度: 0/0 已标注")
        self.lb_task = QLabel("任务: 检测")
        for lb in (self.lb_pos, self.lb_img, self.lb_zoom, self.lb_size,
                   self.lb_class, self.lb_count, self.lb_progress, self.lb_task):
            lb.setContentsMargins(8, 0, 8, 0)
            self.statusBar().addWidget(lb)

    def _add_action(self, menu, text, slot, shortcut=None) -> QAction:
        act = QAction(text, self)
        if shortcut:
            act.setShortcut(QKeySequence(shortcut))
        act.triggered.connect(slot)
        menu.addAction(act)
        return act

    def _add_menu_recent(self, menu) -> None:
        recent = config.recent_projects
        if recent:
            sub = menu.addMenu("最近打开")
            for p in recent[:5]:
                act = QAction(Path(p).name, self)
                act.setToolTip(p)
                act.triggered.connect(lambda _=False, path=p: self._open_dataset(path))
                sub.addAction(act)

    def _open_settings(self) -> None:
        from app.ui.settings_dialog import SettingsDialog
        dlg = SettingsDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            # 应用标注相关设置
            from app.config import config as cfg
            from app.ui.canvas import COCO_SKELETON
            self._pose_kpts = cfg.get("pose_kpts", 17)
            if self.canvas.image_item is not None:
                self.canvas.pose_kpts = self._pose_kpts
            skel = None
            if cfg.get("skeleton", "none") == "coco17":
                skel = COCO_SKELETON
            self.canvas.skeleton = skel
            for s in self.canvas.shapes:
                s.skeleton = skel
                s.update()
            self.canvas.show_labels = cfg.get("show_labels", True)
            self.act_show_labels.setChecked(self.canvas.show_labels)
            for s in self.canvas.shapes:
                s.show_label = self.canvas.show_labels
            self.canvas.set_crosshair(cfg.get("crosshair", False))
            self.act_crosshair.setChecked(cfg.get("crosshair", False))
            self.canvas.viewport().update()

    # ================= 主题 =================
    def _set_theme(self, name: str) -> None:
        theme.apply(QApplication.instance(), name)
        self._update_theme_ui()

    def _update_theme_ui(self) -> None:
        is_dark = theme.is_dark
        self._theme_dark.setChecked(is_dark)
        self._theme_light.setChecked(not is_dark)

    def _on_theme_changed(self, name: str) -> None:
        self._update_theme_ui()

    # ================= 数据集 =================
    def _new_dataset(self) -> None:
        from app.ui.dataset_dialog import NewDatasetDialog
        dlg = NewDatasetDialog(self)
        if dlg.exec() != QDialog.DialogCode.Accepted:
            return
        root = dlg.dataset_root()
        classes = dlg.classes()
        task = dlg.task_type()
        try:
            ds = create_dataset(root, classes, task,
                                pose_kpts=int(config.get("pose_kpts", 17)))
        except Exception as e:
            QMessageBox.critical(self, "错误", f"创建数据集失败：{e}")
            return
        self._open_dataset(str(ds.root), ds)

    def _open_dataset_dialog(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self._open_dataset(path)

    def _open_dataset(self, root: str, ds: Dataset | None = None) -> None:
        try:
            dataset = ds or Dataset(root)
        except Exception as e:
            QMessageBox.critical(self, "错误", f"打开数据集失败：{e}")
            return
        if not dataset.meta.classes:
            QMessageBox.warning(
                self, "提示",
                "该目录不是有效的数据集（缺少 dataset.json）。\n请通过「新建数据集」创建，或选择包含 dataset.json 的目录。",
            )
            return
        # 记住上一个数据集的浏览位置
        self._save_current_annotation(quiet=True)
        self._save_dataset_state()
        self.dataset = dataset
        self.images = dataset.scan_images()
        self.train_panel.set_dataset(dataset)
        self.class_panel.set_classes(dataset.meta.classes)
        self.canvas.classes = dataset.meta.classes
        self.canvas.current_class_id = 0
        self._pose_kpts = dataset.meta.pose_kpts
        self.canvas.pose_kpts = self._pose_kpts
        self._labeled_cache = dataset.labeled_set()
        self.image_panel.set_images(self.images, self._labeled_cache)
        config.add_recent(str(dataset.root))
        self.welcome_page.refresh()
        self.lb_task.setText(f"任务: {_task_name(dataset.meta.task)}")
        if self.images:
            # 恢复上次浏览位置与工具
            state = config.get("dataset_state", {}) or {}
            st = state.get(str(dataset.root), {}) or {}
            idx = st.get("index", 0)
            self.current_idx = idx if isinstance(idx, int) and 0 <= idx < len(self.images) else 0
            tool = st.get("tool")
            if tool in TOOL_KEYS:
                self._tool_name = tool
            # 先切到标注页再加载图片：否则画布视口尚未布局，
            # reset_zoom 会按极小视口适配导致图片几乎不可见
            self.main_tabs.setCurrentWidget(self.canvas)
            self._load_current_image()
        else:
            self.current_idx = -1
            self._clear_canvas()
            self.main_tabs.setCurrentWidget(self.canvas)
            QMessageBox.information(self, "提示", "数据集为空，请通过「文件 → 导入图片」添加图片。")
        self._update_actions()
        logger.info("已打开数据集: %s (%d 张图片)", root, len(self.images))

    def _clear_canvas(self) -> None:
        self.current_ann = None
        self.canvas.current_ann = None
        self.canvas.clear_shapes()
        self._undo_stack.clear()
        self._redo_stack.clear()
        if self.canvas.image_item:
            self.canvas.scene.removeItem(self.canvas.image_item)
            self.canvas.image_item = None

    def _import_images(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "提示", "请先创建或打开数据集。")
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "图片文件 (*.jpg *.jpeg *.png *.bmp *.webp)")
        if not files:
            return
        current_path = self.current_ann.image_path if self.current_ann is not None else None
        self._save_current_annotation(quiet=True)
        imported, skipped, failed = self.dataset.import_images(files)
        self.images = self.dataset.scan_images()
        self._labeled_cache = self.dataset.labeled_set()
        self.image_panel.set_images(self.images, self._labeled_cache)
        if current_path in self.images:
            self.current_idx = self.images.index(current_path)
            self._load_current_image()
        elif self.images:
            self.current_idx = min(max(self.current_idx, 0), len(self.images) - 1)
            self._load_current_image()
        msg = f"成功导入 {imported} 张图片。"
        if failed:
            msg += f"\n失败 {failed} 张（文件不存在或不可读）。"
        QMessageBox.information(self, "完成", msg)

    # ================= 图片浏览与标注保存 =================
    def _load_current_image(self) -> None:
        if self.current_idx < 0 or self.current_idx >= len(self.images):
            return
        path = self.images[self.current_idx]
        img = QImage(path)
        if img.isNull():
            QMessageBox.warning(self, "警告", f"无法读取图片：{path}")
            return
        self.canvas.set_image(img)
        ann = self.dataset.read_annotation(path)
        self.current_ann = ann
        self.canvas.set_annotations(ann)
        # 重置绘制工具状态，避免上一张图未完成的绘制（如画了一半的多边形/关键点）残留到本图
        self._set_tool(self._tool_name)
        self._undo_stack.clear()
        self._redo_stack.clear()
        self.image_panel.select_path(path)
        name = Path(path).name
        self.lb_img.setText(f"图片: {self.current_idx + 1}/{len(self.images)}  {name}")
        self.lb_count.setText(f"标注: {len(ann.labels)}")
        self.lb_size.setText(f"尺寸: {img.width()}x{img.height()}")
        self._update_progress_label()
        ds_name = Path(self.dataset.root).name
        self.setWindowTitle(f"{name} - {ds_name} - YOLO 标注训练工具")
        self._update_actions()

    def _update_progress_label(self) -> None:
        total = len(self.images)
        labeled = len(self._labeled_cache) if hasattr(self, "_labeled_cache") else 0
        self.lb_progress.setText(f"进度: {labeled}/{total} 已标注")

    def _navigate(self, delta: int) -> None:
        if not self.images:
            return
        new_idx = self.current_idx + delta
        if 0 <= new_idx < len(self.images):
            self._save_current_annotation()
            self.current_idx = new_idx
            self._load_current_image()

    def _on_image_clicked(self, path: str) -> None:
        if path in self.images:
            new_idx = self.images.index(path)
            if new_idx != self.current_idx:
                self._save_current_annotation()
                self.current_idx = new_idx
                self._load_current_image()

    def _save_current_annotation(self, quiet: bool = False) -> None:
        """把画布标注写回 label txt。"""
        if self.dataset is None or self.current_ann is None:
            return
        path = self.current_ann.image_path
        if not path:
            return
        ann = self.canvas.get_annotation()
        ann.image_path = path
        write_yolo_txt(ann, self.dataset.label_path(path))
        self.current_ann = ann
        labeled = bool(ann.labels)
        # 只更新单张状态，避免全量扫描目录
        if labeled:
            self._labeled_cache.add(path)
        else:
            self._labeled_cache.discard(path)
        self.image_panel.mark_labeled(path, labeled)
        self._update_progress_label()

    # ================= 标注编辑 =================
    def _on_edit_begin(self) -> None:
        """编辑开始：压入当前快照用于撤销。"""
        if self.current_ann is None:
            return
        snapshot = json.dumps(self.canvas.get_annotation().to_dict(), ensure_ascii=False)
        if self._undo_stack and self._undo_stack[-1] == snapshot:
            return
        self._undo_stack.append(snapshot)
        if len(self._undo_stack) > 100:
            self._undo_stack.pop(0)
        self._redo_stack.clear()

    def _on_labels_changed(self) -> None:
        """标注变化：更新状态栏计数（保存延迟到切图/退出）。"""
        if self.current_ann is not None:
            n = len(self.canvas.collect_shapes())
            self.lb_count.setText(f"标注: {n}")
            self._update_actions()
            if config.get("auto_save", True):
                # 自动保存（写盘开销小）
                self._save_current_annotation(quiet=True)

    def _undo(self) -> None:
        if not self._undo_stack:
            return
        current = json.dumps(self.canvas.get_annotation().to_dict(), ensure_ascii=False)
        self._redo_stack.append(current)
        snap = self._undo_stack.pop()
        ann = ImageAnnotation.from_dict(json.loads(snap))
        ann.image_path = self.current_ann.image_path
        self.canvas.set_annotations(ann)
        self._on_labels_changed()
        self._update_actions()

    def _redo(self) -> None:
        if not self._redo_stack:
            return
        current = json.dumps(self.canvas.get_annotation().to_dict(), ensure_ascii=False)
        self._undo_stack.append(current)
        snap = self._redo_stack.pop()
        ann = ImageAnnotation.from_dict(json.loads(snap))
        ann.image_path = self.current_ann.image_path
        self.canvas.set_annotations(ann)
        self._on_labels_changed()
        self._update_actions()

    def _copy_labels(self) -> None:
        self._clipboard = self.canvas.copy_selected_labels()
        if not self._clipboard:
            QMessageBox.information(self, "提示", "请先选中要复制的标注。")

    def _paste_labels(self) -> None:
        if getattr(self, "_clipboard", None):
            self.canvas.paste_labels(self._clipboard)

    def _on_class_changed(self, class_id: int) -> None:
        self.canvas.current_class_id = class_id
        name = self.dataset.meta.classes[class_id] if self.dataset else "-"
        self.lb_class.setText(f"类别: {class_id} {name}")

    def _on_classes_changed(self, classes: list[str]) -> None:
        if self.dataset is not None:
            self.dataset.meta.classes = classes
            self.dataset.save_meta()
            self.canvas.classes = classes

    def _on_shape_selected(self, item) -> None:
        if item is not None:
            cid = item.label.class_id
            name = self.dataset.meta.classes[cid] if self.dataset and cid < len(self.dataset.meta.classes) else str(cid)
            self.lb_class.setText(f"选中: {cid} {name}")
        else:
            self.lb_class.setText(
                f"类别: {self.canvas.current_class_id} "
                f"{self.dataset.meta.classes[self.canvas.current_class_id] if self.dataset else '-'}")

    def _on_mouse_moved(self, pos) -> None:
        self.lb_pos.setText(f"坐标: ({int(pos.x())}, {int(pos.y())})")

    def _toggle_show_labels(self, checked: bool) -> None:
        config.set("show_labels", checked)
        for s in self.canvas.shapes:
            s.show_label = checked
            s.update()

    def _toggle_crosshair(self, checked: bool) -> None:
        config.set("crosshair", checked)
        self.canvas.set_crosshair(checked)

    def _on_zoom_changed(self, zoom: float) -> None:
        """画布缩放变化：同步工具栏与状态栏。"""
        text = f"{zoom * 100:.0f}%"
        self.lb_zoom_tb.setText(text)
        self.lb_zoom.setText(f"缩放: {text}")

    def _on_class_key(self, class_id: int) -> None:
        """数字键 1-9：切换当前标注类别。"""
        if self.dataset is None:
            return
        if 0 <= class_id < len(self.dataset.meta.classes):
            self.class_panel.set_current(class_id)

    def _on_canvas_context_menu(self, event) -> None:
        """画布右键菜单：常用编辑操作 + 更改类别。"""
        menu = QMenu(self)
        has_sel = self.canvas.selected_shape() is not None
        has_img = self.current_idx >= 0
        menu.addAction(self.act_undo)
        menu.addAction(self.act_redo)
        menu.addSeparator()
        act_del = menu.addAction("删除选中标注")
        act_del.setEnabled(has_sel)
        act_del.triggered.connect(self.canvas.delete_selected)
        act_copy = menu.addAction("复制标注")
        act_copy.setEnabled(has_sel)
        act_copy.triggered.connect(self._copy_labels)
        act_paste = menu.addAction("粘贴标注")
        act_paste.setEnabled(has_img and bool(getattr(self, "_clipboard", None)))
        act_paste.triggered.connect(self._paste_labels)
        act_dup = menu.addAction("原位复制选中标注")
        act_dup.setEnabled(has_sel)
        act_dup.triggered.connect(self.canvas.duplicate_selected)
        menu.addSeparator()
        if self.dataset and self.dataset.meta.classes:
            sub = menu.addMenu("更改类别")
            for i, name in enumerate(self.dataset.meta.classes):
                a = sub.addAction(f"{i} - {name}")
                a.triggered.connect(lambda _=False, cid=i: self.canvas.set_class_of_selected(cid))
        menu.addSeparator()
        menu.addAction(self.act_fit)
        menu.exec(event.globalPos())

    def _save_dataset_state(self) -> None:
        """记住当前数据集浏览位置/工具（重新打开时恢复）。"""
        if self.dataset is None:
            return
        state = dict(config.get("dataset_state", {}) or {})
        state[str(self.dataset.root)] = {"index": self.current_idx,
                                         "tool": self._tool_name}
        config.set("dataset_state", state)

    # ================= 工具 =================
    def _set_tool(self, name: str) -> None:
        self._tool_name = name
        self.canvas.set_tool(tools_mod.make_tool(name, self.canvas))
        for n, act in self._tool_actions.items():
            act.setChecked(n == name)
        self._tool_actions.get("select").setChecked(name == "select")

    def _on_tool_auto_changed(self, name: str) -> None:
        """画布内部自动切换工具（如 OBB 完成），同步工具栏状态。"""
        self._tool_name = name
        for n, act in self._tool_actions.items():
            act.setChecked(n == name)

    def _update_actions(self) -> None:
        has_dataset = self.dataset is not None and bool(self.images)
        has_img = self.current_idx >= 0
        self.act_next.setEnabled(has_img and self.current_idx < len(self.images) - 1)
        self.act_prev.setEnabled(has_img and self.current_idx > 0)
        self.act_undo.setEnabled(bool(self._undo_stack))
        self.act_redo.setEnabled(bool(self._redo_stack))
        for act in self._tool_actions.values():
            act.setEnabled(has_img)
        self.act_fit.setEnabled(has_img)
        self.act_fit_tb.setEnabled(has_img)
        self.act_zoom_in.setEnabled(has_img)
        self.act_zoom_out.setEnabled(has_img)
        self.act_zoom_in_menu.setEnabled(has_img)
        self.act_zoom_out_menu.setEnabled(has_img)
        self.act_copy.setEnabled(has_img)
        self.act_paste.setEnabled(has_img)
        self.act_duplicate.setEnabled(has_img)
        self.act_show_labels.setEnabled(has_img)
        self.act_crosshair.setEnabled(has_img)

    # ================= 数据集功能 =================
    def _dataset_stats(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "提示", "请先打开数据集。")
            return
        from app.ui.dataset_dialog import StatsDialog
        StatsDialog(self, self.dataset).exec()

    def _split_dataset(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "提示", "请先打开数据集。")
            return
        self._save_current_annotation()
        from app.ui.dataset_dialog import SplitDialog
        dlg = SplitDialog(self, self.dataset)
        dlg.exec()

    def _convert_voc(self) -> None:
        from app.ui.dataset_dialog import ConvertVocDialog
        dlg = ConvertVocDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            src = dlg.source_dir()
            classes = dlg.classes()
            self._run_task(
                "转换 VOC XML", lambda cb: __import__(
                    "app.core.format_convert", fromlist=["voc_to_yolo"]
                ).voc_to_yolo(src, Path(src).parent / "yolo_labels", classes, None, cb),
                "转换完成",
            )

    def _convert_coco(self) -> None:
        from app.ui.dataset_dialog import ConvertCocoDialog
        dlg = ConvertCocoDialog(self)
        if dlg.exec() == QDialog.DialogCode.Accepted:
            src = dlg.source_file()
            self._run_task(
                "转换 COCO JSON", lambda cb: __import__(
                    "app.core.format_convert", fromlist=["coco_to_yolo"]
                ).coco_to_yolo(src, Path(src).parent / "yolo_labels", cb),
                "转换完成",
            )

    def _export_visualization(self) -> None:
        if self.dataset is None or self.current_idx < 0:
            QMessageBox.information(self, "提示", "请先打开图片。")
            return
        path = self.images[self.current_idx]
        out, _ = QFileDialog.getSaveFileName(self, "导出可视化图",
                                             Path(path).stem + "_vis.jpg", "图片 (*.jpg *.png)")
        if not out:
            return
        try:
            from app.core.format_convert import export_visualization
            export_visualization(path, self.dataset.labels_dir, out,
                                 self.dataset.meta.classes)
            QMessageBox.information(self, "完成", f"已导出：{out}")
        except Exception as e:
            QMessageBox.critical(self, "错误", f"导出失败：{e}")

    def _export_visualization_batch(self) -> None:
        if self.dataset is None or not self.images:
            QMessageBox.information(self, "提示", "请先打开包含图片的数据集。")
            return
        out_dir = QFileDialog.getExistingDirectory(self, "选择可视化图输出目录")
        if not out_dir:
            return
        self._save_current_annotation()
        self._run_task(
            "批量导出可视化图",
            lambda cb: self._do_batch_export(out_dir, cb),
            f"批量导出完成",
        )

    def _do_batch_export(self, out_dir: str, cb) -> str:
        """后台执行：为每张图片导出带标注的可视化图。"""
        from app.core.format_convert import export_visualization
        total = len(self.images)
        ok = 0
        for i, img in enumerate(self.images):
            try:
                export_visualization(
                    img, self.dataset.labels_dir,
                    Path(out_dir) / (Path(img).stem + ".jpg"),
                    self.dataset.meta.classes,
                )
                ok += 1
            except Exception as e:
                logger.warning("导出失败 %s: %s", img, e)
            if cb and (i % 5 == 0 or i == total - 1):
                cb(i + 1, total)
        return f"成功导出 {ok}/{total} 张 → {out_dir}"

    def _run_task(self, title: str, func, done_msg: str) -> None:
        """通用耗时任务：进度对话框 + 后台线程。"""
        from PySide6.QtCore import QThread, Signal

        class _Task(QThread):
            progress = Signal(int, int)

            def __init__(self):
                super().__init__()
                self.result = None
                self.error = None

            def run(self):
                def report(done, total):
                    if self.isInterruptionRequested():
                        raise InterruptedError
                    self.progress.emit(done, total)

                try:
                    self.result = func(report)
                except InterruptedError:
                    pass
                except Exception as e:
                    self.error = e

        dlg = QProgressDialog(title, "取消", 0, 100, self)
        dlg.setWindowModality(Qt.WindowModal)
        dlg.setMinimumDuration(0)
        task = _Task()
        self._bg_tasks = getattr(self, "_bg_tasks", [])
        self._bg_tasks.append(task)  # 保持引用，防止线程运行中被 GC 回收

        def update_progress(done, total):
            dlg.setRange(0, max(total, 1))
            dlg.setValue(done)

        task.progress.connect(update_progress)
        task.finished.connect(
            lambda: None if task.isInterruptionRequested() else
            (QMessageBox.critical(self, "错误", str(task.error)) if task.error
             else QMessageBox.information(self, "完成", f"{done_msg}\n{task.result or ''}"))
        )
        dlg.canceled.connect(task.requestInterruption)
        task.start()
        dlg.exec()

    # ================= AI / 训练入口 =================
    def _prelabel(self) -> None:
        from app.ui.prelabel_dialog import PrelabelDialog
        if self.dataset is None:
            QMessageBox.information(self, "提示", "请先打开数据集。")
            return
        self._save_current_annotation(quiet=True)
        dlg = PrelabelDialog(self, self.dataset)
        dlg.exec()
        self._labeled_cache = self.dataset.labeled_set()
        for path in self.images:
            self.image_panel.mark_labeled(path, path in self._labeled_cache)
        if self.current_idx >= 0:
            self._load_current_image()

    def _sam_segment(self) -> None:
        from app.ui.sam_dialog import SamDialog
        if self.dataset is None or self.current_idx < 0:
            QMessageBox.information(self, "提示", "请先打开一张图片。")
            return
        dlg = SamDialog(self, self.dataset, self.canvas, self._pose_kpts)
        dlg.exec()

    def _show_train_panel(self) -> None:
        if self.dataset is None:
            QMessageBox.information(self, "提示", "请先打开数据集。")
            return
        self.main_tabs.setCurrentWidget(self.train_panel)

    def _show_test_panel(self) -> None:
        self.main_tabs.setCurrentWidget(self.test_panel)

    def _on_train_running(self, running: bool) -> None:
        """训练状态变化时更新页签文本。"""
        idx = self.main_tabs.indexOf(self.train_panel)
        self.main_tabs.setTabText(idx, "训练中..." if running else "训练")

    # ================= 其他 =================
    def _show_shortcuts(self) -> None:
        lines = [
            "标注工具：",
            "  V - 选择/编辑    W - 矩形框    E - 多边形",
            "  R - 关键点       T - 旋转框(OBB)",
            "",
            "图片浏览：",
            "  A / ← - 上一张    D / → - 下一张    F - 适应窗口",
            "",
            "编辑：",
            "  Ctrl+Z 撤销    Ctrl+Y 重做    Ctrl+C/V 复制粘贴",
            "  Ctrl+D 原位复制选中标注    Delete 删除选中标注",
            "  Ctrl+A 全选    Ctrl+点击 多选   拖动任一选中标注可整体移动",
            "",
            "类别与视图：",
            "  1-9 - 快速切换标注类别    + / - 缩放",
            "  空格或鼠标中键拖拽 - 平移画布    滚轮 - 缩放",
            "  Esc - 取消当前绘制/选择    右键 - 上下文菜单",
            "",
            "绘制：",
            "  多边形/关键点：双击或回车完成，右键或 Esc 取消",
            "  关键点：先拖拽画框，再依次点击关键点",
            "  旋转框：画完后拖动粉色菱形手柄旋转",
        ]
        QMessageBox.information(self, "快捷键说明", "\n".join(lines))

    def closeEvent(self, event) -> None:
        # 训练进行中不允许直接退出（线程销毁会导致崩溃）
        w = getattr(self.train_panel, "_worker", None)
        if w is not None and w.isRunning():
            QMessageBox.warning(
                self, "训练进行中",
                "训练正在进行中，请先到「训练」页点击「停止训练」后再关闭程序。",
            )
            event.ignore()
            return
        self._save_current_annotation()
        self._save_dataset_state()
        # 保存窗口几何
        config.set("window", {"x": self.x(), "y": self.y(),
                              "w": self.width(), "h": self.height()})
        super().closeEvent(event)


def _task_name(t: LabelType) -> str:
    return {LabelType.DETECT: "检测", LabelType.SEGMENT: "分割",
            LabelType.POSE: "关键点", LabelType.OBB: "旋转框"}.get(t, str(t))
