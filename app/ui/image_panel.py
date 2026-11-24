"""图片列表面板：缩略图浏览 + 已标注/未标注过滤。"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QSize, Qt, QThread, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit,
                               QListWidget, QListWidgetItem, QVBoxLayout,
                               QWidget)

from app.theme import theme

logger = logging.getLogger(__name__)


class _ThumbLoader(QThread):
    """后台缩略图加载线程。

    注意：QPixmap 只能在 GUI 线程创建（非 GUI 线程创建会导致硬崩溃），
    因此线程内只产生 QImage（线程安全），由主线程槽转为 QPixmap。
    """
    loaded = Signal(str, QImage)  # (图片路径, 缩略图 QImage)

    def __init__(self, image_paths: list[str], size: int = 96):
        super().__init__()
        self._paths = image_paths
        self._size = size
        self._cancel = False

    def run(self) -> None:
        for p in self._paths:
            if self._cancel:
                return
            try:
                img = QImage(p)
                if img.isNull():
                    continue
                thumb = img.scaled(
                    self._size, self._size,
                    Qt.KeepAspectRatio, Qt.SmoothTransformation,
                )
                self.loaded.emit(p, thumb)
            except Exception as e:
                logger.debug("缩略图加载失败 %s: %s", p, e)


class ImagePanel(QWidget):
    """右侧图片缩略图面板。"""

    image_selected = Signal(str)   # 图片绝对路径
    stats_loaded = Signal(int, int)  # (总数, 已标注数)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._images: list[str] = []
        self._labeled: set[str] = set()
        self._filter = "all"
        self._search = ""
        self._loader: _ThumbLoader | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        # 过滤栏
        bar = QHBoxLayout()
        self.filter_combo = QComboBox()
        self.filter_combo.addItem("全部图片", "all")
        self.filter_combo.addItem("未标注", "unlabeled")
        self.filter_combo.addItem("已标注", "labeled")
        self.filter_combo.currentIndexChanged.connect(self._on_filter)
        bar.addWidget(self.filter_combo)
        layout.addLayout(bar)

        # 搜索框
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("搜索图片名称...")
        self.search_edit.setClearButtonEnabled(True)
        self.search_edit.textChanged.connect(self._on_search)
        layout.addWidget(self.search_edit)

        # 统计行
        self.count_label = QLabel("0 张")
        layout.addWidget(self.count_label)

        # 缩略图列表
        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(96, 96))
        self.list_widget.setViewMode(QListWidget.IconMode)
        self.list_widget.setResizeMode(QListWidget.Adjust)
        self.list_widget.setSpacing(4)
        self.list_widget.setUniformItemSizes(True)
        self.list_widget.setWordWrap(True)
        self.list_widget.itemClicked.connect(self._on_clicked)
        self.list_widget.itemDoubleClicked.connect(self._on_clicked)
        # 键盘在列表中移动/回车时同步加载图片（select_path 已 blockSignals，不受影响）
        self.list_widget.itemActivated.connect(self._on_clicked)
        self.list_widget.currentItemChanged.connect(self._on_current_changed)
        layout.addWidget(self.list_widget)

        # 面板销毁时停止缩略图线程，避免退出时线程崩溃
        self.destroyed.connect(self._shutdown)

    def _shutdown(self) -> None:
        if self._loader is None:
            return
        loader = self._loader
        if loader.isRunning():
            loader._cancel = True
            loader.wait()
        try:
            loader.loaded.disconnect(self._on_thumb)
        except Exception:
            pass
        self._loader = None

    # ---------- 数据 ----------
    def set_images(self, paths: list[str], labeled: set[str]) -> None:
        """设置图片列表与已标注集合。"""
        self._images = paths
        self._labeled = labeled
        self._refresh()

    def mark_labeled(self, path: str, labeled: bool) -> None:
        if labeled:
            self._labeled.add(path)
        else:
            self._labeled.discard(path)
        # 更新对应 item 的角标
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == path:
                item.setData(Qt.UserRole + 1, labeled)
                self._restyle_item(item)
                break

    def _refresh(self) -> None:
        self.list_widget.clear()
        filtered = self._apply_filter()
        labeled_n = len([p for p in filtered if p in self._labeled])
        self.count_label.setText(f"{len(filtered)} 张（已标注 {labeled_n}）")
        for p in filtered:
            item = QListWidgetItem(Path(p).stem)
            item.setData(Qt.UserRole, p)
            item.setData(Qt.UserRole + 1, p in self._labeled)
            self._restyle_item(item)
            self.list_widget.addItem(item)
        # 停止旧线程：请求取消并等待退出（避免旧线程运行时被 GC 销毁导致崩溃）
        if self._loader is not None:
            old = self._loader
            if old.isRunning():
                old._cancel = True
                old.wait()
            old.loaded.disconnect(self._on_thumb)
        # 后台加载缩略图
        self._loader = _ThumbLoader(filtered)
        self._loader.loaded.connect(self._on_thumb)
        self._loader.start()

    def _apply_filter(self) -> list[str]:
        imgs = self._images
        if self._filter == "unlabeled":
            imgs = [p for p in imgs if p not in self._labeled]
        elif self._filter == "labeled":
            imgs = [p for p in imgs if p in self._labeled]
        if self._search:
            needle = self._search.lower()
            imgs = [p for p in imgs if needle in Path(p).stem.lower()]
        return imgs

    def _restyle_item(self, item: QListWidgetItem) -> None:
        labeled = bool(item.data(Qt.UserRole + 1))
        item.setToolTip(item.data(Qt.UserRole))
        c = theme.colors
        item.setForeground(QColor(c.success) if labeled else QColor(c.text_dim))

    # ---------- 事件 ----------
    def _on_thumb(self, path: str, image: QImage) -> None:
        # GUI 线程中转换为 QPixmap
        pixmap = QPixmap.fromImage(image)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == path:
                item.setIcon(QPixmap())
                item.setIcon(pixmap)
                break

    def _on_clicked(self, item: QListWidgetItem) -> None:
        self.image_selected.emit(item.data(Qt.UserRole))

    def _on_current_changed(self, cur, _prev) -> None:
        """键盘上下键在缩略图列表移动时同步加载图片。"""
        if cur is not None:
            self.image_selected.emit(cur.data(Qt.UserRole))

    def _on_filter(self, index: int) -> None:
        self._filter = self.filter_combo.itemData(index)
        self._refresh()

    def _on_search(self, text: str) -> None:
        self._search = text.strip()
        self._refresh()

    def select_path(self, path: str) -> None:
        """高亮指定路径（不触发信号）。"""
        self.list_widget.blockSignals(True)
        for i in range(self.list_widget.count()):
            item = self.list_widget.item(i)
            if item.data(Qt.UserRole) == path:
                self.list_widget.setCurrentItem(item)
                self.list_widget.scrollToItem(item)
                break
        self.list_widget.blockSignals(False)
