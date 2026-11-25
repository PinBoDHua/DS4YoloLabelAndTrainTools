"""SAM 智能分割对话框：框/点提示交互分割，结果转为多边形标注。"""
from __future__ import annotations

import logging

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QImage, QPainter, QPen, QPixmap
from PySide6.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox,
                               QPushButton, QVBoxLayout, QWidget)

from app.core.label_data import LabelItem, LabelType
from app.core.sam_engine import SamEngine

logger = logging.getLogger(__name__)


class SamPreview(QWidget):
    """SAM 提示交互预览：左键正点/右键负点/拖拽框提示。"""

    hint_changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image: np.ndarray | None = None        # BGR
        self.points: list[list] = []                # [[x, y], ...]
        self.point_labels: list[int] = []           # 1 正 / 0 负
        self.box: list | None = None                # [x1,y1,x2,y2] 原图坐标
        self.mask_overlay: np.ndarray | None = None  # 二值掩码（原图尺寸）
        self._drag_start: QPointF | None = None
        self._scale = 1.0
        self._offset = QPointF(0, 0)
        self._pixmap: QPixmap | None = None

    def set_image(self, image_bgr: np.ndarray) -> None:
        self.image = image_bgr
        # clear_hints 内部会重建像素图
        self.clear_hints()
        self.update()

    def clear_hints(self) -> None:
        self.points = []
        self.point_labels = []
        self.box = None
        self.mask_overlay = None
        # 掩码烘焙在 _pixmap 里，必须重建才能清除显示
        self._update_pixmap()

    def set_mask(self, mask: np.ndarray | None) -> None:
        self.mask_overlay = mask
        # 掩码在 _update_pixmap 中合成进 _pixmap，只 update() 不会生效
        self._update_pixmap()

    def _update_pixmap(self) -> None:
        if self.image is None:
            return
        img = self.image.copy()
        if self.mask_overlay is not None:
            # 掩码叠加：绿色半透明
            overlay = img.copy()
            overlay[self.mask_overlay > 0] = (0, 180, 0)
            img = cv2.addWeighted(img, 0.6, overlay, 0.4, 0)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy()
        self._pixmap = QPixmap.fromImage(qimg)
        self._scale = min(
            max(self.width() - 20, 10) / w,
            max(self.height() - 20, 10) / h,
        )
        self._offset = QPointF((self.width() - w * self._scale) / 2,
                               (self.height() - h * self._scale) / 2)
        self.setMinimumSize(420, 300)
        self.update()

    def resizeEvent(self, event) -> None:
        self._update_pixmap()
        super().resizeEvent(event)

    def paintEvent(self, event) -> None:
        p = QPainter(self)
        p.fillRect(self.rect(), Qt.darkGray)
        if self._pixmap:
            p.drawPixmap(QRectF(self._offset.x(), self._offset.y(),
                                self._pixmap.width() * self._scale,
                                self._pixmap.height() * self._scale),
                         self._pixmap, QRectF(self._pixmap.rect()))
            # 提示点
            r = 4
            for (x, y), label in zip(self.points, self.point_labels):
                sx = self._offset.x() + x * self._scale
                sy = self._offset.y() + y * self._scale
                p.setPen(QPen(Qt.white, 1))
                p.setBrush(Qt.green if label else Qt.red)
                p.drawEllipse(QPointF(sx, sy), r, r)
            # 框提示
            if self.box:
                x1, y1, x2, y2 = self.box
                sx1 = self._offset.x() + x1 * self._scale
                sy1 = self._offset.y() + y1 * self._scale
                sx2 = self._offset.x() + x2 * self._scale
                sy2 = self._offset.y() + y2 * self._scale
                p.setPen(QPen(QColor_blue(), 2, Qt.DashLine))
                p.setBrush(Qt.NoBrush)
                p.drawRect(QRectF(QPointF(sx1, sy1), QPointF(sx2, sy2)))

    def _to_image(self, pos: QPointF) -> list:
        x = (pos.x() - self._offset.x()) / self._scale
        y = (pos.y() - self._offset.y()) / self._scale
        h, w = self.image.shape[:2]
        return [float(np.clip(x, 0, w)), float(np.clip(y, 0, h))]

    def mousePressEvent(self, event) -> None:
        if self.image is None:
            return
        if event.button() == Qt.LeftButton:
            self._drag_start = event.position()
        elif event.button() == Qt.RightButton:
            self.points.append(self._to_image(event.position()))
            self.point_labels.append(0)  # 负点
            self.box = None
            self.hint_changed.emit()
            self.update()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_start is not None:
            dx = event.position().x() - self._drag_start.x()
            dy = event.position().y() - self._drag_start.y()
            if abs(dx) + abs(dy) > 4:
                p1 = self._to_image(self._drag_start)
                p2 = self._to_image(event.position())
                self.box = [min(p1[0], p2[0]), min(p1[1], p2[1]), max(p1[0], p2[0]), max(p1[1], p2[1])]
                self.points = []
                self.point_labels = []
                self.update()

    def mouseReleaseEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self._drag_start is not None:
            if self.box is None:
                # 单击：正点提示
                self.points.append(self._to_image(event.position()))
                self.point_labels.append(1)
            self._drag_start = None
            self.hint_changed.emit()
            self.update()


def QColor_blue():
    from PySide6.QtGui import QColor
    return QColor("#4a9eff")


class SamDialog(QDialog):
    """SAM 分割对话框。"""

    def __init__(self, parent, dataset, canvas, pose_kpts: int = 17):
        super().__init__(parent)
        self.dataset = dataset
        self.canvas = canvas
        self.setWindowTitle("SAM 智能分割")
        self.resize(900, 700)
        self._engine = SamEngine(self)
        self._engine.segment_done.connect(self._on_segment_done)
        self._req_id = 0
        self._last_poly: list | None = None

        layout = QHBoxLayout(self)

        # 左：预览交互区
        left = QVBoxLayout()
        self.preview = SamPreview()
        self.preview.hint_changed.connect(self._on_hint_changed)
        left.addWidget(self.preview)

        self.lb_status = QLabel("加载 SAM 模型中...")
        left.addWidget(self.lb_status)

        btn_row = QHBoxLayout()
        self.btn_clear = QPushButton("清除提示")
        self.btn_clear.clicked.connect(self._clear_hints)
        btn_row.addWidget(self.btn_clear)
        self.btn_add = QPushButton("添加为多边形标注")
        self.btn_add.setProperty("primary", True)
        self.btn_add.setEnabled(False)
        self.btn_add.clicked.connect(self._add_polygon)
        btn_row.addWidget(self.btn_add)
        left.addLayout(btn_row)

        left_widget = QWidget()
        left_widget.setLayout(left)
        layout.addWidget(left_widget, 3)

        # 右：说明
        help_text = QLabel(
            "<b>使用说明</b><br><br>"
            "1. 左键单击：<font color='green'><b>正点提示</b></font>（物体上）<br>"
            "2. 右键单击：<font color='red'><b>负点提示</b></font>（排除区域）<br>"
            "3. 左键拖拽：<b>框提示</b>（框选物体）<br><br>"
            "每次操作后自动分割，绿色区域为预测掩码。<br>"
            "满意后点击「添加为多边形标注」。<br><br>"
            "提示：正负点配合使用可精确修正边界。"
        )
        help_text.setWordWrap(True)
        help_text.setAlignment(Qt.AlignTop)
        layout.addWidget(help_text, 1)

        # 加载当前图片 + 模型
        self._load_image()
        self._engine.ensure_loaded(self._on_loaded)

    # ---------- 图片 ----------
    def _load_image(self) -> None:
        path = self.canvas.current_ann.image_path
        # 用 PIL 读取（numpy fromfile 对中文路径不兼容）
        try:
            pil_img = Image.open(path)
            rgb = np.asarray(pil_img.convert("RGB"))
            img = rgb[:, :, ::-1].copy()  # RGB -> BGR
        except Exception as e:
            logger.warning("SAM 图片读取失败 %s: %s", path, e)
            QMessageBox.warning(self, "警告", f"无法读取图片：{path}")
            self.reject()
            return
        if img is None or img.size == 0:
            QMessageBox.warning(self, "警告", "无法读取图片")
            self.reject()
            return
        self.preview.set_image(img)

    # ---------- SAM ----------
    def _on_loaded(self, ok: bool, err: str) -> None:
        if ok:
            self.lb_status.setText("模型就绪。左键加点 / 拖拽框选 / 右键负点")
        else:
            self.lb_status.setText("模型加载失败")
            QMessageBox.critical(self, "错误", err)

    def _on_hint_changed(self) -> None:
        if not SamEngine.is_loaded():
            self.lb_status.setText("模型加载中，请稍候...")
            return
        self.lb_status.setText("分割中...")
        self._req_id += 1
        self._engine.segment(
            self.preview.image,
            list(self.preview.points),
            list(self.preview.point_labels),
            list(self.preview.box) if self.preview.box else None,
            self._req_id,
        )

    def _on_segment_done(self, req_id: int, polygon) -> None:
        if req_id != self._req_id:
            return
        if polygon:
            self._last_poly = polygon
            # 渲染掩码叠加
            mask = np.zeros(self.preview.image.shape[:2], dtype=np.uint8)
            pts = np.array([[int(x), int(y)] for x, y in polygon], dtype=np.int32)
            cv2.fillPoly(mask, [pts], 1)
            self.preview.set_mask(mask)
            self.btn_add.setEnabled(True)
            self.lb_status.setText(f"分割完成（{len(polygon)} 个顶点）。可继续加点修正或添加标注")
        else:
            # 新请求无结果：清空旧多边形/掩码，避免残留结果被误添加
            self._last_poly = None
            self.btn_add.setEnabled(False)
            self.preview.set_mask(None)
            self.lb_status.setText("未检测到目标，请调整提示")

    def _clear_hints(self) -> None:
        # 作废在途请求：否则其完成后 req_id 仍匹配，旧结果会重新出现
        self._req_id += 1
        self.preview.clear_hints()
        self.btn_add.setEnabled(False)
        self._last_poly = None
        self.lb_status.setText("已清除提示")

    # ---------- 添加到标注 ----------
    def _add_polygon(self) -> None:
        if not self._last_poly:
            return
        self.canvas.edit_begin.emit()
        lb = LabelItem(LabelType.SEGMENT, self.canvas.current_class_id,
                       points=self._last_poly)
        item = self.canvas.add_shape_item(lb)
        item.setSelected(True)
        self.canvas.labels_changed.emit()
        QMessageBox.information(self, "完成",
                                "已添加多边形标注。\n可以继续分割更多目标，或关闭本窗口回画布编辑。")
        self._clear_hints()

    def done(self, result: int) -> None:
        """关闭前停止后台线程，避免线程运行时被销毁崩溃。"""
        try:
            self._engine.shutdown()
        except Exception:
            pass
        super().done(result)
