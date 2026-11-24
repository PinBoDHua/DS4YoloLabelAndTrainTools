"""标注画布：图片显示、缩放平移、标注形状渲染与编辑、工具交互。"""
from __future__ import annotations

import logging
from typing import Callable

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import (QBrush, QColor, QImage, QPainter, QPainterPath,
                           QPen, QPixmap, QPolygonF)
from PySide6.QtWidgets import (QGraphicsItem, QGraphicsPixmapItem,
                               QGraphicsScene, QGraphicsView)

from app.core.label_data import CLASS_COLORS, ImageAnnotation, LabelItem, LabelType
from app.theme import theme

logger = logging.getLogger(__name__)


def _sub(a: QPointF, b: QPointF) -> QPointF:
    """QPointF 减法（PySide6 6.11 的 Qt 运算符绑定有 bug，统一用分量运算）。"""
    return QPointF(a.x() - b.x(), a.y() - b.y())


def _mul(a: QPointF, f: float) -> QPointF:
    """QPointF 数乘。"""
    return QPointF(a.x() * f, a.y() * f)

# COCO 17 关键点标准骨架连线（索引对）
COCO_SKELETON = [
    [15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12],
    [5, 6], [5, 7], [6, 8], [7, 9], [8, 10], [1, 2], [0, 1], [0, 2],
    [1, 3], [2, 4], [3, 5], [4, 6],
]


def class_color(class_id: int) -> QColor:
    return QColor(CLASS_COLORS[class_id % len(CLASS_COLORS)])


class ShapeItem(QGraphicsItem):
    """单个标注形状。几何直接读写 label.points（绝对像素坐标）。

    编辑交互（在 Select 工具下）：
      - 拖拽 body 移动整个形状
      - 拖拽手柄修改顶点/调整矩形/旋转 OBB
    """

    # 矩形/pose 的 8 个 resize 手柄: id -> (作用描述)
    _RECT_RESIZE = {
        "tl": ("corner", 0), "tr": ("corner_tr", None), "bl": ("corner_bl", None),
        "br": ("corner", 1), "tm": ("edge", "y1"), "bm": ("edge", "y2"),
        "ml": ("edge", "x1"), "mr": ("edge", "x2"),
    }

    def __init__(self, label: LabelItem, color: QColor | None = None,
                 show_label: bool = True, skeleton: list | None = None):
        super().__init__()
        self.label = label
        self.color = color or class_color(label.class_id)
        self.show_label = show_label
        self.skeleton = skeleton  # pose 骨骼连线
        self._hover_handle: str | None = None
        self._drag_handle: str | None = None
        self._drag_offset: QPointF | None = None
        # 编辑回调（QGraphicsItem 非 QObject，用回调）
        self.on_edit_begin: Callable | None = None
        self.on_edited: Callable | None = None

        self.setFlags(
            QGraphicsItem.ItemIsSelectable
            | QGraphicsItem.ItemSendsGeometryChanges
        )
        self.setAcceptHoverEvents(True)
        self.setZValue(10)

    # ---------- 几何辅助 ----------
    def _view_scale(self, painter: QPainter | None = None) -> float:
        """视图缩放因子（用于恒定屏幕尺寸的线宽/手柄）。"""
        try:
            if painter is not None:
                t = painter.transform()
                return max(abs(t.m11()), abs(t.m22())) or 1.0
            t = self.scene().views()[0].transform() if self.scene() and self.scene().views() else None
            if t is not None:
                return max(abs(t.m11()), abs(t.m22())) or 1.0
        except Exception:
            pass
        return 1.0

    def handles(self) -> dict[str, QPointF]:
        """当前形状的全部手柄：id -> 场景坐标。"""
        h = {}
        pts = self.label.points
        if self.label.type in (LabelType.DETECT, LabelType.POSE):
            x1, y1, x2, y2 = self.label.rect
            h = {
                "tl": QPointF(x1, y1), "tr": QPointF(x2, y1),
                "bl": QPointF(x1, y2), "br": QPointF(x2, y2),
                "tm": QPointF((x1 + x2) / 2, y1), "bm": QPointF((x1 + x2) / 2, y2),
                "ml": QPointF(x1, (y1 + y2) / 2), "mr": QPointF(x2, (y1 + y2) / 2),
            }
            if self.label.type == LabelType.POSE:
                for i, (kx, ky, _v) in enumerate(self.label.keypoints):
                    h[f"k{i}"] = QPointF(kx, ky)
        else:
            for i, (x, y) in enumerate(pts):
                h[f"v{i}"] = QPointF(x, y)
            if self.label.type == LabelType.OBB and len(pts) >= 4:
                # 旋转手柄：上方边缘中点外延
                top = QPointF((pts[0][0] + pts[1][0]) / 2, (pts[0][1] + pts[1][1]) / 2)
                bottom = QPointF((pts[2][0] + pts[3][0]) / 2, (pts[2][1] + pts[3][1]) / 2)
                dx, dy = bottom.x() - top.x(), bottom.y() - top.y()
                ln = (dx * dx + dy * dy) ** 0.5 or 1.0
                h["rot"] = QPointF(top.x() - dx / ln * 24.0, top.y() - dy / ln * 24.0)
        return h

    def handle_at(self, pos: QPointF) -> str | None:
        tol = 8.0 / self._view_scale()
        for hid, hpos in self.handles().items():
            if abs(hpos.x() - pos.x()) + abs(hpos.y() - pos.y()) <= tol * 1.6:
                return hid
        return None

    def _set_rect(self, x1: float, y1: float, x2: float, y2: float) -> None:
        self.label.points[0] = [min(x1, x2), min(y1, y2)]
        self.label.points[1] = [max(x1, x2), max(y1, y2)]

    def apply_handle_move(self, hid: str, pos: QPointF) -> None:
        """拖动手柄 hid 到 pos，更新几何。"""
        t = self.label.type
        if t in (LabelType.DETECT, LabelType.POSE):
            x1, y1, x2, y2 = self.label.rect
            kind, arg = self._RECT_RESIZE[hid]
            if kind == "corner":
                if arg == 0:
                    self._set_rect(pos.x(), pos.y(), x2, y2)
                else:
                    self._set_rect(x1, y1, pos.x(), pos.y())
            elif kind == "corner_tr":
                self._set_rect(x1, pos.y(), pos.x(), y2)
            elif kind == "corner_bl":
                self._set_rect(pos.x(), y1, x2, pos.y())
            elif arg == "y1":
                self._set_rect(x1, pos.y(), x2, y2)
            elif arg == "y2":
                self._set_rect(x1, y1, x2, pos.y())
            elif arg == "x1":
                self._set_rect(pos.x(), y1, x2, y2)
            elif arg == "x2":
                self._set_rect(x1, y1, pos.x(), y2)
        elif hid.startswith("v"):
            idx = int(hid[1:])
            if 0 <= idx < len(self.label.points):
                self.label.points[idx] = [pos.x(), pos.y()]
        elif hid.startswith("k"):
            idx = int(hid[1:])
            if 0 <= idx < len(self.label.keypoints):
                k = self.label.keypoints[idx]
                self.label.keypoints[idx] = [pos.x(), pos.y(), k[2]]
        elif hid == "rot":
            # 绕 OBB 中心旋转
            cx = sum(p[0] for p in self.label.points) / len(self.label.points)
            cy = sum(p[1] for p in self.label.points) / len(self.label.points)
            import math
            a1 = math.atan2(pos.y() - cy, pos.x() - cx)
            # 当前旋转角（以第一个顶点方向为基准）
            p0 = self.label.points[0]
            a0 = math.atan2(p0[1] - cy, p0[0] - cx)
            da = a1 - a0
            # 上方中点手柄代表 -90 方向，修正基准
            da += math.pi / 2
            pts = []
            for x, y in self.label.points:
                dx, dy = x - cx, y - cy
                nx = cx + dx * math.cos(da) - dy * math.sin(da)
                ny = cy + dx * math.sin(da) + dy * math.cos(da)
                pts.append([nx, ny])
            self.label.points = pts
        self.prepareGeometryChange()
        self.update()

    def move_by(self, dx: float, dy: float) -> None:
        for p in self.label.points:
            p[0] += dx
            p[1] += dy
        for k in self.label.keypoints:
            k[0] += dx
            k[1] += dy
        self.prepareGeometryChange()
        self.update()

    # ---------- 绘制 ----------
    def boundingRect(self) -> QRectF:
        r = QRectF()
        for x, y in self.label.points:
            r = r.united(QRectF(x - 6, y - 6, 12, 12))
        for kx, ky, _v in self.label.keypoints:
            r = r.united(QRectF(kx - 6, ky - 6, 12, 12))
        return r

    def _outline_path(self) -> QPainterPath:
        path = QPainterPath()
        pts = self.label.points
        t = self.label.type
        if t in (LabelType.DETECT, LabelType.POSE):
            x1, y1, x2, y2 = self.label.rect
            path.addRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)))
        elif len(pts) >= 3:
            poly = QPolygonF([QPointF(x, y) for x, y in pts])
            path.addPolygon(poly)
            path.closeSubpath()
        return path

    def shape(self) -> QPainterPath:
        """命中检测路径：body + 手柄区域。"""
        path = self._outline_path()
        s = 10.0 / self._view_scale()
        for hp in self.handles().values():
            path.addEllipse(hp, s, s)
        return path

    def paint(self, painter: QPainter, option, widget=None) -> None:
        s = self._view_scale(painter)
        lw = max(2.0 / s, 1.5)
        t = self.label.type
        c = self.color
        fill = QColor(c)
        fill.setAlpha(36)
        pen = QPen(c, lw, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        pen_selected = QPen(QColor("#ffb100"), lw + 1.5 / s, Qt.DashLine)

        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setBrush(fill)
        painter.setPen(pen_selected if self.isSelected() else pen)

        if t in (LabelType.DETECT, LabelType.POSE):
            x1, y1, x2, y2 = self.label.rect
            painter.drawRect(QRectF(QPointF(x1, y1), QPointF(x2, y2)))
        else:
            poly = QPolygonF([QPointF(x, y) for x, y in self.label.points])
            if len(poly) >= 3:
                painter.drawPolygon(poly)

        # 关键点与骨骼
        if t == LabelType.POSE:
            painter.setPen(pen)
            kpts = self.label.keypoints
            if self.skeleton:
                skel_pen = QPen(QColor("#00e5c3"), lw, Qt.SolidLine)
                painter.setPen(skel_pen)
                for i, j in self.skeleton:
                    if i < len(kpts) and j < len(kpts) and kpts[i][2] > 0 and kpts[j][2] > 0:
                        painter.drawLine(QPointF(kpts[i][0], kpts[i][1]),
                                         QPointF(kpts[j][0], kpts[j][1]))
                painter.setPen(pen)
            painter.setBrush(QBrush(c))
            for i, (kx, ky, v) in enumerate(kpts):
                r = 4.5 / s
                if v == 0:
                    continue
                if v == 1:
                    painter.setBrush(Qt.NoBrush)
                painter.drawEllipse(QPointF(kx, ky), r, r)
                painter.setBrush(QBrush(c))

        # 类别文本
        if self.show_label:
            label_text = f" {self.label.class_id} "
            font = painter.font()
            font.setPixelSize(max(12, int(13 / s ** 0.6)))
            painter.setFont(font)
            fm = painter.fontMetrics()
            tw = fm.horizontalAdvance(label_text)
            th = fm.height()
            anchor = QPointF(*self.label.points[0])
            tr = QRectF(anchor.x(), anchor.y() - th - 4 / s, tw + 6, th + 4)
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(c))
            painter.drawRoundedRect(tr, 3, 3)
            painter.setPen(QPen(Qt.white))
            painter.drawText(tr, Qt.AlignCenter, label_text)

        # 选中手柄
        if self.isSelected():
            self._paint_handles(painter, s)

    def _paint_handles(self, painter: QPainter, s: float) -> None:
        size = 7.0 / s
        painter.setPen(QPen(Qt.white, 1.5 / s))
        painter.setBrush(QBrush(QColor("#ffb100")))
        for hid, hp in self.handles().items():
            if hid == "rot":
                painter.setBrush(QBrush(QColor("#ff6ec7")))
                painter.setPen(QPen(Qt.white, 1.5 / s))
                # 旋转手柄画成菱形
                d = QPolygonF([
                    QPointF(hp.x(), hp.y() - size * 1.4),
                    QPointF(hp.x() + size * 1.4, hp.y()),
                    QPointF(hp.x(), hp.y() + size * 1.4),
                    QPointF(hp.x() - size * 1.4, hp.y()),
                ])
                painter.drawPolygon(d)
                painter.setBrush(QBrush(QColor("#ffb100")))
                painter.setPen(QPen(Qt.white, 1.5 / s))
            else:
                painter.drawRect(QRectF(hp.x() - size, hp.y() - size, size * 2, size * 2))

    # ---------- 交互 ----------
    def mousePressEvent(self, event) -> None:
        if self._drag_handle is None and event.button() == Qt.LeftButton:
            h = self.handle_at(event.pos())
            if h:
                self._drag_handle = h
                if self.on_edit_begin:
                    self.on_edit_begin()
            else:
                # 开始移动 body
                self._drag_offset = _sub(event.pos(), QPointF(*self.label.points[0]))
                if self.on_edit_begin:
                    self.on_edit_begin()
            event.accept()
        elif event.button() == Qt.RightButton and self.label.type == LabelType.POSE:
            # 右键关键点循环可见性
            h = self.handle_at(event.pos())
            if h and h.startswith("k"):
                idx = int(h[1:])
                k = self.label.keypoints[idx]
                k[2] = 2 if k[2] != 2 else 1
                if self.on_edit_begin:
                    self.on_edit_begin()
                if self.on_edited:
                    self.on_edited()
                self.update()
                event.accept()
                return
            event.ignore()
        else:
            event.ignore()

    def mouseMoveEvent(self, event) -> None:
        if self._drag_handle:
            self.apply_handle_move(self._drag_handle, event.pos())
        elif self._drag_offset is not None:
            target = _sub(event.pos(), self._drag_offset)
            cur = QPointF(*self.label.points[0])
            self.move_by(target.x() - cur.x(), target.y() - cur.y())
        else:
            self._hover_handle = self.handle_at(event.pos())
            self.update()
            event.ignore()
            return
        event.accept()

    def mouseReleaseEvent(self, event) -> None:
        if self._drag_handle or self._drag_offset is not None:
            self._drag_handle = None
            self._drag_offset = None
            if self.on_edited:
                self.on_edited()
            event.accept()
        else:
            event.ignore()

    def hoverMoveEvent(self, event) -> None:
        h = self.handle_at(event.pos())
        if h != self._hover_handle:
            self._hover_handle = h
            self.update()
        super().hoverMoveEvent(event)


class TempPreviewItem(QGraphicsItem):
    """绘制中的临时预览形状。"""

    def __init__(self, label_type: LabelType, color: QColor):
        super().__init__()
        self.label_type = label_type
        self.color = color
        self.points: list[QPointF] = []
        self.cur: QPointF | None = None  # 当前鼠标位置（橡皮筋）
        self.kpts: list[QPointF] = []
        self.setZValue(100)

    def boundingRect(self) -> QRectF:
        r = QRectF()
        for p in self.points + (self.kpts or []) + ([self.cur] if self.cur else []):
            r = r.united(QRectF(p.x() - 8, p.y() - 8, 16, 16))
        return r

    def paint(self, painter: QPainter, option, widget=None) -> None:
        try:
            t = painter.transform()
            s = max(abs(t.m11()), abs(t.m22())) or 1.0
        except Exception:
            s = 1.0
        lw = max(2.0 / s, 1.5)
        pen = QPen(self.color, lw, Qt.DashLine)
        fill = QColor(self.color)
        fill.setAlpha(30)
        painter.setRenderHint(QPainter.Antialiasing, True)
        painter.setPen(pen)
        painter.setBrush(fill)

        if self.label_type in (LabelType.DETECT, LabelType.OBB, LabelType.POSE) and len(self.points) >= 1:
            p0 = self.points[0]
            p1 = self.cur or p0
            painter.drawRect(QRectF(p0, p1))
        else:  # 多边形
            poly = QPolygonF(self.points + ([self.cur] if self.cur else []))
            if len(poly) >= 2:
                painter.drawPolyline(poly)
            if len(poly) >= 3:
                painter.drawPolygon(poly)
            painter.setBrush(QBrush(self.color))
            r = 4.0 / s
            for p in self.points:
                painter.drawEllipse(p, r, r)
        # 关键点预览
        if self.label_type == LabelType.POSE:
            painter.setPen(QPen(QColor("#00e5c3"), lw))
            painter.setBrush(QBrush(QColor("#00e5c3")))
            r = 4.0 / s
            for p in self.kpts:
                painter.drawEllipse(p, r, r)


class LabelCanvas(QGraphicsView):
    """标注画布视图：缩放、平移、工具派发、撤销信号。"""

    labels_changed = Signal()          # 标注内容变化（几何/增删/改类）
    edit_begin = Signal()              # 编辑开始（撤销快照时机）
    mouse_moved = Signal(QPointF)      # 场景坐标（像素）
    shape_selected = Signal(object)    # 选中的 ShapeItem（可空）
    polygon_point_added = Signal()     # 多边形/关键点加点（用于画布状态提示）
    tool_auto_changed = Signal(str)    # 工具自动切换（如 OBB 画完自动转选择）
    zoom_changed = Signal(float)       # 缩放比例变化
    context_menu_requested = Signal(object)  # 右键菜单请求（携带原始事件）
    navigate_requested = Signal(int)   # ←/→ 键请求切换图片（delta: -1/+1）
    class_key = Signal(int)            # 数字键 1-9 请求切换当前类别

    def __init__(self, parent=None):
        super().__init__(parent)
        self.scene = QGraphicsScene(self)
        self.setScene(self.scene)
        self.image_item: QGraphicsPixmapItem | None = None
        self.shapes: list[ShapeItem] = []
        self.temp_item: TempPreviewItem | None = None
        self.tool = None  # 由外部 tools 模块注入
        self.classes: list[str] = []
        self.current_class_id = 0
        self.show_labels = True
        self.skeleton: list | None = None

        self._panning = False
        self._pan_start = QPointF()
        self._zoom = 1.0
        self.crosshair = False

        self.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        self.setTransformationAnchor(QGraphicsView.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.AnchorViewCenter)
        self.setDragMode(QGraphicsView.NoDrag)
        self.setMouseTracking(True)
        self.setViewportUpdateMode(QGraphicsView.FullViewportUpdate)
        self.scene.selectionChanged.connect(self._on_selection_changed)

        # 十字准星（跟随鼠标的两条虚线，随主题着色）
        self._cross_pen = QPen(QColor(theme.colors.canvas_grid), 1, Qt.DashLine)
        self._cross_pen.setCosmetic(True)
        self._cross_v = self.scene.addLine(0, 0, 0, 0, self._cross_pen)
        self._cross_h = self.scene.addLine(0, 0, 0, 0, self._cross_pen)
        for ln in (self._cross_v, self._cross_h):
            ln.setZValue(999)
            ln.setVisible(False)

        # 颜色（跟随主题）
        theme.theme_changed.connect(self._refresh_background)
        self._refresh_background()

    def _refresh_background(self, _name: str = "") -> None:
        self.setBackgroundBrush(QColor(theme.colors.canvas_bg))
        self._cross_pen.setColor(QColor(theme.colors.canvas_grid))
        if self._cross_v is not None:
            self._cross_v.setPen(self._cross_pen)
            self._cross_h.setPen(self._cross_pen)
        self.viewport().update()

    def set_crosshair(self, enabled: bool) -> None:
        """开关跟随鼠标的十字准星。"""
        self.crosshair = enabled
        if not enabled:
            self._cross_v.setVisible(False)
            self._cross_h.setVisible(False)

    # ---------- 内容管理 ----------
    def set_image(self, image: QImage | QPixmap) -> None:
        """替换当前图片，重置视图。"""
        pix = image if isinstance(image, QPixmap) else QPixmap.fromImage(image)
        if self.image_item is None:
            self.image_item = QGraphicsPixmapItem(pix)
            self.image_item.setZValue(-10)
            self.scene.addItem(self.image_item)
        else:
            self.image_item.setPixmap(pix)
        self.scene.setSceneRect(QRectF(pix.rect()))
        if self.viewport().width() < 50:
            # 视口尚未布局（页签未显示等）：延迟到布局完成后适配窗口
            from PySide6.QtCore import QTimer
            QTimer.singleShot(0, self.reset_zoom)
        else:
            self.reset_zoom()

    def reset_zoom(self) -> None:
        """适配窗口显示整图。"""
        if not self.image_item:
            return
        self.resetTransform()
        r = self.image_item.pixmap().rect()
        if r.width() > 10 and r.height() > 10:
            vw = max(self.viewport().width() - 8, 10)
            vh = max(self.viewport().height() - 8, 10)
            self._zoom = min(vw / r.width(), vh / r.height(), 3.0)
            self.scale(self._zoom, self._zoom)
        else:
            self._zoom = 1.0
        self.zoom_changed.emit(self._zoom)

    def clear_shapes(self) -> None:
        for s in self.shapes:
            self.scene.removeItem(s)
        self.shapes = []

    def set_annotations(self, ann: ImageAnnotation) -> None:
        self.clear_shapes()
        self._cancel_temp()  # 清理上一张图的绘制残留
        self.current_ann = ann
        for lb in ann.labels:
            self.add_shape_item(lb)

    def add_shape_item(self, label: LabelItem) -> ShapeItem:
        item = ShapeItem(label, show_label=self.show_labels, skeleton=self.skeleton)
        item.on_edit_begin = lambda: self.edit_begin.emit()
        item.on_edited = lambda: self.labels_changed.emit()
        self.scene.addItem(item)
        self.shapes.append(item)
        return item

    def remove_shape_item(self, item: ShapeItem) -> None:
        if item in self.shapes:
            self.shapes.remove(item)
        self.scene.removeItem(item)

    def get_annotation(self) -> ImageAnnotation:
        """从当前画布收集标注。"""
        ann = ImageAnnotation(image_path=getattr(self, "current_ann", None) and
                              self.current_ann.image_path or "")
        for s in self.shapes:
            ann.labels.append(s.label)
        return ann

    def collect_shapes(self) -> list[ShapeItem]:
        return list(self.shapes)

    # ---------- 选择/编辑 ----------
    def _on_selection_changed(self) -> None:
        sel = self.scene.selectedItems()
        self.shape_selected.emit(sel[0] if sel else None)

    def selected_shape(self) -> ShapeItem | None:
        sel = self.scene.selectedItems()
        return sel[0] if sel and isinstance(sel[0], ShapeItem) else None

    def delete_selected(self) -> None:
        sel = [i for i in self.scene.selectedItems() if isinstance(i, ShapeItem)]
        if not sel:
            return
        self.edit_begin.emit()
        for item in sel:
            self.remove_shape_item(item)
        self.labels_changed.emit()

    def set_class_of_selected(self, class_id: int) -> None:
        sel = self.scene.selectedItems()
        changed = False
        for item in sel:
            if isinstance(item, ShapeItem):
                if item.label.class_id != class_id:
                    if not changed:
                        self.edit_begin.emit()
                    item.label.class_id = class_id
                    item.color = class_color(class_id)
                    item.update()
                    changed = True
        if changed:
            self.labels_changed.emit()

    def select_all(self) -> None:
        for s in self.shapes:
            s.setSelected(True)

    def clear_selection(self) -> None:
        self.scene.clearSelection()

    # ---------- 复制粘贴 ----------
    def copy_selected_labels(self) -> list[dict]:
        sel = self.scene.selectedItems()
        return [i.label.to_dict() for i in sel if isinstance(i, ShapeItem)]

    def duplicate_selected(self, offset: float = 20.0) -> None:
        """原位复制选中的标注（偏移一点，便于区分）。"""
        dicts = self.copy_selected_labels()
        if not dicts:
            return
        self.paste_labels(dicts, offset)

    def paste_labels(self, dicts: list[dict], offset: float = 20.0) -> None:
        if not dicts:
            return
        self.edit_begin.emit()
        self.scene.clearSelection()
        for d in dicts:
            lb = LabelItem.from_dict(d)
            for p in lb.points:
                p[0] += offset
                p[1] += offset
            for k in lb.keypoints:
                k[0] += offset
                k[1] += offset
            item = self.add_shape_item(lb)
            item.setSelected(True)
        self.labels_changed.emit()

    # ---------- 工具管理 ----------
    def set_tool(self, tool) -> None:
        """切换交互工具。tool 为 None 时回落到选择工具。"""
        from app.ui import tools as tools_mod
        if self.tool is not None:
            self.tool.deactivate()  # 清理旧工具的绘制草稿
        self.tool = tool if tool is not None else tools_mod.SelectTool(self)
        self.clear_selection()
        self._cancel_temp()
        self.setCursor(self.tool.cursor())

    def _cancel_temp(self) -> None:
        if self.temp_item:
            self.scene.removeItem(self.temp_item)
            self.temp_item = None

    def create_temp(self, label_type: LabelType, color: QColor) -> TempPreviewItem:
        self._cancel_temp()
        self.temp_item = TempPreviewItem(label_type, color)
        self.scene.addItem(self.temp_item)
        return self.temp_item

    def finish_temp(self, label: LabelItem | None) -> None:
        """临时预览转正式标注（label 为 None 则取消）。"""
        self._cancel_temp()
        if label is not None:
            self.edit_begin.emit()
            item = self.add_shape_item(label)
            item.setSelected(True)
            self.labels_changed.emit()

    # ---------- 视图交互 ----------
    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        new_zoom = self._zoom * factor
        if 0.02 < new_zoom < 40:
            self._zoom = new_zoom
            self.scale(factor, factor)
            self.zoom_changed.emit(self._zoom)
        event.accept()

    def zoom_by(self, factor: float) -> None:
        """以视图中心为锚缩放（工具栏 +/- 按钮用）。"""
        new_zoom = self._zoom * factor
        if 0.02 < new_zoom < 40:
            self._zoom = new_zoom
            old = self.transform().m11()
            self.scale(new_zoom / old if old else 1.0, new_zoom / old if old else 1.0)
            self.zoom_changed.emit(self._zoom)

    def mousePressEvent(self, event) -> None:
        # 中键 或 空格+左键：平移
        if event.button() == Qt.MiddleButton:
            self._panning = True
            self._pan_start = event.position()
            self.setCursor(Qt.ClosedHandCursor)
            event.accept()
            return
        if self.tool is not None:
            self.tool.mousePress(event)
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event) -> None:
        if self._panning:
            new_pos = event.position()
            delta_x = new_pos.x() - self._pan_start.x()
            delta_y = new_pos.y() - self._pan_start.y()
            self._pan_start = new_pos
            self.horizontalScrollBar().setValue(self.horizontalScrollBar().value() - int(delta_x))
            self.verticalScrollBar().setValue(self.verticalScrollBar().value() - int(delta_y))
            event.accept()
            return
        pos = self.mapToScene(event.position().toPoint())
        self.mouse_moved.emit(pos)
        self._update_crosshair(pos)
        if self.tool is not None:
            self.tool.mouseMove(event)
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._panning and event.button() == Qt.MiddleButton:
            self._panning = False
            self.setCursor(Qt.ArrowCursor)
            event.accept()
            return
        if self.tool is not None:
            self.tool.mouseRelease(event)
        else:
            super().mouseReleaseEvent(event)

    def mouseDoubleClickEvent(self, event) -> None:
        if self.tool is not None:
            self.tool.mouseDoubleClick(event)
        else:
            super().mouseDoubleClickEvent(event)

    def leaveEvent(self, event) -> None:
        """鼠标离开画布时隐藏十字准星。"""
        self._cross_v.setVisible(False)
        self._cross_h.setVisible(False)
        super().leaveEvent(event)

    def _update_crosshair(self, pos: QPointF) -> None:
        if not self.crosshair or self.image_item is None:
            self._cross_v.setVisible(False)
            self._cross_h.setVisible(False)
            return
        r = self.image_item.pixmap().rect()
        self._cross_v.setLine(pos.x(), 0, pos.x(), r.height())
        self._cross_h.setLine(0, pos.y(), r.width(), pos.y())
        self._cross_v.setVisible(True)
        self._cross_h.setVisible(True)

    def contextMenuEvent(self, event) -> None:
        """右键菜单：转发给主窗口构建菜单。"""
        self.context_menu_requested.emit(event)
        event.accept()

    def keyPressEvent(self, event) -> None:
        # 数字键 1-9 切换当前类别
        if event.key() in (Qt.Key_1, Qt.Key_2, Qt.Key_3, Qt.Key_4, Qt.Key_5,
                           Qt.Key_6, Qt.Key_7, Qt.Key_8, Qt.Key_9):
            idx = int(chr(event.key()))
            self.class_key.emit(idx - 1)
            return
        # ←/→ 切换图片（绘制进行中时不切换，避免丢失草稿）
        if event.key() in (Qt.Key_Left, Qt.Key_Right):
            if self.tool is not None and self.tool.has_active_draft():
                return
            self.navigate_requested.emit(-1 if event.key() == Qt.Key_Left else 1)
            return
        # +/- 缩放
        if event.key() in (Qt.Key_Plus, Qt.Key_Equal):
            self.zoom_by(1.25)
            return
        if event.key() == Qt.Key_Minus:
            self.zoom_by(0.8)
            return
        if self.tool is not None and self.tool.keyPress(event):
            return
        if event.key() == Qt.Key_Delete or event.key() == Qt.Key_Backspace:
            self.delete_selected()
            return
        if event.key() == Qt.Key_Escape:
            self._cancel_temp()
            self.clear_selection()
            return
        super().keyPressEvent(event)

    def drawBackground(self, painter: QPainter, rect: QRectF) -> None:
        """画布背景 + 浅色网格。"""
        super().drawBackground(painter, rect)
        c = theme.colors
        painter.fillRect(rect, QColor(c.canvas_bg))
        # 网格
        grid = 40.0 * self._zoom
        if grid >= 8:
            pen = QPen(QColor(c.canvas_grid), 0.5)
            pen.setCosmetic(True)
            painter.setPen(pen)
            # 以场景坐标绘制网格，随缩放移动
            srect = self.mapToScene(self.viewport().rect()).boundingRect()
            x = int(srect.left() / grid) * grid
            while x < srect.right():
                painter.drawLine(QPointF(x, srect.top()), QPointF(x, srect.bottom()))
                x += grid
            y = int(srect.top() / grid) * grid
            while y < srect.bottom():
                painter.drawLine(QPointF(srect.left(), y), QPointF(srect.right(), y))
                y += grid
