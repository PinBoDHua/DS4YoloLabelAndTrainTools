"""标注工具状态机：选择/矩形/多边形/关键点/OBB 五种交互模式。"""
from __future__ import annotations

import logging

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QCursor

from app.core.label_data import LabelItem, LabelType
from app.ui.canvas import LabelCanvas, ShapeItem, class_color

logger = logging.getLogger(__name__)


class BaseTool:
    """工具基类。鼠标事件坐标为 scene（像素）坐标。"""

    CURSOR = Qt.ArrowCursor

    def __init__(self, canvas: LabelCanvas):
        self.canvas = canvas

    def cursor(self) -> QCursor:
        return QCursor(self.CURSOR)

    def _pos(self, event) -> QPointF:
        return self.canvas.mapToScene(event.position().toPoint())

    # 事件返回 True 表示已处理（阻止向下传递）
    def mousePress(self, event) -> bool:
        return False

    def mouseMove(self, event) -> bool:
        return False

    def mouseRelease(self, event) -> bool:
        return False

    def mouseDoubleClick(self, event) -> bool:
        return False

    def keyPress(self, event) -> bool:
        return False

    def deactivate(self) -> None:
        """切换离开时清理临时状态。"""
        pass

    def has_active_draft(self) -> bool:
        """是否有未完成的绘制草稿（←/→ 切图前检查，避免丢失）。"""
        return False


class SelectTool(BaseTool):
    """选择/编辑工具：点击选中形状，拖拽移动，手柄编辑在 ShapeItem 内处理。

    支持多选批量移动：拖动任一已选中形状时，所有选中形状一起移动；
    Ctrl+点击切换单个形状的选中状态。
    """
    CURSOR = Qt.ArrowCursor

    def __init__(self, canvas):
        super().__init__(canvas)
        self._press_pos: QPointF | None = None
        self._group_base: dict = {}
        self._dragging = False

    def mousePress(self, event) -> bool:
        if event.button() == Qt.RightButton:
            # POSE 关键点右键循环可见性：转发给命中关键点的形状；
            # 未命中则不处理，留给画布的上下文菜单
            pos = self._pos(event)
            for shape in self.canvas.shapes:
                if shape.label.type != LabelType.POSE:
                    continue
                h = shape.handle_at(pos)
                if h and h.startswith("k"):
                    shape.mousePressEvent(_ItemEvent(event, pos))
                    return True
            return False
        if event.button() != Qt.LeftButton:
            return False
        pos = self._pos(event)
        item = self.canvas.itemAt(event.position().toPoint())
        shape = self._shape_of(item)
        # 命中形状
        if shape is not None:
            # 命中手柄：交给 ShapeItem 处理（单形状顶点/旋转编辑）
            if shape.handle_at(pos):
                shape.mousePressEvent(_ItemEvent(event, pos))
                return True
            # Ctrl+点击：切换选中状态，不进入拖动
            if bool(event.modifiers() & Qt.ControlModifier):
                shape.setSelected(not shape.isSelected())
                return True
            # 点击未选中形状：先选中它（若已有其他选中则只保留它）
            if not shape.isSelected():
                self.canvas.scene.clearSelection()
                shape.setSelected(True)
            # 记录所有选中形状的拖动起点（多选整体移动）
            self._press_pos = pos
            self._group_base = {
                s: QPointF(*s.label.points[0])
                for s in self.canvas.shapes if s.isSelected()
            }
            self._dragging = False
            return True
        # 空白处：清空选择
        self.canvas.clear_selection()
        return True

    def mouseMove(self, event) -> bool:
        if self._press_pos is not None:
            pos = self._pos(event)
            if not self._dragging:
                if abs(pos.x() - self._press_pos.x()) < 3 and \
                        abs(pos.y() - self._press_pos.y()) < 3:
                    return True  # 尚未超过拖动阈值
                self._dragging = True
                self.canvas.edit_begin.emit()
            for s, base in self._group_base.items():
                cur = QPointF(*s.label.points[0])
                s.move_by(base.x() + pos.x() - self._press_pos.x() - cur.x(),
                          base.y() + pos.y() - self._press_pos.y() - cur.y())
            return True
        shape = self._selected()
        if shape is not None and (shape._drag_handle or shape._drag_offset is not None):
            shape.mouseMoveEvent(_ItemEvent(event, self._pos(event)))
            return True
        return False

    def mouseRelease(self, event) -> bool:
        if self._press_pos is not None:
            self._press_pos = None
            was_drag, self._dragging = self._dragging, False
            self._group_base = {}
            if was_drag:
                self.canvas.labels_changed.emit()
            return True
        shape = self._selected()
        if shape is not None and (shape._drag_handle or shape._drag_offset is not None):
            shape.mouseReleaseEvent(_ItemEvent(event, self._pos(event)))
            return True
        return False

    def mouseDoubleClick(self, event) -> bool:
        return False

    def _shape_of(self, item):
        if isinstance(item, ShapeItem):
            return item
        # 手柄/子项一般直接命中 ShapeItem（shape() 包含手柄）
        return None

    def _selected(self) -> ShapeItem | None:
        sel = self.canvas.scene.selectedItems()
        for s in sel:
            if isinstance(s, ShapeItem):
                return s
        return None


class RectTool(BaseTool):
    """矩形框绘制：拖拽出矩形，松开完成。"""
    CURSOR = Qt.CrossCursor

    def __init__(self, canvas):
        super().__init__(canvas)
        self._start: QPointF | None = None

    def mousePress(self, event) -> bool:
        if event.button() == Qt.LeftButton:
            self._start = self._pos(event)
            self.canvas.create_temp(LabelType.DETECT, class_color(self.canvas.current_class_id))
            return True
        return False

    def mouseMove(self, event) -> bool:
        if self._start is not None and self.canvas.temp_item:
            self.canvas.temp_item.points = [self._start]
            self.canvas.temp_item.cur = self._pos(event)
            self.canvas.temp_item.prepareGeometryChange()
            self.canvas.temp_item.update()
            return True
        return False

    def mouseRelease(self, event) -> bool:
        if event.button() == Qt.LeftButton and self._start is not None:
            end = self._pos(event)
            x1, y1 = self._start.x(), self._start.y()
            x2, y2 = end.x(), end.y()
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                lb = LabelItem(LabelType.DETECT, self.canvas.current_class_id,
                               points=[[x1, y1], [x2, y2]])
                self.canvas.finish_temp(lb)
            else:
                self.canvas.finish_temp(None)  # 过小取消
            self._start = None
            return True
        return False

    def keyPress(self, event) -> bool:
        if event.key() == Qt.Key_Escape:
            self._start = None
            self.canvas._cancel_temp()
            return True
        return False

    def has_active_draft(self) -> bool:
        return self._start is not None


class PolygonTool(BaseTool):
    """多边形分割：左键加点，双击/回车闭合，Esc 取消。"""
    CURSOR = Qt.CrossCursor

    def __init__(self, canvas):
        super().__init__(canvas)
        self._points: list[QPointF] = []

    def mousePress(self, event) -> bool:
        if event.button() == Qt.LeftButton:
            if not self._points:
                self.canvas.create_temp(LabelType.SEGMENT,
                                        class_color(self.canvas.current_class_id))
            self._points.append(self._pos(event))
            self._sync_temp()
            return True
        if event.button() == Qt.RightButton and self._points:  # 右键闭合
            self._finish()
            return True
        return False

    def mouseMove(self, event) -> bool:
        if self._points and self.canvas.temp_item:
            self.canvas.temp_item.cur = self._pos(event)
            self.canvas.temp_item.prepareGeometryChange()
            self.canvas.temp_item.update()
            return True
        return False

    def mouseDoubleClick(self, event) -> bool:
        if self._points and event.button() == Qt.LeftButton:
            self._finish()
            return True
        return False

    def keyPress(self, event) -> bool:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self._points:
            self._finish()
            return True
        if event.key() == Qt.Key_Escape:
            self._points = []
            self.canvas._cancel_temp()
            return True
        return False

    def _sync_temp(self) -> None:
        if self.canvas.temp_item:
            self.canvas.temp_item.points = list(self._points)
            self.canvas.temp_item.prepareGeometryChange()
            self.canvas.temp_item.update()

    def _finish(self) -> None:
        if len(self._points) >= 3:
            lb = LabelItem(LabelType.SEGMENT, self.canvas.current_class_id,
                           points=[[p.x(), p.y()] for p in self._points])
            self.canvas.finish_temp(lb)
        else:
            self.canvas.finish_temp(None)
        self._points = []

    def deactivate(self) -> None:
        self._points = []
        self.canvas._cancel_temp()

    def has_active_draft(self) -> bool:
        return bool(self._points)


class PoseTool(BaseTool):
    """关键点标注：先拖拽画框，再依次点击添加关键点，双击/回车完成实例。"""
    CURSOR = Qt.CrossCursor

    def __init__(self, canvas):
        super().__init__(canvas)
        self._start: QPointF | None = None
        self._box: LabelItem | None = None   # 阶段 2 的实例
        self._kpts: list[list] = []

    def mousePress(self, event) -> bool:
        if event.button() != Qt.LeftButton:
            return False
        pos = self._pos(event)
        if self._box is None:
            # 阶段 1：画框
            if self._start is None:
                self._start = pos
                self.canvas.create_temp(LabelType.POSE, class_color(self.canvas.current_class_id))
            return True
        # 阶段 2：加关键点
        kpt_limit = getattr(self.canvas, "pose_kpts", 17)
        self._kpts.append([pos.x(), pos.y(), 2])
        self._sync_temp()
        self.canvas.polygon_point_added.emit()
        if len(self._kpts) >= kpt_limit:
            self._finish()
        return True

    def mouseMove(self, event) -> bool:
        if self._start is not None and self.canvas.temp_item and self._box is None:
            self.canvas.temp_item.points = [self._start]
            self.canvas.temp_item.cur = self._pos(event)
            self.canvas.temp_item.prepareGeometryChange()
            self.canvas.temp_item.update()
            return True
        return False

    def mouseRelease(self, event) -> bool:
        if event.button() == Qt.LeftButton and self._start is not None and self._box is None:
            end = self._pos(event)
            if abs(end.x() - self._start.x()) >= 3 and abs(end.y() - self._start.y()) >= 3:
                # 框完成 → 进入阶段 2，临时预览变为 keypoints
                self._box = LabelItem(LabelType.POSE, self.canvas.current_class_id,
                                      points=[[self._start.x(), self._start.y()],
                                              [end.x(), end.y()]])
                self._kpts = []
                self.canvas.create_temp(LabelType.POSE, class_color(self.canvas.current_class_id))
                self._sync_temp()
            else:
                self._start = None
                self.canvas.finish_temp(None)
            self._start = None
            return True
        return False

    def mouseDoubleClick(self, event) -> bool:
        if self._box is not None and self._kpts:
            self._finish()
            return True
        return False

    def keyPress(self, event) -> bool:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter) and self._box is not None:
            self._finish()
            return True
        if event.key() == Qt.Key_Escape:
            self._start = None
            self._box = None
            self._kpts = []
            self.canvas._cancel_temp()
            return True
        return False

    def _sync_temp(self) -> None:
        if self.canvas.temp_item:
            t = self.canvas.temp_item
            if self._box is not None:
                x1, y1, x2, y2 = self._box.rect
                t.points = [QPointF(x1, y1), QPointF(x2, y2)]
                t.kpts = [QPointF(k[0], k[1]) for k in self._kpts]
            t.prepareGeometryChange()
            t.update()

    def _finish(self) -> None:
        if self._box is not None and self._kpts:
            self._box.keypoints = self._kpts
            self.canvas.finish_temp(self._box)
        else:
            self.canvas.finish_temp(None)
        self._box = None
        self._kpts = []
        self._start = None

    def deactivate(self) -> None:
        self._start = None
        self._box = None
        self._kpts = []
        self.canvas._cancel_temp()

    def has_active_draft(self) -> bool:
        return self._start is not None or self._box is not None


class ObbTool(BaseTool):
    """OBB 旋转框：拖拽绘制矩形，完成后自动切换到选择模式编辑旋转手柄。"""
    CURSOR = Qt.CrossCursor

    def __init__(self, canvas):
        super().__init__(canvas)
        self._start: QPointF | None = None

    def mousePress(self, event) -> bool:
        if event.button() == Qt.LeftButton:
            self._start = self._pos(event)
            self.canvas.create_temp(LabelType.OBB, class_color(self.canvas.current_class_id))
            return True
        return False

    def mouseMove(self, event) -> bool:
        if self._start is not None and self.canvas.temp_item:
            self.canvas.temp_item.points = [self._start]
            self.canvas.temp_item.cur = self._pos(event)
            self.canvas.temp_item.prepareGeometryChange()
            self.canvas.temp_item.update()
            return True
        return False

    def mouseRelease(self, event) -> bool:
        if event.button() == Qt.LeftButton and self._start is not None:
            end = self._pos(event)
            x1, y1 = self._start.x(), self._start.y()
            x2, y2 = end.x(), end.y()
            self._start = None
            if abs(x2 - x1) >= 3 and abs(y2 - y1) >= 3:
                pts = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]  # 顺时针
                lb = LabelItem(LabelType.OBB, self.canvas.current_class_id, points=pts)
                self.canvas.finish_temp(lb)
                # 完成后切回选择工具，方便直接旋转（通知主窗口同步工具栏状态）
                self.canvas.set_tool(SelectTool(self.canvas))
                self.canvas.tool_auto_changed.emit("select")
            else:
                self.canvas.finish_temp(None)
            return True
        return False

    def keyPress(self, event) -> bool:
        if event.key() == Qt.Key_Escape:
            self._start = None
            self.canvas._cancel_temp()
            return True
        return False

    def has_active_draft(self) -> bool:
        return self._start is not None


class _ItemEvent:
    """将 QGraphicsViewMouseEvent 适配为 QGraphicsSceneMouseEvent 的轻量封装。"""

    def __init__(self, view_event, scene_pos: QPointF):
        self._ve = view_event
        self._pos = scene_pos

    def button(self):
        return self._ve.button()

    def pos(self) -> QPointF:
        return self._pos

    def accept(self):
        self._ve.accept()

    def ignore(self):
        self._ve.ignore()


# 工具工厂
def make_tool(name: str, canvas: LabelCanvas):
    return {
        "select": SelectTool,
        "rect": RectTool,
        "polygon": PolygonTool,
        "pose": PoseTool,
        "obb": ObbTool,
    }[name](canvas)
