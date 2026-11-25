"""类别管理面板：类别增删改、颜色、当前标注类别选择。"""
from __future__ import annotations

import logging

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtGui import QColor, QPixmap
from PySide6.QtWidgets import (QHBoxLayout, QInputDialog, QLabel, QListWidget,
                               QListWidgetItem, QMessageBox, QPushButton,
                               QVBoxLayout, QWidget)

from app.core.label_data import CLASS_COLORS

logger = logging.getLogger(__name__)


def _color_icon(color: str, size: int = 14) -> QPixmap:
    pm = QPixmap(size, size)
    pm.fill(QColor(color))
    return pm


class ClassPanel(QWidget):
    """左侧类别面板。"""

    classes_changed = Signal(list)       # 类别名列表（增/改名，不改变 id 语义）
    current_class_changed = Signal(int)  # 当前类别 id
    # 删除/重排会改变既有标注的 class_id 语义，必须由主窗口确认并
    # 重映射所有标注文件后回写面板，面板自身只发请求
    class_delete_requested = Signal(int)   # 请求删除类别（索引）
    class_move_requested = Signal(int, int)  # 请求交换两个类别（索引 i, j）

    def __init__(self, parent=None):
        super().__init__(parent)
        self._classes: list[str] = []
        self._current = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)

        title = QLabel("类别")
        title.setStyleSheet("font-weight: bold;")
        layout.addWidget(title)

        self.list_widget = QListWidget()
        self.list_widget.setIconSize(QSize(16, 16))
        self.list_widget.setSpacing(2)
        self.list_widget.currentRowChanged.connect(self._on_row_changed)
        self.list_widget.itemDoubleClicked.connect(self._on_rename)
        layout.addWidget(self.list_widget)

        btn_row1 = QHBoxLayout()
        for text, slot in (("添加", self._on_add), ("重命名", self._on_rename_btn),
                           ("删除", self._on_delete)):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row1.addWidget(b)
        layout.addLayout(btn_row1)

        btn_row2 = QHBoxLayout()
        for text, slot in (("上移", lambda: self._move(-1)), ("下移", lambda: self._move(1))):
            b = QPushButton(text)
            b.clicked.connect(slot)
            btn_row2.addWidget(b)
        layout.addLayout(btn_row2)

        # 当前类别提示
        self.cur_label = QLabel("当前: -")
        layout.addWidget(self.cur_label)

    # ---------- 数据 ----------
    def set_classes(self, classes: list[str], current: int = 0) -> None:
        self._classes = list(classes)
        self._current = max(0, min(current, len(classes) - 1)) if classes else 0
        self._refresh()
        self.current_class_changed.emit(self._current)

    @property
    def classes(self) -> list[str]:
        return self._classes

    @property
    def current_class(self) -> int:
        return self._current

    def _refresh(self) -> None:
        self.list_widget.blockSignals(True)
        self.list_widget.clear()
        for i, name in enumerate(self._classes):
            shortcut = f"  [{i + 1}]" if i < 9 else ""
            item = QListWidgetItem(_color_icon(CLASS_COLORS[i % len(CLASS_COLORS)]),
                                   f" {name}{shortcut}")
            item.setData(Qt.UserRole, i)
            item.setToolTip(f"类别 {i}: {name}\n快捷键: {i + 1}" if i < 9
                            else f"类别 {i}: {name}")
            self.list_widget.addItem(item)
        if self._classes:
            self.list_widget.setCurrentRow(self._current)
        self.cur_label.setText(
            f"当前: {self._classes[self._current] if self._classes else '-'}"
            f"{' [快捷键 ' + str(self._current + 1) + ']' if self._classes and self._current < 9 else ''}")
        self.list_widget.blockSignals(False)

    def set_current(self, class_id: int) -> None:
        """设置当前类别（如数字键 1-9 快捷切换），并通知主窗口。"""
        if not self._classes:
            return
        class_id = max(0, min(class_id, len(self._classes) - 1))
        self._current = class_id
        self.list_widget.blockSignals(True)
        self.list_widget.setCurrentRow(class_id)
        self.list_widget.blockSignals(False)
        self.cur_label.setText(
            f"当前: {self._classes[class_id]}"
            f"{' [快捷键 ' + str(class_id + 1) + ']' if class_id < 9 else ''}")
        self.current_class_changed.emit(class_id)

    # ---------- 操作 ----------
    def _on_add(self) -> None:
        name, ok = QInputDialog.getText(self, "添加类别", "类别名称:")
        if ok and name.strip():
            if name.strip() in self._classes:
                QMessageBox.warning(self, "提示", f"类别 '{name}' 已存在")
                return
            self._classes.append(name.strip())
            self._current = len(self._classes) - 1
            self._refresh()
            self.classes_changed.emit(self._classes)
            self.current_class_changed.emit(self._current)

    def _on_rename_btn(self) -> None:
        row = self.list_widget.currentRow()
        if 0 <= row < len(self._classes):
            self._on_rename(self.list_widget.item(row))

    def _on_rename(self, item: QListWidgetItem) -> None:
        idx = item.data(Qt.UserRole)
        old = self._classes[idx]
        name, ok = QInputDialog.getText(self, "重命名类别", "类别名称:", text=old)
        if ok and name.strip() and name.strip() != old:
            if name.strip() in self._classes:
                QMessageBox.warning(self, "提示", f"类别 '{name}' 已存在")
                return
            self._classes[idx] = name.strip()
            self._refresh()
            self.classes_changed.emit(self._classes)

    def _on_delete(self) -> None:
        if not self._classes:
            return
        # 删除会改变后续类别的 class_id，需要主窗口重映射所有标注文件
        self.class_delete_requested.emit(self._current)

    def _move(self, delta: int) -> None:
        i = self._current
        j = i + delta
        if 0 <= j < len(self._classes):
            # 交换两个类别会互换它们的 class_id 语义，需要主窗口重映射
            self.class_move_requested.emit(i, j)

    # ---------- 事件 ----------
    def _on_row_changed(self, row: int) -> None:
        if 0 <= row < len(self._classes):
            self._current = row
            self.cur_label.setText(f"当前: {self._classes[row]}")
            self.current_class_changed.emit(row)
