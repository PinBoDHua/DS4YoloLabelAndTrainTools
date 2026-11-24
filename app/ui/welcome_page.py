"""欢迎页：无数据集时显示的起始页，提供新建/打开数据集、最近项目与功能导览。"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (QFrame, QHBoxLayout, QLabel, QPushButton,
                               QVBoxLayout, QWidget)

from app.config import config
from app.theme import theme

logger = logging.getLogger(__name__)

_FEATURES = [
    ("🔲", "四种标注", "检测框 / 分割多边形 / 关键点 / 旋转框（OBB），支持撤销重做、复制粘贴。"),
    ("🧠", "AI 辅助", "模型预标注 + SAM 智能分割，一键生成标注草稿。"),
    ("📈", "训练监控", "内置训练面板，实时损失曲线，支持导出 ONNX / TensorRT 等格式。"),
    ("🔍", "推理测试", "对图片、文件夹、视频直接推理验证模型效果。"),
]

_STEPS = [
    ("1", "创建或打开数据集", "新建数据集时设置类别与任务类型"),
    ("2", "标注图片", "在画布上绘制标注，快捷键提升效率"),
    ("3", "划分并训练", "自动划分训练/验证/测试集，一键开始训练"),
    ("4", "导出与推理", "导出模型或在推理页验证效果"),
]


class _Card(QFrame):
    """欢迎页信息卡片。"""

    def __init__(self, icon: str, title: str, desc: str, parent=None):
        super().__init__(parent)
        self.setObjectName("welcome_card")
        lay = QHBoxLayout(self)
        lay.setContentsMargins(14, 12, 14, 12)
        icon_lb = QLabel(icon)
        icon_lb.setObjectName("card_icon")
        icon_lb.setFixedWidth(40)
        lay.addWidget(icon_lb, 0, Qt.AlignTop)
        text_lay = QVBoxLayout()
        t = QLabel(title)
        t.setObjectName("card_title")
        d = QLabel(desc)
        d.setObjectName("card_desc")
        d.setWordWrap(True)
        text_lay.addWidget(t)
        text_lay.addWidget(d)
        lay.addLayout(text_lay, 1)


class WelcomePage(QWidget):
    """起始页（主窗口「开始」页签）。"""

    new_dataset_requested = Signal()
    open_dataset_requested = Signal()
    open_path_requested = Signal(str)  # 最近项目 / 继续上次会话

    def __init__(self, parent=None):
        super().__init__(parent)
        self._recent_buttons: list[QPushButton] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addStretch(2)

        content = QVBoxLayout()
        content.setSpacing(10)

        # 标题区
        title = QLabel("YOLO 标注训练工具")
        title.setObjectName("welcome_title")
        title.setAlignment(Qt.AlignCenter)
        content.addWidget(title)
        subtitle = QLabel("一站式标注 · 训练 · 推理，支持检测 / 分割 / 关键点 / 旋转框")
        subtitle.setObjectName("welcome_subtitle")
        subtitle.setAlignment(Qt.AlignCenter)
        content.addWidget(subtitle)

        # 主操作按钮
        btn_row = QHBoxLayout()
        btn_row.setSpacing(12)
        self.btn_new = QPushButton("＋ 新建数据集")
        self.btn_new.setProperty("primary", True)
        self.btn_new.setCursor(Qt.PointingHandCursor)
        self.btn_new.clicked.connect(self.new_dataset_requested.emit)
        self.btn_open = QPushButton("打开数据集")
        self.btn_open.setCursor(Qt.PointingHandCursor)
        self.btn_open.clicked.connect(self.open_dataset_requested.emit)
        self.btn_new.setMinimumHeight(44)
        self.btn_open.setMinimumHeight(44)
        btn_row.addStretch(1)
        btn_row.addWidget(self.btn_new)
        btn_row.addWidget(self.btn_open)
        btn_row.addStretch(1)
        content.addLayout(btn_row)

        # 继续上次会话
        last = config.get("last_dataset")
        self.btn_continue = QPushButton()
        self.btn_continue.setCursor(Qt.PointingHandCursor)
        self.btn_continue.setVisible(bool(last) and Path(last).is_dir())
        if last:
            self.btn_continue.setText(f"▶ 继续上次会话：{Path(last).name}")
            self.btn_continue.setToolTip(last)
            self.btn_continue.clicked.connect(
                lambda: self.open_path_requested.emit(last))
        content.addWidget(self.btn_continue, 0, Qt.AlignHCenter)

        # 最近项目
        self.recent_box = QWidget()
        recent_lay = QVBoxLayout(self.recent_box)
        recent_lay.setContentsMargins(0, 0, 0, 0)
        self.recent_box.setVisible(False)
        self.recent_title = QLabel("最近打开")
        self.recent_title.setObjectName("section_title")
        recent_lay.addWidget(self.recent_title, 0, Qt.AlignHCenter)
        self.recent_list = QVBoxLayout()
        self.recent_list.setSpacing(6)
        recent_lay.addLayout(self.recent_list)
        content.addWidget(self.recent_box)
        self._refresh_recent()

        # 功能导览卡片
        cards_row = QHBoxLayout()
        cards_row.setSpacing(12)
        for icon, t, d in _FEATURES:
            cards_row.addWidget(_Card(icon, t, d))
        content.addLayout(cards_row)

        # 使用流程
        steps_row = QHBoxLayout()
        steps_row.setSpacing(12)
        for num, t, d in _STEPS:
            card = QFrame()
            card.setObjectName("step_card")
            v = QVBoxLayout(card)
            v.setContentsMargins(12, 10, 12, 10)
            n = QLabel(num)
            n.setObjectName("step_num")
            tlb = QLabel(t)
            tlb.setObjectName("card_title")
            dlb = QLabel(d)
            dlb.setObjectName("card_desc")
            dlb.setWordWrap(True)
            v.addWidget(n)
            v.addWidget(tlb)
            v.addWidget(dlb)
            steps_row.addWidget(card)
        content.addLayout(steps_row)

        outer.addLayout(content)
        outer.addStretch(3)

        theme.theme_changed.connect(self._apply_style)
        self._apply_style()

    def _apply_style(self, _name: str = "") -> None:
        c = theme.colors
        self.setStyleSheet(f"""
            QLabel#welcome_title {{
                font-size: 30px;
                font-weight: bold;
                color: {c.text};
            }}
            QLabel#welcome_subtitle {{
                font-size: 14px;
                color: {c.text_dim};
                margin-bottom: 8px;
            }}
            QLabel#section_title {{
                font-size: 13px;
                color: {c.text_dim};
                margin-top: 10px;
            }}
            QLabel#card_icon {{ font-size: 22px; }}
            QLabel#card_title {{
                font-weight: bold;
                color: {c.text};
                font-size: 13px;
            }}
            QLabel#card_desc {{
                color: {c.text_dim};
                font-size: 12px;
            }}
            QLabel#step_num {{
                font-size: 18px;
                font-weight: bold;
                color: {c.accent};
            }}
            QFrame#welcome_card, QFrame#step_card {{
                background-color: {c.panel_bg};
                border: 1px solid {c.border};
                border-radius: 8px;
            }}
            QPushButton {{
                background-color: {c.panel_bg};
                border: 1px solid {c.border};
                border-radius: 6px;
                padding: 8px 22px;
                color: {c.text};
                font-size: 14px;
            }}
            QPushButton:hover {{
                border-color: {c.accent};
                background-color: {c.panel_alt};
            }}
            QPushButton[primary="true"] {{
                background-color: {c.accent};
                color: {c.accent_text};
                border-color: {c.accent};
                font-weight: bold;
            }}
            QPushButton[primary="true"]:hover {{
                background-color: {c.accent_hover};
            }}
        """)

    def _refresh_recent(self) -> None:
        # 清空旧按钮
        for b in self._recent_buttons:
            self.recent_list.removeWidget(b)
            b.deleteLater()
        self._recent_buttons = []
        recent = [p for p in config.recent_projects if Path(p).is_dir()]
        if not recent:
            self.recent_box.setVisible(False)
            return
        self.recent_box.setVisible(True)
        for p in recent[:5]:
            b = QPushButton(f"{Path(p).name}  —  {p}")
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(p)
            b.clicked.connect(lambda _=False, path=p: self.open_path_requested.emit(path))
            self._recent_buttons.append(b)
            self.recent_list.addWidget(b, 0, Qt.AlignHCenter)

    def refresh(self) -> None:
        """数据集变更后刷新最近项目与继续会话按钮。"""
        self._refresh_recent()
        last = config.get("last_dataset")
        has_last = bool(last) and Path(last).is_dir()
        if has_last:
            self.btn_continue.setText(f"▶ 继续上次会话：{Path(last).name}")
            self.btn_continue.setToolTip(last)
            try:
                self.btn_continue.clicked.disconnect()
            except Exception:
                pass
            self.btn_continue.clicked.connect(
                lambda: self.open_path_requested.emit(last))
        self.btn_continue.setVisible(has_last)
