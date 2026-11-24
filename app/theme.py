"""主题管理：浅色（默认）/ 深色两套主题，运行时热切换。"""
from __future__ import annotations

from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

from app.config import config


class ThemeColors:
    """主题色板：供 QSS 与自定义绘制（画布/图表）共用。"""

    def __init__(self, name: str, dark: bool):
        self.name = name
        self.dark = dark
        if not dark:
            # ---- 浅色主题 ----
            self.window_bg = "#f4f5f7"       # 主窗口背景
            self.panel_bg = "#ffffff"        # 面板/列表背景
            self.panel_alt = "#fafbfc"       # 交替行
            self.input_bg = "#ffffff"        # 输入框背景
            self.border = "#d8dce1"          # 边框
            self.text = "#1b1f24"            # 主文本
            self.text_dim = "#6a7178"        # 次要文本
            self.accent = "#1f6feb"          # 强调色
            self.accent_hover = "#3a84f2"
            self.accent_text = "#ffffff"
            self.selection_bg = "#cfe0fb"    # 选中背景
            self.danger = "#d64545"
            self.success = "#2da44e"
            self.canvas_bg = "#e9ebee"       # 标注画布背景
            self.canvas_grid = "#d6dade"     # 画布网格线
            self.toolbar_bg = "#ffffff"
            self.status_bg = "#ffffff"
            self.scrollbar = "#c4c9cf"
        else:
            # ---- 深色主题 ----
            self.window_bg = "#1b1d21"
            self.panel_bg = "#22252b"
            self.panel_alt = "#282b32"
            self.input_bg = "#2c3037"
            self.border = "#3a3f48"
            self.text = "#e2e5ea"
            self.text_dim = "#9aa3ae"
            self.accent = "#4a9eff"
            self.accent_hover = "#6cb2ff"
            self.accent_text = "#ffffff"
            self.selection_bg = "#1d3a63"
            self.danger = "#e05555"
            self.success = "#3fb950"
            self.canvas_bg = "#14161a"
            self.canvas_grid = "#26292f"
            self.toolbar_bg = "#22252b"
            self.status_bg = "#22252b"
            self.scrollbar = "#4a4f58"


LIGHT = ThemeColors("light", dark=False)
DARK = ThemeColors("dark", dark=True)


def _build_qss(c: ThemeColors) -> str:
    """根据色板生成完整 QSS 样式表。"""
    return f"""
* {{
    font-family: "Microsoft YaHei UI", "Microsoft YaHei", "Segoe UI", sans-serif;
    font-size: 13px;
    outline: none;
}}
QMainWindow, QDialog {{
    background-color: {c.window_bg};
    color: {c.text};
}}
QWidget {{
    color: {c.text};
}}
QMenuBar {{
    background-color: {c.toolbar_bg};
    border-bottom: 1px solid {c.border};
    padding: 2px;
}}
QMenuBar::item {{
    background: transparent;
    padding: 5px 10px;
    border-radius: 4px;
}}
QMenuBar::item:selected {{
    background-color: {c.selection_bg};
}}
QMenu {{
    background-color: {c.panel_bg};
    border: 1px solid {c.border};
    padding: 4px;
}}
QMenu::item {{
    padding: 6px 28px 6px 24px;
    border-radius: 4px;
}}
QMenu::item:selected {{
    background-color: {c.selection_bg};
}}
QMenu::separator {{
    height: 1px;
    background: {c.border};
    margin: 4px 8px;
}}
QToolBar {{
    background-color: {c.toolbar_bg};
    border-bottom: 1px solid {c.border};
    padding: 4px;
    spacing: 4px;
}}
QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 5px;
    padding: 5px 10px;
    color: {c.text};
}}
QToolButton:hover {{
    background-color: {c.selection_bg};
}}
QToolButton:checked {{
    background-color: {c.accent};
    color: {c.accent_text};
    border-color: {c.accent};
}}
QToolButton:disabled {{
    color: {c.text_dim};
}}
QPushButton {{
    background-color: {c.panel_bg};
    border: 1px solid {c.border};
    border-radius: 5px;
    padding: 6px 14px;
    color: {c.text};
}}
QPushButton:hover {{
    background-color: {c.panel_alt};
    border-color: {c.accent};
}}
QPushButton:pressed {{
    background-color: {c.selection_bg};
}}
QPushButton:disabled {{
    color: {c.text_dim};
    border-color: {c.border};
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
QPushButton[danger="true"] {{
    background-color: {c.danger};
    color: white;
    border-color: {c.danger};
}}
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit, QPlainTextEdit {{
    background-color: {c.input_bg};
    border: 1px solid {c.border};
    border-radius: 5px;
    padding: 4px 8px;
    color: {c.text};
    selection-background-color: {c.selection_bg};
    selection-color: {c.text};
}}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {c.accent};
}}
QComboBox::drop-down {{
    border: none;
    width: 22px;
}}
QComboBox::down-arrow {{
    image: none;
    border-left: 4px solid transparent;
    border-right: 4px solid transparent;
    border-top: 5px solid {c.text_dim};
    margin-right: 6px;
}}
QComboBox QAbstractItemView {{
    background-color: {c.panel_bg};
    border: 1px solid {c.border};
    selection-background-color: {c.selection_bg};
    selection-color: {c.text};
    outline: none;
}}
QListWidget, QTreeWidget, QTableWidget {{
    background-color: {c.panel_bg};
    border: 1px solid {c.border};
    border-radius: 5px;
    color: {c.text};
    alternate-background-color: {c.panel_alt};
}}
QListWidget::item, QTreeWidget::item {{
    padding: 4px 6px;
    border-radius: 4px;
}}
QListWidget::item:selected, QTreeWidget::item:selected {{
    background-color: {c.selection_bg};
    color: {c.text};
}}
QListWidget::item:hover, QTreeWidget::item:hover {{
    background-color: {c.panel_alt};
}}
QTableWidget::item:selected {{
    background-color: {c.selection_bg};
}}
QHeaderView::section {{
    background-color: {c.panel_alt};
    border: none;
    border-bottom: 1px solid {c.border};
    border-right: 1px solid {c.border};
    padding: 5px 8px;
    color: {c.text};
}}
QGroupBox {{
    border: 1px solid {c.border};
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 6px;
    background-color: {c.panel_bg};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: {c.accent};
}}
QSplitter::handle {{
    background-color: {c.border};
}}
QSplitter::handle:horizontal {{ width: 2px; }}
QSplitter::handle:vertical {{ height: 2px; }}
QStatusBar {{
    background-color: {c.status_bg};
    border-top: 1px solid {c.border};
    color: {c.text_dim};
}}
QProgressBar {{
    background-color: {c.panel_alt};
    border: 1px solid {c.border};
    border-radius: 5px;
    text-align: center;
    color: {c.text};
    height: 16px;
}}
QProgressBar::chunk {{
    background-color: {c.accent};
    border-radius: 4px;
}}
QScrollBar:vertical {{
    background: transparent;
    width: 10px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {c.scrollbar};
    border-radius: 5px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: {c.accent};
}}
QScrollBar:horizontal {{
    background: transparent;
    height: 10px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {c.scrollbar};
    border-radius: 5px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: {c.accent};
}}
QScrollBar::add-line, QScrollBar::sub-line {{
    width: 0; height: 0; background: transparent;
}}
QScrollBar::add-page, QScrollBar::sub-page {{
    background: transparent;
}}
QTabWidget::pane {{
    border: 1px solid {c.border};
    border-radius: 5px;
    background-color: {c.panel_bg};
}}
QTabBar::tab {{
    background: transparent;
    padding: 7px 16px;
    border: none;
    border-bottom: 2px solid transparent;
    color: {c.text_dim};
}}
QTabBar::tab:selected {{
    color: {c.accent};
    border-bottom: 2px solid {c.accent};
    font-weight: bold;
}}
QTabBar::tab:hover {{
    color: {c.text};
}}
QCheckBox, QRadioButton {{
    spacing: 6px;
    color: {c.text};
}}
QCheckBox::indicator, QRadioButton::indicator {{
    width: 16px; height: 16px;
    border: 1px solid {c.border};
    border-radius: 4px;
    background: {c.input_bg};
}}
QCheckBox::indicator:checked, QRadioButton::indicator:checked {{
    background-color: {c.accent};
    border-color: {c.accent};
}}
QLabel {{
    color: {c.text};
    background: transparent;
}}
QToolTip {{
    background-color: {c.panel_bg};
    color: {c.text};
    border: 1px solid {c.border};
    padding: 4px 8px;
}}
QDockWidget {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}}
QDockWidget::title {{
    background-color: {c.toolbar_bg};
    padding: 6px 10px;
    border-bottom: 1px solid {c.border};
    color: {c.text};
    font-weight: bold;
}}
QSlider::groove:horizontal {{
    height: 4px;
    background: {c.border};
    border-radius: 2px;
}}
QSlider::handle:horizontal {{
    width: 14px; height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    background: {c.accent};
}}
QSlider::sub-page:horizontal {{
    background: {c.accent};
    border-radius: 2px;
}}
QMessageBox {{
    background-color: {c.window_bg};
}}
"""


def _build_palette(c: ThemeColors) -> QPalette:
    """构建 QPalette，保证 QSS 未覆盖的原生控件颜色一致。"""
    p = QPalette()
    p.setColor(QPalette.Window, QColor(c.window_bg))
    p.setColor(QPalette.WindowText, QColor(c.text))
    p.setColor(QPalette.Base, QColor(c.panel_bg))
    p.setColor(QPalette.AlternateBase, QColor(c.panel_alt))
    p.setColor(QPalette.Text, QColor(c.text))
    p.setColor(QPalette.Button, QColor(c.panel_bg))
    p.setColor(QPalette.ButtonText, QColor(c.text))
    p.setColor(QPalette.Highlight, QColor(c.accent))
    p.setColor(QPalette.HighlightedText, QColor(c.accent_text))
    p.setColor(QPalette.ToolTipBase, QColor(c.panel_bg))
    p.setColor(QPalette.ToolTipText, QColor(c.text))
    p.setColor(QPalette.PlaceholderText, QColor(c.text_dim))
    p.setColor(QPalette.Link, QColor(c.accent))
    disabled = QColor(c.text_dim)
    p.setColor(QPalette.Disabled, QPalette.Text, disabled)
    p.setColor(QPalette.Disabled, QPalette.ButtonText, disabled)
    p.setColor(QPalette.Disabled, QPalette.WindowText, disabled)
    return p


class ThemeManager(QObject):
    """全局主题管理器（单例）。"""

    theme_changed = Signal(str)  # 参数: "light" / "dark"

    def __init__(self):
        super().__init__()
        self._current: ThemeColors = LIGHT if config.theme != "dark" else DARK
        self._qss_light = _build_qss(LIGHT)
        self._qss_dark = _build_qss(DARK)

    @property
    def colors(self) -> ThemeColors:
        """当前主题色板（画布/图表自定义绘制时读取）。"""
        return self._current

    @property
    def is_dark(self) -> bool:
        return self._current.dark

    def apply(self, app: QApplication, name: str | None = None) -> None:
        """应用主题到整个应用，运行时热切换。"""
        if name is not None:
            self._current = LIGHT if name == "light" else DARK
        app.setPalette(_build_palette(self._current))
        app.setStyleSheet(self._qss_light if not self._current.dark else self._qss_dark)
        config.theme = self._current.name
        self.theme_changed.emit(self._current.name)

    def toggle(self, app: QApplication) -> None:
        self.apply(app, "dark" if not self._current.dark else "light")


# 全局单例
theme = ThemeManager()
