"""设置面板：主题切换、标注选项、AI 选项。"""
from __future__ import annotations

import logging

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QFormLayout, QGroupBox, QHBoxLayout, QLabel,
                               QSpinBox, QVBoxLayout)

from app.config import config

logger = logging.getLogger(__name__)


class SettingsDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("设置")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)

        # ---- 主题 ----
        box_theme = QGroupBox("主题")
        f1 = QFormLayout(box_theme)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("浅色主题（默认）", "light")
        self.theme_combo.addItem("深色主题", "dark")
        idx = self.theme_combo.findData(config.theme)
        self.theme_combo.setCurrentIndex(max(idx, 0))
        f1.addRow("外观:", self.theme_combo)
        tip = QLabel("切换立即生效，无需重启，自动保存")
        tip.setStyleSheet("color: gray; font-size: 12px;")
        f1.addRow("", tip)
        layout.addWidget(box_theme)

        # ---- 标注选项 ----
        box_label = QGroupBox("标注")
        f2 = QFormLayout(box_label)
        self.cb_autosave = QCheckBox("切换图片时自动保存")
        self.cb_autosave.setChecked(config.get("auto_save", True))
        f2.addRow("", self.cb_autosave)
        self.cb_showlabels = QCheckBox("画布上显示类别文本")
        self.cb_showlabels.setChecked(config.get("show_labels", True))
        f2.addRow("", self.cb_showlabels)
        self.cb_crosshair = QCheckBox("显示十字准星（跟随鼠标，便于对齐）")
        self.cb_crosshair.setChecked(config.get("crosshair", False))
        f2.addRow("", self.cb_crosshair)
        self.spin_pose_kpts = QSpinBox()
        self.spin_pose_kpts.setRange(1, 100)
        self.spin_pose_kpts.setValue(config.get("pose_kpts", 17))
        f2.addRow("关键点数量:", self.spin_pose_kpts)
        self.skel_combo = QComboBox()
        self.skel_combo.addItem("无骨骼连线", "none")
        self.skel_combo.addItem("COCO 17 点人体骨架", "coco17")
        idx = self.skel_combo.findData(config.get("skeleton", "none"))
        self.skel_combo.setCurrentIndex(max(idx, 0))
        f2.addRow("关键点骨骼:", self.skel_combo)
        layout.addWidget(box_label)

        # ---- 通用选项 ----
        box_gen = QGroupBox("通用")
        f4 = QFormLayout(box_gen)
        self.cb_autoopen = QCheckBox("启动时自动打开上次的数据集")
        self.cb_autoopen.setChecked(config.get("auto_open_last", True))
        f4.addRow("", self.cb_autoopen)
        layout.addWidget(box_gen)

        # ---- AI 选项 ----
        box_ai = QGroupBox("AI 辅助")
        f3 = QFormLayout(box_ai)
        self.sam_combo = QComboBox()
        self.sam_combo.addItem("MobileSAM（推荐，约 40MB，速度快）", "mobile_sam")
        self.sam_combo.addItem("SAM-B（精度更高，约 370MB）", "sam_b")
        idx = self.sam_combo.findData(config.get("sam_model", "mobile_sam"))
        self.sam_combo.setCurrentIndex(max(idx, 0))
        f3.addRow("SAM 模型:", self.sam_combo)
        layout.addWidget(box_ai)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _accept(self) -> None:
        from app.theme import theme
        from PySide6.QtWidgets import QApplication
        theme.apply(QApplication.instance(), self.theme_combo.currentData())
        config.set("auto_save", self.cb_autosave.isChecked())
        config.set("show_labels", self.cb_showlabels.isChecked())
        config.set("crosshair", self.cb_crosshair.isChecked())
        config.set("pose_kpts", self.spin_pose_kpts.value())
        config.set("skeleton", self.skel_combo.currentData())
        config.set("sam_model", self.sam_combo.currentData())
        config.set("auto_open_last", self.cb_autoopen.isChecked())
        self.accept()
