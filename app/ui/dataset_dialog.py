"""数据集相关对话框：新建数据集、统计、划分、格式转换。"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (QCheckBox, QComboBox, QDialog, QDialogButtonBox,
                               QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSpinBox, QVBoxLayout)

from app.core.dataset_manager import Dataset
from app.core.label_data import CLASS_COLORS, LabelType

logger = logging.getLogger(__name__)

TASK_NAMES = {
    LabelType.DETECT: "检测（矩形框）",
    LabelType.SEGMENT: "分割（多边形）",
    LabelType.POSE: "关键点",
    LabelType.OBB: "旋转框 (OBB)",
}


class NewDatasetDialog(QDialog):
    """新建数据集。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("新建数据集")
        self.setMinimumWidth(520)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        # 目录
        dir_row = QHBoxLayout()
        self.dir_edit = QLineEdit()
        self.dir_edit.setPlaceholderText("选择数据集保存目录（建议新建空文件夹）")
        btn_browse = QPushButton("浏览...")
        btn_browse.clicked.connect(self._browse)
        dir_row.addWidget(self.dir_edit)
        dir_row.addWidget(btn_browse)
        form.addRow("数据集目录:", dir_row)

        # 类别
        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("用逗号分隔，例如：person, car, dog, cat")
        form.addRow("类别列表:", self.classes_edit)

        # 任务类型
        self.task_combo = QComboBox()
        for t, name in TASK_NAMES.items():
            self.task_combo.addItem(name, t.value)
        form.addRow("任务类型:", self.task_combo)

        layout.addLayout(form)

        tip = QLabel("创建后目录结构：images/（图片）+ labels/（标注）+ dataset.json（元数据）")
        tip.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(tip)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "选择数据集目录")
        if path:
            self.dir_edit.setText(path)

    def dataset_root(self) -> str:
        return self.dir_edit.text().strip()

    def classes(self) -> list[str]:
        return [c.strip() for c in self.classes_edit.text().replace("，", ",").split(",")
                if c.strip()]

    def task_type(self) -> LabelType:
        return LabelType(self.task_combo.currentData())

    def _accept(self) -> None:
        if not self.dataset_root():
            QMessageBox.warning(self, "提示", "请选择数据集目录。")
            return
        if not self.classes():
            QMessageBox.warning(self, "提示", "请至少输入一个类别名称。")
            return
        self.accept()


class StatsDialog(QDialog):
    """数据集统计。"""

    def __init__(self, parent, dataset: Dataset):
        super().__init__(parent)
        self.setWindowTitle("数据集统计")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        stats = dataset.stats()

        info = QLabel(
            f"<b>{Path(dataset.root).name}</b><br>"
            f"图片总数: <b>{stats['total']}</b> | 已标注: <b>{stats['labeled']}</b> | "
            f"未标注: {stats['unlabeled']}"
        )
        layout.addWidget(info)

        # 每类实例数（带颜色块）
        box = QGroupBox("每类实例数量")
        box_layout = QVBoxLayout(box)
        for cid, name in enumerate(dataset.meta.classes):
            cnt = stats["per_class"].get(cid, 0)
            row = QHBoxLayout()
            color_lb = QLabel()
            color_lb.setFixedSize(14, 14)
            color_lb.setStyleSheet(
                f"background-color: {CLASS_COLORS[cid % len(CLASS_COLORS)]}; border-radius: 3px;")
            name_lb = QLabel(f"{cid} - {name}")
            cnt_lb = QLabel(str(cnt))
            cnt_lb.setAlignment(Qt.AlignRight)
            row.addWidget(color_lb)
            row.addWidget(name_lb)
            row.addStretch()
            row.addWidget(cnt_lb)
            box_layout.addLayout(row)
        layout.addWidget(box)

        # 类型分布
        type_names = {v.value: v.value for v in LabelType}
        type_text = " | ".join(
            f"{type_names.get(k, k)}: {v}" for k, v in stats["per_type"].items() if v > 0)
        layout.addWidget(QLabel(f"标注类型分布: {type_text or '无'}"))

        btns = QDialogButtonBox(QDialogButtonBox.Close)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)


class SplitDialog(QDialog):
    """划分训练/验证/测试集。"""

    def __init__(self, parent, dataset: Dataset):
        super().__init__(parent)
        self.dataset = dataset
        self.setWindowTitle("划分训练/验证/测试集")
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        form = QFormLayout()

        self.spin_train = QDoubleSpinBox()
        self.spin_train.setRange(0.1, 0.98)
        self.spin_train.setSingleStep(0.05)
        self.spin_train.setValue(0.8)
        form.addRow("训练集比例:", self.spin_train)

        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.01, 0.5)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.1)
        form.addRow("验证集比例:", self.spin_val)

        self.spin_seed = QSpinBox()
        self.spin_seed.setRange(0, 99999)
        self.spin_seed.setValue(42)
        form.addRow("随机种子:", self.spin_seed)

        self.cb_copy = QCheckBox("复制文件（保留原始图片；移动模式已禁用）")
        self.cb_copy.setChecked(True)
        self.cb_copy.setEnabled(False)
        form.addRow("", self.cb_copy)

        layout.addLayout(form)

        tip = QLabel("划分后生成 images/{train,val,test} 与 labels/{train,val,test}，\n"
                     "并自动生成训练用的 data.yaml")
        tip.setStyleSheet("color: gray; font-size: 12px;")
        layout.addWidget(tip)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _accept(self) -> None:
        tr, va = self.spin_train.value(), self.spin_val.value()
        if tr + va >= 1.0:
            QMessageBox.warning(self, "提示", "训练集 + 验证集比例必须小于 1.0")
            return
        te = round(1.0 - tr - va, 4)
        ret = QMessageBox.question(
            self, "确认划分",
            f"将按 训练 {tr:.0%} / 验证 {va:.0%} / 测试 {te:.0%} 划分数据集，继续？",
        )
        if ret != QMessageBox.Yes:
            return
        try:
            counts = self.dataset.split((tr, va, te), self.spin_seed.value(),
                                        self.cb_copy.isChecked())
            yaml = self.dataset.write_data_yaml()
            QMessageBox.information(
                self, "完成",
                f"划分完成：\n训练集 {counts['train']} 张\n验证集 {counts['val']} 张\n"
                f"测试集 {counts['test']} 张\n\n已生成训练配置: {yaml.name}",
            )
            self.accept()
        except Exception as e:
            QMessageBox.critical(self, "错误", f"划分失败：{e}")


class ConvertVocDialog(QDialog):
    """VOC XML 转 YOLO。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VOC XML 转 YOLO")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        src_row = QHBoxLayout()
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("含 VOC 标注 XML 的目录")
        btn = QPushButton("浏览...")
        btn.clicked.connect(lambda: self._browse(self.src_edit, True))
        src_row.addWidget(self.src_edit)
        src_row.addWidget(btn)
        form.addRow("XML 目录:", src_row)

        self.classes_edit = QLineEdit()
        self.classes_edit.setPlaceholderText("按 YOLO 顺序排列，逗号分隔，如 person, car")
        form.addRow("类别列表:", self.classes_edit)

        layout.addLayout(form)
        layout.addWidget(QLabel("转换结果输出到 XML 目录旁的 yolo_labels/ 文件夹。"))
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse(self, edit: QLineEdit, is_dir: bool) -> None:
        if is_dir:
            path = QFileDialog.getExistingDirectory(self, "选择目录")
        else:
            path, _ = QFileDialog.getOpenFileName(self, "选择文件", "", "JSON (*.json)")
        if path:
            edit.setText(path)

    def source_dir(self) -> str:
        return self.src_edit.text().strip()

    def classes(self) -> list[str]:
        return [c.strip() for c in self.classes_edit.text().replace("，", ",").split(",")
                if c.strip()]

    def _accept(self) -> None:
        if not self.source_dir() or not self.classes():
            QMessageBox.warning(self, "提示", "请填写 XML 目录和类别列表。")
            return
        self.accept()


class ConvertCocoDialog(QDialog):
    """COCO JSON 转 YOLO。"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("COCO JSON 转 YOLO")
        self.setMinimumWidth(480)
        layout = QVBoxLayout(self)
        form = QFormLayout()

        src_row = QHBoxLayout()
        self.src_edit = QLineEdit()
        self.src_edit.setPlaceholderText("COCO 标注 JSON 文件路径")
        btn = QPushButton("浏览...")
        btn.clicked.connect(self._browse)
        src_row.addWidget(self.src_edit)
        src_row.addWidget(btn)
        form.addRow("JSON 文件:", src_row)

        layout.addLayout(form)
        layout.addWidget(QLabel("支持检测框/分割/关键点，输出到 JSON 旁的 yolo_labels/ 文件夹。"))
        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(self._accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def _browse(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "选择 JSON 文件", "", "JSON (*.json)")
        if path:
            self.src_edit.setText(path)

    def source_file(self) -> str:
        return self.src_edit.text().strip()

    def _accept(self) -> None:
        if not self.source_file():
            QMessageBox.warning(self, "提示", "请选择 COCO JSON 文件。")
            return
        self.accept()
