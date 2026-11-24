"""标注数据模型：统一表示矩形框/多边形分割/关键点/OBB 四种任务类型。"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class LabelType(Enum):
    """标注任务类型，对应 YOLO txt 的四种格式。"""
    DETECT = "detect"    # 矩形框
    SEGMENT = "segment"  # 多边形分割
    POSE = "pose"        # 关键点
    OBB = "obb"          # 旋转框


# 类别默认调色板（20 种高对比度颜色，深浅主题下均清晰可见）
CLASS_COLORS = [
    "#e6194b", "#3cb44b", "#ffe119", "#4363d8", "#f58231",
    "#911eb4", "#42d4f4", "#f032e6", "#bfef45", "#fabed4",
    "#469990", "#dcbeff", "#9a6324", "#fffac8", "#800000",
    "#aaffc3", "#808000", "#ffd8b1", "#000075", "#a9a9a9",
]


@dataclass
class LabelItem:
    """单个标注实例。

    points 为绝对像素坐标 (x, y) 列表：
      - DETECT: 2 个点（左上、右下）
      - SEGMENT: N 个顶点
      - OBB: 4 个顶点（顺时针）
    keypoints 仅 POSE 使用：[(x, y, visible), ...]，visible 为 0/1/2
    """
    type: LabelType
    class_id: int
    points: list = field(default_factory=list)
    keypoints: list = field(default_factory=list)

    # ---- 矩形几何辅助（DETECT） ----
    @property
    def rect(self) -> tuple[float, float, float, float]:
        """返回 (x1, y1, x2, y2)，规范化为左上-右下。"""
        (x1, y1), (x2, y2) = self.points[0], self.points[1]
        return (min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2))

    def to_dict(self) -> dict:
        return {
            "type": self.type.value,
            "class_id": self.class_id,
            "points": [[round(x, 2), round(y, 2)] for x, y in self.points],
            "keypoints": [[round(x, 2), round(y, 2), int(v)] for x, y, v in self.keypoints],
        }

    @staticmethod
    def from_dict(d: dict) -> "LabelItem":
        return LabelItem(
            type=LabelType(d["type"]),
            class_id=int(d["class_id"]),
            points=[[float(p[0]), float(p[1])] for p in d.get("points", [])],
            keypoints=[[float(k[0]), float(k[1]), int(k[2])] for k in d.get("keypoints", [])],
        )


@dataclass
class ImageAnnotation:
    """一张图片的全部标注。"""
    image_path: str
    labels: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return {"image_path": self.image_path, "labels": [l.to_dict() for l in self.labels]}

    @staticmethod
    def from_dict(d: dict) -> "ImageAnnotation":
        return ImageAnnotation(
            image_path=d["image_path"],
            labels=[LabelItem.from_dict(l) for l in d.get("labels", [])],
        )


class DatasetMeta:
    """数据集元数据（dataset.json）。"""

    def __init__(self, root: str, classes: list[str], task: LabelType = LabelType.DETECT,
                 pose_kpts: int = 17):
        self.root = root
        self.classes = classes
        self.task = task
        self.pose_kpts = pose_kpts

    def to_dict(self) -> dict:
        return {"classes": self.classes, "task": self.task.value,
                "pose_kpts": self.pose_kpts}

    @staticmethod
    def from_dict(root: str, d: dict) -> "DatasetMeta":
        return DatasetMeta(root, d.get("classes", []), LabelType(d.get("task", "detect")),
                           int(d.get("pose_kpts", 17)))
