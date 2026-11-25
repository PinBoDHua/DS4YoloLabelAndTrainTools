"""全局配置管理：持久化到项目根目录 config.json，与项目一起分发。"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

# 项目根目录（app/config.py -> 项目根）
ROOT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = ROOT_DIR / "models"
LOGS_DIR = ROOT_DIR / "logs"
# 测试可用 YOLO_LABEL_CONFIG 环境变量重定向配置文件路径（必须在导入前设置），
# 避免写真实项目配置
CONFIG_FILE = Path(os.environ.get("YOLO_LABEL_CONFIG", str(ROOT_DIR / "config.json")))

# 图片过滤条件
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}

DEFAULTS: dict = {
    "theme": "light",              # light / dark
    "recent_projects": [],         # 最近打开的数据集
    "last_dataset": None,          # 上次打开的数据集目录
    "window": None,                # 主窗口几何 {"x","y","w","h"}
    "auto_open_last": True,        # 启动时自动打开上次数据集
    "dataset_state": {},           # 每个数据集的浏览位置 {root: {"index": n, "tool": name}}
    "sam_model": "mobile_sam",     # SAM 模型名（mobile_sam / sam_b）
    "auto_save": True,             # 切换图片自动保存
    "crosshair": False,            # 十字准星
    "show_labels": True,           # 显示类别文本
    "pose_kpts": 17,               # 关键点数量
    "skeleton": "none",            # 关键点骨骼连线（none / coco17）
    "train": {                     # 最近一次训练参数
        "model": "yolov8n.pt",
        "imgsz": 640,
        "epochs": 100,
        "batch": 8,
        "lr0": 0.01,
        "device": "0",
    },
    "prelabel": {
        "conf": 0.25,
        "iou": 0.5,
    },
}


class Config:
    """配置读写封装。所有修改立即落盘。"""

    def __init__(self, path: Path = CONFIG_FILE):
        self._path = path
        self._data: dict = {}
        self.load()

    def load(self) -> None:
        data = {}
        try:
            if self._path.exists():
                data = json.loads(self._path.read_text(encoding="utf-8"))
        except Exception as e:  # 配置损坏时回退默认
            logger.warning("配置文件读取失败，使用默认配置: %s", e)
        merged = json.loads(json.dumps(DEFAULTS))
        merged.update(data or {})
        self._data = merged

    def save(self) -> None:
        try:
            self._path.write_text(
                json.dumps(self._data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
        except Exception as e:
            logger.warning("配置保存失败: %s", e)

    # ---- 通用访问 ----
    def get(self, key: str, default=None):
        return self._data.get(key, default)

    def set(self, key: str, value) -> None:
        self._data[key] = value
        self.save()

    def update(self, d: dict) -> None:
        self._data.update(d)
        self.save()

    # ---- 便捷访问 ----
    @property
    def theme(self) -> str:
        return self._data.get("theme", "light")

    @theme.setter
    def theme(self, v: str) -> None:
        self._data["theme"] = v
        self.save()

    @property
    def recent_projects(self) -> list:
        return self._data.get("recent_projects", [])

    def add_recent(self, path: str) -> None:
        recent = [p for p in self.recent_projects if p != path]
        recent.insert(0, path)
        self._data["recent_projects"] = recent[:10]
        self._data["last_dataset"] = path
        self.save()


# 全局单例
config = Config()
