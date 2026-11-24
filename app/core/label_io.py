"""YOLO txt 格式读写：检测/分割/OBB/关键点四种任务，自动识别格式。"""
from __future__ import annotations

import logging
from pathlib import Path

from PIL import Image

from app.core.label_data import ImageAnnotation, LabelItem, LabelType

logger = logging.getLogger(__name__)

# 已知关键点任务的标准骨架数（17: COCO 人体姿态，其余按 3 的倍数通用判定）
_POSE_KPT_COUNTS = {17, 13, 22, 23, 4, 6, 8, 10, 12, 14, 16, 18, 21, 26, 32, 34, 68}


def get_image_size(image_path: str | Path) -> tuple[int, int]:
    """读取图片宽高（PIL 无中文路径问题，只读头部速度快）。"""
    with Image.open(image_path) as im:
        return im.size  # (w, h)


def _looks_like_pose(values: list[float]) -> bool:
    """判断 float 序列是否符合关键点格式（box + 每组 3 个值，可见性为 0/1/2）。"""
    n = len(values) - 1  # 去掉 class
    if n < 7 or (n - 4) % 3 != 0:
        return False
    # v 值位于 class(0) + box(1..4) + kx,ky 之后：索引 7, 10, 13...
    for idx in range(7, len(values), 3):
        v = values[idx]
        if v not in (0.0, 1.0, 2.0):
            return False
    return True


def detect_format(values: list[float], prefer_type: LabelType | None = None) -> LabelType:
    """根据行 token 数自动识别标注类型。

    values 为 float 列表 [class, 其余值...]。
    prefer_type 用于消除歧义：OBB 与 4 顶点分割多边形格式完全相同（8 个坐标），
    此时以数据集声明的任务类型为准。
    """
    n = len(values) - 1  # 去掉 class
    if n == 4:
        return LabelType.DETECT
    if n == 8:
        # 歧义：OBB 还是 4 顶点分割？优先数据集任务类型
        if prefer_type in (LabelType.SEGMENT, LabelType.OBB):
            return prefer_type
        return LabelType.OBB
    if _looks_like_pose(values):
        return LabelType.POSE
    if n >= 6 and n % 2 == 0:
        return LabelType.SEGMENT
    return LabelType.SEGMENT  # 兜底


def read_yolo_txt(txt_path: str | Path, image_path: str | Path,
                  img_w: int | None = None, img_h: int | None = None,
                  prefer_type: LabelType | None = None) -> ImageAnnotation:
    """读取单个 YOLO txt 为 ImageAnnotation。坐标统一转为绝对像素。"""
    txt_path = Path(txt_path)
    ann = ImageAnnotation(image_path=str(image_path))
    if not txt_path.exists():
        return ann
    if img_w is None or img_h is None:
        img_w, img_h = get_image_size(image_path)

    try:
        for line in txt_path.read_text(encoding="utf-8").strip().splitlines():
            line = line.strip()
            if not line:
                continue
            tokens = line.split()
            try:
                values = [float(t) for t in tokens]
            except ValueError:
                logger.warning("跳过无法解析的行: %s -> %s", txt_path, line[:60])
                continue
            class_id = int(values[0])
            rest = values[1:]
            ltype = detect_format(values, prefer_type)

            if ltype == LabelType.DETECT:
                cx, cy, w, h = rest
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                ann.labels.append(LabelItem(ltype, class_id, points=[[x1, y1], [x2, y2]]))
            elif ltype == LabelType.SEGMENT:
                pts = [[rest[i] * img_w, rest[i + 1] * img_h]
                       for i in range(0, len(rest), 2)]
                ann.labels.append(LabelItem(ltype, class_id, points=pts))
            elif ltype == LabelType.OBB:
                pts = [[rest[i] * img_w, rest[i + 1] * img_h]
                       for i in range(0, len(rest), 2)]
                ann.labels.append(LabelItem(ltype, class_id, points=pts))
            elif ltype == LabelType.POSE:
                cx, cy, w, h = rest[:4]
                x1 = (cx - w / 2) * img_w
                y1 = (cy - h / 2) * img_h
                x2 = (cx + w / 2) * img_w
                y2 = (cy + h / 2) * img_h
                kpts = []
                krest = rest[4:]
                for i in range(0, len(krest), 3):
                    kx, ky, v = krest[i], krest[i + 1], int(krest[i + 2])
                    kpts.append([kx * img_w, ky * img_h, v])
                ann.labels.append(
                    LabelItem(ltype, class_id, points=[[x1, y1], [x2, y2]], keypoints=kpts)
                )
    except Exception as e:
        logger.warning("读取标注文件失败 %s: %s", txt_path, e)
    return ann


def write_yolo_txt(ann: ImageAnnotation, txt_path: str | Path,
                   img_w: int | None = None, img_h: int | None = None) -> None:
    """将 ImageAnnotation 写为 YOLO txt（归一化坐标）。"""
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    if not ann.labels:
        # 无标注时写空文件（保留占位，表明该图已处理过）
        txt_path.write_text("", encoding="utf-8")
        return
    if img_w is None or img_h is None:
        img_w, img_h = get_image_size(ann.image_path)

    lines = []
    for lb in ann.labels:
        if not lb.points:
            continue
        parts = [str(lb.class_id)]
        if lb.type == LabelType.DETECT:
            x1, y1, x2, y2 = lb.rect
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = max((x2 - x1) / img_w, 1e-6)
            h = max((y2 - y1) / img_h, 1e-6)
            parts += [f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
        elif lb.type in (LabelType.SEGMENT, LabelType.OBB):
            for x, y in lb.points:
                parts += [f"{min(max(x / img_w, 0), 1):.6f}",
                          f"{min(max(y / img_h, 0), 1):.6f}"]
        elif lb.type == LabelType.POSE:
            x1, y1, x2, y2 = lb.rect
            cx = (x1 + x2) / 2 / img_w
            cy = (y1 + y2) / 2 / img_h
            w = max((x2 - x1) / img_w, 1e-6)
            h = max((y2 - y1) / img_h, 1e-6)
            parts += [f"{cx:.6f}", f"{cy:.6f}", f"{w:.6f}", f"{h:.6f}"]
            for kx, ky, v in lb.keypoints:
                parts += [f"{min(max(kx / img_w, 0), 1):.6f}",
                          f"{min(max(ky / img_h, 0), 1):.6f}", str(int(v))]
        lines.append(" ".join(parts))

    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def label_path_for(image_path: str | Path, labels_dir: str | Path) -> Path:
    """图片路径 -> 对应 label txt 路径（同名换扩展名）。"""
    p = Path(image_path)
    return Path(labels_dir) / (p.stem + ".txt")
