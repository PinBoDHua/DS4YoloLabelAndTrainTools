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


def detect_format(values: list[float], prefer_type: LabelType | None = None) -> LabelType | None:
    """根据行 token 数自动识别标注类型。

    values 为 float 列表 [class, 其余值...]。
    prefer_type 参与全部格式判断：
      - 无 prefer_type 时按 token 数自动识别（n==8 的 OBB/4 顶点分割歧义默认按 OBB）；
      - 给定 prefer_type（数据集声明的任务）时，行内容与该任务一致则按该任务解析，
        不一致返回 None（调用方跳过该行并保留原始内容），避免损坏行被误读成其他任务。
    """
    n = len(values) - 1  # 去掉 class
    if prefer_type is not None:
        if prefer_type == LabelType.DETECT:
            return LabelType.DETECT if n == 4 else None
        if prefer_type == LabelType.OBB:
            return LabelType.OBB if n == 8 else None
        if prefer_type == LabelType.SEGMENT:
            return LabelType.SEGMENT if n >= 6 and n % 2 == 0 else None
        if prefer_type == LabelType.POSE:
            return LabelType.POSE if _looks_like_pose(values) else None
        return None
    if n == 4:
        return LabelType.DETECT
    if n == 8:
        return LabelType.OBB
    if _looks_like_pose(values):
        return LabelType.POSE
    if n >= 6 and n % 2 == 0:
        return LabelType.SEGMENT
    return None


def read_yolo_txt(txt_path: str | Path, image_path: str | Path,
                  img_w: int | None = None, img_h: int | None = None,
                  prefer_type: LabelType | None = None) -> ImageAnnotation:
    """读取单个 YOLO txt 为 ImageAnnotation。坐标统一转为绝对像素。

    单行损坏（无法解析/与任务类型不符/坐标数不足）只跳过该行，
    不终止整个文件；被跳过的行保存在 ann.raw_lines，写回时原样保留，
    避免部分读取后自动保存把坏行静默清掉。
    """
    txt_path = Path(txt_path)
    ann = ImageAnnotation(image_path=str(image_path))
    if not txt_path.exists():
        return ann
    if img_w is None or img_h is None:
        img_w, img_h = get_image_size(image_path)

    try:
        # errors="replace"：非 UTF-8 文件不至于整文件读取失败后被空内容覆盖
        text = txt_path.read_text(encoding="utf-8", errors="replace")
    except Exception as e:
        logger.warning("读取标注文件失败 %s: %s", txt_path, e)
        return ann

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        tokens = line.split()
        try:
            values = [float(t) for t in tokens]
            class_id = int(values[0])
            rest = values[1:]
        except ValueError:
            logger.warning("跳过无法解析的行: %s -> %s", txt_path, line[:60])
            ann.raw_lines.append(line)
            continue
        ltype = detect_format(values, prefer_type)
        if ltype is None:
            logger.warning("跳过与数据集任务不符的行: %s -> %s", txt_path, line[:60])
            ann.raw_lines.append(line)
            continue
        try:
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
        except (IndexError, ValueError) as e:
            logger.warning("跳过坐标数不足的行: %s -> %s (%s)", txt_path, line[:60], e)
            ann.raw_lines.append(line)
    return ann


def write_yolo_txt(ann: ImageAnnotation, txt_path: str | Path,
                   img_w: int | None = None, img_h: int | None = None) -> None:
    """将 ImageAnnotation 写为 YOLO txt（归一化坐标）。"""
    txt_path = Path(txt_path)
    txt_path.parent.mkdir(parents=True, exist_ok=True)
    raw_lines = list(getattr(ann, "raw_lines", []) or [])
    if not ann.labels and not raw_lines:
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

    # 读取时无法解析的行原样写回，避免自动保存静默丢弃
    lines.extend(raw_lines)
    txt_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


def _parse_class_token(token: str) -> int | None:
    try:
        return int(token)
    except ValueError:
        try:
            return int(float(token))
        except ValueError:
            return None


def count_class_usage(labels_dir: str | Path) -> dict[int, int]:
    """统计 labels 目录（含划分/嵌套子目录）各 class_id 的标注行数。"""
    counts: dict[int, int] = {}
    for f in Path(labels_dir).rglob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for line in text.splitlines():
            tokens = line.strip().split()
            if not tokens:
                continue
            cid = _parse_class_token(tokens[0])
            if cid is not None:
                counts[cid] = counts.get(cid, 0) + 1
    return counts


def remap_class_ids(labels_dir: str | Path,
                    mapping: dict[int, int | None]) -> tuple[int, int]:
    """按 mapping 重映射所有 label txt 的 class_id。

    mapping: 旧 class_id -> 新 class_id；值为 None 表示删除该标注行。
    只重写有变化的文件；未涉及的行保持原样（含空行/坏行）。
    返回 (重写文件数, 删除行数)。
    """
    rewritten = removed = 0
    for f in Path(labels_dir).rglob("*.txt"):
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.warning("类别重映射跳过无法读取的文件 %s: %s", f, e)
            continue
        out: list[str] = []
        changed = False
        for line in text.splitlines():
            stripped = line.strip()
            if not stripped:
                out.append(line)
                continue
            tokens = stripped.split()
            cid = _parse_class_token(tokens[0]) if tokens else None
            if cid is None or cid not in mapping:
                out.append(line)
                continue
            new_id = mapping[cid]
            if new_id is None:
                changed = True
                removed += 1
                continue
            out.append(" ".join([str(new_id)] + tokens[1:]))
            changed = True
        if changed:
            f.write_text("\n".join(out) + ("\n" if out else ""), encoding="utf-8")
            rewritten += 1
    return rewritten, removed


def label_path_for(image_path: str | Path, labels_dir: str | Path) -> Path:
    """图片路径 -> 对应 label txt 路径（同名换扩展名）。"""
    p = Path(image_path)
    return Path(labels_dir) / (p.stem + ".txt")
