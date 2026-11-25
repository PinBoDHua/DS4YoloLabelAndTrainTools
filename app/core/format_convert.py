"""格式转换：VOC XML / COCO JSON ↔ YOLO，以及标注可视化导出。"""
from __future__ import annotations

import json
import logging
import xml.etree.ElementTree as ET
from pathlib import Path

from app.core.label_data import LabelItem, LabelType
from app.core.label_io import write_yolo_txt

logger = logging.getLogger(__name__)


def voc_to_yolo(xml_dir: str | Path, labels_out: str | Path,
                classes: list[str], image_size: tuple[int, int] | None = None,
                progress_cb=None) -> tuple[int, int]:
    """VOC 标注 XML 目录转 YOLO txt。

    image_size 可选 (w, h)；未提供时从 XML 的 size 节点读取。
    返回 (成功数, 失败数)。
    """
    xml_dir, labels_out = Path(xml_dir), Path(labels_out)
    labels_out.mkdir(parents=True, exist_ok=True)
    cls_index = {name: i for i, name in enumerate(classes)}
    ok = fail = 0
    xmls = sorted(xml_dir.glob("*.xml"))
    for idx, xf in enumerate(xmls):
        try:
            # 安全防护：拒绝含 DOCTYPE/实体的 XML，防实体扩展攻击
            raw = xf.read_text(encoding="utf-8", errors="ignore")
            if "<!DOCTYPE" in raw.upper() or "<!ENTITY" in raw.upper():
                logger.warning("跳过含 DTD/实体的 XML: %s", xf.name)
                fail += 1
                continue
            tree = ET.fromstring(raw)
            root = tree
            if image_size is None:
                size_node = root.find("size")
                w = float(size_node.find("width").text)
                h = float(size_node.find("height").text)
            else:
                w, h = image_size
            lines = []
            for obj in root.findall("object"):
                name = obj.find("name").text.strip()
                if name not in cls_index:
                    logger.warning("跳过未知类别 '%s' (文件 %s)", name, xf.name)
                    continue
                b = obj.find("bndbox")
                x1 = float(b.find("xmin").text)
                y1 = float(b.find("ymin").text)
                x2 = float(b.find("xmax").text)
                y2 = float(b.find("ymax").text)
                cx = (x1 + x2) / 2 / w
                cy = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_index[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
            (labels_out / (xf.stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            ok += 1
        except Exception as e:
            logger.warning("转换失败 %s: %s", xf, e)
            fail += 1
        if progress_cb and idx % 10 == 0:
            progress_cb(idx + 1, len(xmls))
    if progress_cb:
        progress_cb(len(xmls), len(xmls))
    return ok, fail


def coco_to_yolo(json_path: str | Path, labels_out: str | Path,
                 progress_cb=None) -> tuple[int, int, list[str]]:
    """COCO 标注 JSON 转 YOLO txt（检测/分割/关键点自动识别）。

    返回 (成功数, 失败数, 类别列表)。
    """
    json_path, labels_out = Path(json_path), Path(labels_out)
    labels_out.mkdir(parents=True, exist_ok=True)
    data = json.loads(json_path.read_text(encoding="utf-8"))

    # COCO category_id 可为任意非连续整数（如 1/3/90），必须映射为
    # 按 categories 顺序的连续 YOLO 索引，否则写出的 class 与类别表错位
    categories = data.get("categories", [])
    cat_index = {int(c["id"]): i for i, c in enumerate(categories)}
    classes = [c["name"] for c in categories]
    images = {im["id"]: im for im in data.get("images", [])}

    ok = fail = 0
    anns_by_img: dict[int, list] = {}
    for ann in data.get("annotations", []):
        anns_by_img.setdefault(ann["image_id"], []).append(ann)

    total = len(images)
    for idx, (img_id, im) in enumerate(images.items()):
        try:
            w, h = im["width"], im["height"]
            lines = []
            for ann in anns_by_img.get(img_id, []):
                try:
                    cid = cat_index.get(int(ann["category_id"]))
                except (KeyError, TypeError, ValueError):
                    cid = None
                if cid is None:
                    logger.warning("跳过未知 category_id: %s", ann.get("category_id"))
                    continue
                # 关键点优先：同时含分割与关键点的姿态标注按 POSE 输出
                # （box + kpts），若先走分割分支会丢失关键点
                if ann.get("keypoints"):
                    kpts = ann["keypoints"]
                    x1, y1, bw, bh = ann["bbox"]
                    cx = (x1 + bw / 2) / w
                    cy = (y1 + bh / 2) / h
                    norm_k = []
                    for i in range(0, len(kpts), 3):
                        norm_k += [f"{kpts[i] / w:.6f}", f"{kpts[i + 1] / h:.6f}",
                                   str(int(kpts[i + 2]))]
                    lines.append(
                        f"{cid} {cx:.6f} {cy:.6f} {bw / w:.6f} {bh / h:.6f} " + " ".join(norm_k)
                    )
                    continue
                if ann.get("segmentation"):
                    seg = ann["segmentation"]
                    if isinstance(seg, list) and seg and isinstance(seg[0], list):
                        pts = seg[0]  # 取最大轮廓的第一个多边形
                    else:
                        pts = seg
                    if isinstance(pts, list) and len(pts) >= 6:
                        norm = [f"{v / w:.6f}" if i % 2 == 0 else f"{v / h:.6f}"
                                for i, v in enumerate(pts)]
                        lines.append(f"{cid} " + " ".join(norm))
                        continue
                # 检测框
                x1, y1, bw, bh = ann["bbox"]
                lines.append(
                    f"{cid} {(x1 + bw / 2) / w:.6f} {(y1 + bh / 2) / h:.6f} "
                    f"{bw / w:.6f} {bh / h:.6f}"
                )
            (labels_out / (Path(im["file_name"]).stem + ".txt")).write_text(
                "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
            ok += 1
        except Exception as e:
            logger.warning("转换失败 image_id=%s: %s", img_id, e)
            fail += 1
        if progress_cb and idx % 50 == 0:
            progress_cb(idx + 1, total)
    if progress_cb:
        progress_cb(total, total)
    return ok, fail, classes


def export_visualization(image_path: str | Path, labels_dir: str | Path,
                         out_path: str | Path, classes: list[str],
                         task: LabelType = LabelType.DETECT) -> Path:
    """导出标注可视化图（画框 + 类名）。返回输出路径。"""
    import cv2
    import numpy as np
    from PIL import Image as PILImage

    from app.core.label_data import CLASS_COLORS
    from app.core.label_io import label_path_for, read_yolo_txt

    image_path, out_path = Path(image_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # 用 PIL 读取（cv2.imread/np.fromfile 对中文路径不兼容），转为 BGR 供 cv2 使用
    rgb = np.asarray(PILImage.open(image_path).convert("RGB"))
    img = rgb[:, :, ::-1]
    h, w = img.shape[:2]
    ann = read_yolo_txt(label_path_for(image_path, labels_dir), image_path, w, h)

    def _hex_to_bgr(hex_s: str):
        hx = hex_s.lstrip("#")
        r, g, b = int(hx[0:2], 16), int(hx[2:4], 16), int(hx[4:6], 16)
        return (b, g, r)

    for lb in ann.labels:
        color = _hex_to_bgr(CLASS_COLORS[lb.class_id % len(CLASS_COLORS)])
        name = classes[lb.class_id] if lb.class_id < len(classes) else str(lb.class_id)
        if lb.type == LabelType.DETECT:
            x1, y1, x2, y2 = [int(v) for v in lb.rect]
            cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img, name, (x1, max(y1 - 4, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, color, 2)
        else:
            pts = [[int(x), int(y)] for x, y in lb.points]
            cv2.polylines(img, [np.array(pts)], True, color, 2)
            cv2.putText(img, name, tuple(pts[0]), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    ext = out_path.suffix or ".jpg"
    ok, buf = cv2.imencode(ext, img)
    if ok:
        # 用 PIL 写盘（cv2.imwrite/np.tofile 对中文路径不兼容）
        arr = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        PILImage.fromarray(arr).save(out_path)
    return out_path
