"""模型预标注引擎：用已训练模型批量推理生成可编辑标注。"""
from __future__ import annotations

import logging
from pathlib import Path

from PySide6.QtCore import QThread, Signal

from app.core.label_data import LabelItem, LabelType
from app.core.label_io import get_image_size, write_yolo_txt

logger = logging.getLogger(__name__)


class PrelabelWorker(QThread):
    """批量预标注线程。"""
    progress = Signal(int, int)      # (当前, 总数)
    finished_ok = Signal(int, int, int)   # (成功数, 跳过数, 失败数)
    failed = Signal(str)

    def __init__(self, model_path: str, image_paths: list[str], labels_dir: str,
                 conf: float = 0.25, iou: float = 0.5, imgsz: int = 640,
                 device: str = "0", skip_existing: bool = False):
        super().__init__()
        self._model_path = model_path
        self._images = image_paths
        self._labels_dir = Path(labels_dir)
        self._conf = conf
        self._iou = iou
        self._imgsz = imgsz
        self._device = device
        self._skip = skip_existing
        self._cancel = False

    def cancel(self) -> None:
        self._cancel = True

    def run(self) -> None:
        try:
            from app.utils import patch_opencv_imread
            patch_opencv_imread()  # 中文路径图片支持
            from ultralytics import YOLO
            model = YOLO(self._model_path)
            ok = skip = errors = 0
            total = len(self._images)
            for i, img in enumerate(self._images):
                if self._cancel:
                    break
                p = Path(img)
                images_dir = self._labels_dir.parent / "images"
                txt_out = self._labels_dir / p.relative_to(images_dir).with_suffix(".txt")
                if self._skip and txt_out.exists() and txt_out.stat().st_size > 0:
                    skip += 1
                    self.progress.emit(i + 1, total)
                    continue
                try:
                    results = model.predict(
                        source=img, conf=self._conf, iou=self._iou,
                        imgsz=self._imgsz, device=self._device, verbose=False,
                    )
                    labels = self._extract(results[0], self._conf)
                    write_yolo_txt(self._make_ann(img, labels), txt_out)
                    ok += 1
                except Exception as e:
                    errors += 1
                    logger.warning("预标注失败 %s: %s", img, e)
                self.progress.emit(i + 1, total)
            self.finished_ok.emit(ok, skip, errors)
        except Exception as e:
            logger.exception("预标注引擎错误")
            self.failed.emit(str(e))

    @staticmethod
    def _extract(result, conf: float = 0.25) -> list[LabelItem]:
        """从 ultralytics 结果提取 LabelItem 列表（自动识别任务类型）。"""
        labels: list[LabelItem] = []
        img_w, img_h = result.orig_shape[1], result.orig_shape[0]
        names = result.names

        # 关键点任务
        if getattr(result, "keypoints", None) is not None and result.keypoints.data is not None \
                and len(result.keypoints.data):
            boxes = result.boxes
            for box, kpts in zip(boxes, result.keypoints.data):
                cls = int(box.cls[0])
                x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
                kp_list = []
                for k in kpts.tolist():
                    visibility = 2 if float(k[2]) >= conf else 0
                    kp_list.append([float(k[0]), float(k[1]), visibility])
                labels.append(LabelItem(LabelType.POSE, cls,
                                        points=[[x1, y1], [x2, y2]], keypoints=kp_list))
            return labels

        # 分割任务
        if getattr(result, "masks", None) is not None and result.masks is not None:
            boxes = result.boxes
            for box, mask in zip(boxes, result.masks.data if result.masks.data is not None else []):
                cls = int(box.cls[0])
                poly = _mask_to_polygon(mask.cpu().numpy())
                if poly:
                    labels.append(LabelItem(LabelType.SEGMENT, cls, points=poly))
            return labels

        # OBB 任务
        if getattr(result, "obb", None) is not None and result.obb is not None \
                and len(result.obb.data):
            for pts, cls in zip(result.obb.xyxyxyxy, result.obb.cls):
                labels.append(LabelItem(
                    LabelType.OBB, int(cls), points=pts.tolist()))
            return labels

        # 检测任务
        for box in result.boxes:
            cls = int(box.cls[0])
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            labels.append(LabelItem(LabelType.DETECT, cls, points=[[x1, y1], [x2, y2]]))
        return labels

    @staticmethod
    def _make_ann(img: str, labels: list[LabelItem]):
        from app.core.label_data import ImageAnnotation
        return ImageAnnotation(image_path=img, labels=labels)


def _mask_to_polygon(mask, epsilon_ratio: float = 0.004) -> list:
    """掩码数组 → 多边形顶点（绝对像素）。"""
    import cv2
    mask_u8 = (mask > 0.5).astype("uint8") * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return []
    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < 64:
        return []
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon_ratio * peri, True)
    pts = [[float(p[0][0]), float(p[0][1])] for p in approx]
    return pts if len(pts) >= 3 else []
