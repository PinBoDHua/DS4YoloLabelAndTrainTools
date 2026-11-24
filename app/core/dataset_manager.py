"""数据集管理：创建/打开数据集、图片清单、训练集划分、统计、data.yaml 生成。"""
from __future__ import annotations

import json
import logging
import random
import shutil
import tempfile
from pathlib import Path

from app.config import IMG_EXTS
from app.core.label_data import DatasetMeta, LabelType
from app.core.label_io import read_yolo_txt

logger = logging.getLogger(__name__)

DATASET_META_FILE = "dataset.json"


class Dataset:
    """数据集封装。

    目录结构：
      root/
        dataset.json          # 元数据（classes, task）
        images/               # 全部图片
        labels/               # 全部标注 txt（与图片同名）
        (划分后) images/train|val|test、labels/train|val|test
    """

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.images_dir = self.root / "images"
        self.labels_dir = self.root / "labels"
        self.meta = self._load_meta()
        self._images: list[str] = []

    def _load_meta(self) -> DatasetMeta:
        f = self.root / DATASET_META_FILE
        if f.exists():
            try:
                d = json.loads(f.read_text(encoding="utf-8"))
                return DatasetMeta.from_dict(str(self.root), d)
            except Exception as e:
                logger.warning("数据集元数据读取失败: %s", e)
        return DatasetMeta(str(self.root), [])

    def save_meta(self) -> None:
        f = self.root / DATASET_META_FILE
        f.write_text(json.dumps(self.meta.to_dict(), ensure_ascii=False, indent=2),
                     encoding="utf-8")

    # ---------- 图片清单 ----------
    def scan_images(self) -> list[str]:
        """扫描 images 目录（递归，排除 train/val/test 子目录）。"""
        skip_dirs = {"train", "val", "test", "all"}
        paths = []
        if self.images_dir.exists():
            for p in sorted(self.images_dir.rglob("*")):
                if p.suffix.lower() in IMG_EXTS:
                    # 排除划分后的子目录
                    rel_parts = p.relative_to(self.images_dir).parts
                    if rel_parts and rel_parts[0] in skip_dirs:
                        continue
                    paths.append(str(p))
        self._images = paths
        return paths

    @property
    def images(self) -> list[str]:
        return self._images

    def label_path(self, image_path: str) -> Path:
        """图片对应的标注 txt 路径（保持相对子目录结构）。"""
        p = Path(image_path)
        rel = p.relative_to(self.images_dir)
        return self.labels_dir / rel.with_suffix(".txt")

    def labeled_set(self) -> set[str]:
        """已有非空标注的图片集合。"""
        labeled = set()
        for img in self._images:
            lp = self.label_path(img)
            if lp.exists() and lp.stat().st_size > 0:
                labeled.add(img)
        return labeled

    def read_annotation(self, image_path: str):
        # 传入数据集任务类型以消除 OBB/4 顶点分割的格式歧义
        return read_yolo_txt(self.label_path(image_path), image_path,
                             prefer_type=self.meta.task)

    # ---------- 训练划分 ----------
    def split(self, ratios: tuple[float, float, float], seed: int = 42,
              copy: bool = True) -> dict:
        """划分 train/val/test（比例如 0.8/0.1/0.1）。返回各集合图片数。

        划分后目录：images/{train,val,test}、labels/{train,val,test}。
        """
        if not copy:
            raise ValueError("为避免破坏原始数据，仅支持复制式划分")

        images = self.scan_images()
        if not images:
            raise ValueError("数据集中没有图片，请先导入图片")

        n = len(images)
        if n < 2:
            raise ValueError(
                f"图片数量不足（当前 {n} 张），至少需要 2 张图片才能划分训练/验证集，请先导入更多图片")

        rng = random.Random(seed)
        shuffled = list(images)
        rng.shuffle(shuffled)
        n_train = int(n * ratios[0])
        n_val = int(n * ratios[1])
        # 保证 train/val 至少各 1 张：ultralytics 要求 train、val 目录均非空，
        # 否则（如图片过少导致 val 集为 0 张）训练时会抛 FileNotFoundError。
        n_train = max(n_train, 1)
        n_val = max(n_val, 1)
        if n_train + n_val > n:
            # 总数不足以同时满足 train/val 各至少 1 张，test 置空，优先保证 val
            n_val = max(1, n - n_train)
            n_train = n - n_val
        train = shuffled[:n_train]
        val = shuffled[n_train:n_train + n_val]
        test = shuffled[n_train + n_val:]

        work_dir = Path(tempfile.mkdtemp(prefix=".split-", dir=self.root))
        counts = {}
        keep_work_dir = False
        try:
            staged_images = work_dir / "images"
            staged_labels = work_dir / "labels"
            for split_name, split_imgs in (("train", train), ("val", val), ("test", test)):
                (staged_images / split_name).mkdir(parents=True, exist_ok=True)
                (staged_labels / split_name).mkdir(parents=True, exist_ok=True)
                for img in split_imgs:
                    src = Path(img)
                    rel = src.relative_to(self.images_dir)
                    dst = staged_images / split_name / rel
                    dst.parent.mkdir(parents=True, exist_ok=True)
                    self._place(src, dst, copy)

                    label_rel = rel.with_suffix(".txt")
                    label_dst = staged_labels / split_name / label_rel
                    label_dst.parent.mkdir(parents=True, exist_ok=True)
                    label_src = self.label_path(img)
                    if label_src.exists():
                        self._place(label_src, label_dst, copy)
                    else:
                        label_dst.write_text("", encoding="utf-8")
                counts[split_name] = len(split_imgs)

            affected = []
            try:
                for split_name in ("train", "val", "test"):
                    for kind, target_root, staged_root in (
                            ("images", self.images_dir, staged_images),
                            ("labels", self.labels_dir, staged_labels)):
                        target = target_root / split_name
                        staged = staged_root / split_name
                        backup = work_dir / "backup" / kind / split_name
                        if target.exists():
                            backup.parent.mkdir(parents=True, exist_ok=True)
                            target.rename(backup)
                        affected.append((target, backup))
                        staged.rename(target)
            except Exception as commit_error:
                rollback_errors = []
                for target, backup in reversed(affected):
                    try:
                        if target.exists():
                            shutil.rmtree(target)
                        if backup.exists():
                            backup.rename(target)
                    except Exception as rollback_error:
                        rollback_errors.append(rollback_error)
                if rollback_errors:
                    keep_work_dir = True
                    raise OSError(
                        f"划分提交及回滚失败，旧数据备份保留在: {work_dir}") from commit_error
                raise
        finally:
            if not keep_work_dir:
                shutil.rmtree(work_dir, ignore_errors=True)
        logger.info("数据集划分完成: %s", counts)
        return counts

    @staticmethod
    def _place(src: Path, dst: Path, copy: bool) -> None:
        if copy:
            shutil.copy2(src, dst)
        else:
            shutil.move(str(src), str(dst))

    def write_data_yaml(self, yaml_path: str | Path | None = None) -> Path:
        """生成 ultralytics 训练用的 data.yaml。"""
        if yaml_path is None:
            yaml_path = self.root / "data.yaml"
        yaml_path = Path(yaml_path)
        # 路径使用正斜杠绝对路径
        img = self.images_dir.as_posix()
        lbl = self.labels_dir.as_posix()
        lines = [
            f"path: {self.root.as_posix()}",
            f"train: {img}/train",
            f"val: {img}/val",
        ]
        if (self.images_dir / "test").exists():
            lines.append(f"test: {img}/test")
        if self.meta.task == LabelType.POSE:
            lines.append(f"kpt_shape: [{self.meta.pose_kpts}, 3]")
        lines.append(f"names:")
        for i, name in enumerate(self.meta.classes):
            lines.append(f"  {i}: {name}")
        yaml_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return yaml_path

    # ---------- 统计 ----------
    def stats(self) -> dict:
        """统计：图片数、已标注数、每类实例数、标注类型分布。"""
        images = self.scan_images()
        labeled = self.labeled_set()
        per_class = {i: 0 for i in range(len(self.meta.classes))}
        per_type = {t.value: 0 for t in LabelType}
        for img in images:
            ann = self.read_annotation(img)
            for lb in ann.labels:
                per_class[lb.class_id] = per_class.get(lb.class_id, 0) + 1
                per_type[lb.type.value] += 1
        return {
            "total": len(images),
            "labeled": len(labeled),
            "unlabeled": len(images) - len(labeled),
            "per_class": per_class,
            "per_type": per_type,
        }

    def import_images(self, source_paths: list[str], copy: bool = True) -> tuple[int, int, int]:
        """导入外部图片到数据集 images/。

        返回 (成功数, 重名跳过数, 失败数)。重名时自动添加序号重命名。
        """
        self.images_dir.mkdir(parents=True, exist_ok=True)
        self.labels_dir.mkdir(parents=True, exist_ok=True)
        imported = skipped = failed = 0
        for src in source_paths:
            src_p = Path(src)
            if not src_p.exists():
                failed += 1
                continue
            # 目标重名时自动追加序号，避免静默跳过导致困惑
            dst = self.images_dir / src_p.name
            if dst.exists() and src_p.resolve() != dst.resolve():
                stem, suffix = src_p.stem, src_p.suffix
                i = 1
                while dst.exists():
                    dst = self.images_dir / f"{stem}_{i}{suffix}"
                    i += 1
            try:
                if copy:
                    shutil.copy2(src_p, dst)
                else:
                    shutil.move(str(src_p), str(dst))
                imported += 1
            except Exception as e:
                logger.warning("导入图片失败 %s: %s", src, e)
                failed += 1
        return imported, skipped, failed


def create_dataset(root: str | Path, classes: list[str],
                   task: LabelType = LabelType.DETECT, pose_kpts: int = 17) -> Dataset:
    """创建新数据集（空目录结构 + 元数据）。"""
    root = Path(root)
    (root / "images").mkdir(parents=True, exist_ok=True)
    (root / "labels").mkdir(parents=True, exist_ok=True)
    ds = Dataset(root)
    ds.meta = DatasetMeta(str(root), classes, task, pose_kpts)
    ds.save_meta()
    return ds
