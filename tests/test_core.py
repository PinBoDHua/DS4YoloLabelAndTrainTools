"""核心逻辑自测：四种 YOLO 格式读写 + 数据集创建/切分/统计。"""
import sys
import tempfile
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.core.label_data import ImageAnnotation, LabelItem, LabelType
from app.core.label_io import read_yolo_txt, write_yolo_txt
from app.core.dataset_manager import Dataset, create_dataset
from app.core.prelabel_engine import PrelabelWorker


def make_test_image(path: Path, size=(640, 480)):
    Image.new("RGB", size, (128, 128, 128)).save(path)


def test_label_io():
    tmp = Path(tempfile.mkdtemp())
    img = tmp / "test.png"
    make_test_image(img)

    cases = [
        (ImageAnnotation(str(img), [
            LabelItem(LabelType.DETECT, 0, points=[[100, 80], [300, 240]])]), "detect"),
        (ImageAnnotation(str(img), [
            LabelItem(LabelType.SEGMENT, 1, points=[[10, 10], [100, 10], [100, 100], [10, 100]])]),
         "segment"),
        (ImageAnnotation(str(img), [
            LabelItem(LabelType.OBB, 2, points=[[50, 50], [150, 40], [160, 140], [60, 150]])]), "obb"),
        (ImageAnnotation(str(img), [
            LabelItem(LabelType.POSE, 3, points=[[100, 80], [300, 240]],
                      keypoints=[[110, 100, 2], [120, 110, 1], [130, 120, 0]])]), "pose"),
    ]
    prefer_map = {"detect": None, "segment": LabelType.SEGMENT,
                  "obb": LabelType.OBB, "pose": None}
    for ann, name in cases:
        txt = tmp / f"{name}.txt"
        write_yolo_txt(ann, txt)
        loaded = read_yolo_txt(txt, img, prefer_type=prefer_map[name])
        assert len(loaded.labels) == 1, f"{name}: 数量不匹配"
        lb, lb0 = loaded.labels[0], ann.labels[0]
        assert lb.type == lb0.type, f"{name}: 类型不匹配 {lb.type} vs {lb0.type}"
        assert lb.class_id == lb0.class_id, f"{name}: 类别不匹配"
        for (x1, y1), (x2, y2) in zip(lb.points, lb0.points):
            assert abs(x1 - x2) < 0.5 and abs(y1 - y2) < 0.5, f"{name}: 坐标误差 {(x1,y1)} vs {(x2,y2)}"
        if lb0.keypoints:
            for k1, k2 in zip(lb.keypoints, lb0.keypoints):
                assert abs(k1[0] - k2[0]) < 0.5 and abs(k1[1] - k2[1]) < 0.5 and k1[2] == k2[2], \
                    f"{name}: 关键点误差"
        print(f"  [OK] {name} 格式读写一致")
    # 空标注
    ann_empty = ImageAnnotation(str(img), [])
    write_yolo_txt(ann_empty, tmp / "empty.txt")
    loaded = read_yolo_txt(tmp / "empty.txt", img)
    assert not loaded.labels
    print("  [OK] 空标注处理正常")


def test_dataset():
    tmp = Path(tempfile.mkdtemp())
    ds_root = tmp / "mydataset"
    ds = create_dataset(ds_root, ["cat", "dog"], LabelType.DETECT)
    assert (ds_root / "images").exists() and (ds_root / "dataset.json").exists()

    # 导入 5 张图并标注 3 张
    for i in range(5):
        p = ds_root / "images" / f"img{i}.jpg"
        make_test_image(p)
    ds.scan_images()
    assert len(ds.images) == 5
    for i in range(3):
        lp = ds.label_path(ds.images[i])
        write_yolo_txt(ImageAnnotation(ds.images[i], [
            LabelItem(LabelType.DETECT, i % 2, points=[[10, 10], [50, 50]])]), lp)
    labeled = ds.labeled_set()
    assert len(labeled) == 3, f"应标注 3 张，实际 {len(labeled)}"
    print(f"  [OK] 图片扫描与已标注检测 ({len(labeled)}/5)")

    # 划分 8:1:1
    counts = ds.split((0.8, 0.1, 0.1), seed=42, copy=True)
    assert sum(counts.values()) == 5
    assert (ds.images_dir / "train").exists()
    train_imgs = list((ds.images_dir / "train").glob("*.jpg"))
    assert len(train_imgs) == 4, f"train 应有 4 张，实际 {len(train_imgs)}"
    # 每个 train 图都有对应 label txt
    for p in train_imgs:
        assert (ds.labels_dir / "train" / (p.stem + ".txt")).exists(), f"缺少 label: {p}"
    print(f"  [OK] 划分 train/val/test: {counts}")

    # 统计
    stats = ds.stats()
    assert stats["total"] == 5 and stats["labeled"] == 3
    assert sum(stats["per_class"].values()) == 3
    print(f"  [OK] 统计: {stats['total']} 张, 已标注 {stats['labeled']}, "
          f"每类 {stats['per_class']}")

    # data.yaml
    yaml = ds.write_data_yaml()
    assert yaml.exists() and "names:" in yaml.read_text(encoding="utf-8")
    print(f"  [OK] data.yaml 生成: {yaml}")


def test_split_preserves_nested_paths():
    tmp = Path(tempfile.mkdtemp())
    ds = create_dataset(tmp / "nested", ["object"], LabelType.DETECT)
    for folder in ("a", "b", "c", "d"):
        image = ds.images_dir / folder / "same.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        make_test_image(image, (16, 16))
    counts = ds.split((0.75, 0.25, 0), copy=True)
    placed = []
    for split_name in ("train", "val", "test"):
        placed.extend((ds.images_dir / split_name).rglob("*.jpg"))
    assert len(placed) == 4, f"嵌套同名图片被覆盖: {counts}, 实际 {len(placed)}"
    for folder in ("a", "b", "c", "d"):
        matches = [p for p in placed if p.relative_to(ds.images_dir).parts[1:] == (folder, "same.jpg")]
        assert len(matches) == 1, f"未保留相对目录: {folder}/same.jpg"
    print("  [OK] 划分保留嵌套相对路径")


def test_split_rejects_move():
    tmp = Path(tempfile.mkdtemp())
    ds = create_dataset(tmp / "move", ["object"], LabelType.DETECT)
    for name in ("a.jpg", "b.jpg"):
        make_test_image(ds.images_dir / name, (16, 16))
    try:
        ds.split((0.5, 0.5, 0), copy=False)
    except ValueError:
        pass
    else:
        raise AssertionError("移动式划分应被拒绝")
    assert len(ds.scan_images()) == 2, "拒绝移动划分不应改变原始图片"
    print("  [OK] 移动式划分被安全拒绝")


def test_split_failure_preserves_existing():
    tmp = Path(tempfile.mkdtemp())
    ds = create_dataset(tmp / "atomic", ["object"], LabelType.DETECT)
    for name in ("a.jpg", "b.jpg"):
        make_test_image(ds.images_dir / name, (16, 16))
    old_split = ds.images_dir / "train" / "old.txt"
    old_split.parent.mkdir(parents=True, exist_ok=True)
    old_split.write_text("old", encoding="utf-8")
    original_place = Dataset._place
    calls = {"count": 0}

    def fail_second_place(src, dst, copy):
        calls["count"] += 1
        if calls["count"] == 2:
            raise OSError("模拟复制失败")
        return original_place(src, dst, copy)

    with patch.object(Dataset, "_place", side_effect=fail_second_place):
        try:
            ds.split((0.5, 0.5, 0), copy=True)
        except OSError:
            pass
        else:
            raise AssertionError("模拟复制失败应向上抛出")
    assert old_split.exists(), "失败划分删除了旧结果"
    assert old_split.read_text(encoding="utf-8") == "old", "失败划分破坏了旧结果"
    print("  [OK] 失败划分保留旧结果")


def test_split_preserves_backup_when_rollback_fails():
    tmp = Path(tempfile.mkdtemp())
    ds = create_dataset(tmp / "rollback", ["object"], LabelType.DETECT)
    for name in ("a.jpg", "b.jpg"):
        make_test_image(ds.images_dir / name, (16, 16))
    old_dir = ds.images_dir / "train"
    old_dir.mkdir(parents=True, exist_ok=True)
    (old_dir / "old.txt").write_text("old", encoding="utf-8")
    original_rename = Path.rename
    failure = {"staged": False}

    def fail_commit_and_restore(self, target):
        target = Path(target)
        if target == old_dir and ".split-" in str(self):
            if "backup" in self.parts:
                if failure["staged"]:
                    raise OSError("模拟恢复失败")
            else:
                failure["staged"] = True
                raise OSError("模拟提交失败")
        return original_rename(self, target)

    with patch.object(Path, "rename", new=fail_commit_and_restore):
        try:
            ds.split((0.5, 0.5, 0), copy=True)
        except OSError:
            pass
        else:
            raise AssertionError("模拟提交失败应向上抛出")
    backups = list(ds.root.glob(".split-*/backup/images/train/old.txt"))
    assert old_dir.exists() or backups, "回滚失败时唯一旧划分备份被删除"
    print("  [OK] 回滚失败保留旧划分备份")


def test_prelabel_preserves_nested_paths():
    tmp = Path(tempfile.mkdtemp())
    ds = create_dataset(tmp / "prelabel", ["object"], LabelType.DETECT)
    images = []
    for folder in ("a", "b"):
        image = ds.images_dir / folder / "same.jpg"
        image.parent.mkdir(parents=True, exist_ok=True)
        make_test_image(image, (16, 16))
        images.append(str(image))
    result = SimpleNamespace(orig_shape=(16, 16), names={0: "object"},
                             keypoints=None, masks=None, obb=None, boxes=[])

    class FakeModel:
        def predict(self, **kwargs):
            return [result]

    fake_ultralytics = ModuleType("ultralytics")
    fake_ultralytics.YOLO = lambda path: FakeModel()
    with patch.dict(sys.modules, {"ultralytics": fake_ultralytics}):
        worker = PrelabelWorker("fake.pt", images, str(ds.labels_dir))
        worker.run()
    for folder in ("a", "b"):
        assert (ds.labels_dir / folder / "same.txt").exists(), f"嵌套预标注输出错误: {folder}"
    assert not (ds.labels_dir / "same.txt").exists(), "嵌套同名图片被扁平写入同一标签"
    print("  [OK] 预标注保留嵌套相对路径")


def test_pose_yaml():
    tmp = Path(tempfile.mkdtemp())
    ds = create_dataset(tmp / "pose", ["person"], LabelType.POSE)
    ds.meta.pose_kpts = 3
    ds.save_meta()
    data = yaml.safe_load(ds.write_data_yaml().read_text(encoding="utf-8"))
    assert "kpt_shape" in data, "POSE data.yaml 缺少 kpt_shape"
    assert data["kpt_shape"] == [3, 3], f"kpt_shape 错误: {data['kpt_shape']}"
    reopened = Dataset(ds.root)
    assert getattr(reopened.meta, "pose_kpts", None) == 3, "关键点数量未持久化"
    print("  [OK] POSE data.yaml 包含并持久化 kpt_shape")


def test_prelabel_extraction():
    base = dict(orig_shape=(100, 100), names={0: "object"}, keypoints=None,
                masks=None, boxes=[])
    points = np.array([[[1., 2.], [3., 4.], [5., 6.], [7., 8.]]], dtype=np.float32)
    obb = SimpleNamespace(
        data=np.array([[50., 50., 20., 10., 0., 0.9, 0.]], dtype=np.float32),
        xyxyxyxy=points,
        cls=np.array([0.], dtype=np.float32),
    )
    try:
        labels = PrelabelWorker._extract(SimpleNamespace(**base, obb=obb))
    except Exception as exc:
        raise AssertionError(f"OBB 预标注提取异常: {exc}") from exc
    assert len(labels) == 1 and labels[0].type == LabelType.OBB
    assert labels[0].class_id == 0
    assert np.allclose(labels[0].points, points[0]), labels[0].points

    box = SimpleNamespace(cls=np.array([0.]), xyxy=np.array([[1., 2., 30., 40.]]))
    kpts = SimpleNamespace(data=np.array(
        [[[10., 10., 0.9], [20., 20., 0.4], [30., 30., 0.0]]], dtype=np.float32))
    result = SimpleNamespace(**dict(base, boxes=[box], keypoints=kpts, obb=None))
    labels = PrelabelWorker._extract(result)
    visibility = [k[2] for k in labels[0].keypoints]
    assert visibility == [2, 2, 0], f"POSE 置信度未正确转为可见性: {visibility}"
    print("  [OK] POSE YAML 与 OBB/POSE 预标注提取")


if __name__ == "__main__":
    print("== 测试 label_io 四种格式 ==")
    test_label_io()
    print("== 测试 dataset 管理 ==")
    test_dataset()
    print("== 测试划分回归 ==")
    test_split_preserves_nested_paths()
    test_split_rejects_move()
    test_split_failure_preserves_existing()
    test_split_preserves_backup_when_rollback_fails()
    print("== 测试 POSE/OBB 回归 ==")
    test_pose_yaml()
    test_prelabel_extraction()
    test_prelabel_preserves_nested_paths()
    print("\n全部核心逻辑测试通过！")
