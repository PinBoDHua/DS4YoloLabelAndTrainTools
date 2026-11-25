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
from app.core.label_io import (read_yolo_txt, write_yolo_txt, detect_format,
                               remap_class_ids, count_class_usage)
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


def test_detect_format_prefer():
    """数据集声明的任务类型必须参与全部格式判断。"""
    # DETECT 数据集：非 4 值行应判为不符（None），不能兜底成 SEGMENT
    assert detect_format([0, 0.5, 0.5, 0.2, 0.3, 0.1], LabelType.DETECT) is None
    assert detect_format([0, 0.5, 0.5, 0.2, 0.3, 0.1, 0.4, 0.2], LabelType.DETECT) is None
    assert detect_format([0, 0.5, 0.5, 0.2, 0.3, 0.1], None) is None  # 奇数坐标也不应兜底
    # SEGMENT 数据集：8 值行按声明的 SEGMENT 解析（OBB/分割歧义消歧）
    vals8 = [0] + [0.1] * 8
    assert detect_format(vals8, LabelType.SEGMENT) == LabelType.SEGMENT
    assert detect_format(vals8, LabelType.OBB) == LabelType.OBB
    assert detect_format(vals8, None) == LabelType.OBB
    # POSE 数据集：可见性不符合 0/1/2 的行判为不符
    pose_bad = [0, 0.5, 0.5, 0.2, 0.3, 0.1, 0.2, 1.5]
    assert detect_format(pose_bad, LabelType.POSE) is None
    pose_ok = [0, 0.5, 0.5, 0.2, 0.3, 0.1, 0.2, 2.0]
    assert detect_format(pose_ok, LabelType.POSE) == LabelType.POSE
    print("  [OK] prefer_type 参与全部格式判断")


def test_read_bad_lines_preserved():
    """坏行只跳过不终止整个文件；且写回时原样保留，不被自动保存覆盖。"""
    tmp = Path(tempfile.mkdtemp())
    img = tmp / "bad.png"
    make_test_image(img)

    txt = tmp / "bad.txt"
    # 第 1 行正常、第 2 行无法解析、第 3 行坐标数不足、第 4 行正常
    txt.write_text(
        "0 0.5 0.5 0.2 0.2\n"
        "garbage line here\n"
        "0 0.5 0.5\n"
        "1 0.3 0.3 0.1 0.1\n",
        encoding="utf-8")
    ann = read_yolo_txt(txt, img, prefer_type=LabelType.DETECT)
    assert len(ann.labels) == 2, f"好行应全部读出，实际 {len(ann.labels)}"
    assert ann.labels[0].class_id == 0 and ann.labels[1].class_id == 1
    assert len(ann.raw_lines) == 2, f"坏行应保留在 raw_lines，实际 {ann.raw_lines}"
    # 旧实现：第 3 行 IndexError 会终止整个文件读取（只剩第 1 行）

    # 自动保存（写回）不应丢弃坏行
    write_yolo_txt(ann, txt)
    text = txt.read_text(encoding="utf-8")
    assert "garbage line here" in text, "写回后坏行丢失"
    assert "0 0.5 0.5" in text, "写回后坐标数不足的行丢失"
    # 再读一遍应稳定（幂等）
    ann2 = read_yolo_txt(txt, img, prefer_type=LabelType.DETECT)
    assert len(ann2.labels) == 2 and len(ann2.raw_lines) == 2

    # 任务类型不符的行也应保留
    txt2 = tmp / "mismatch.txt"
    txt2.write_text("0 0.1 0.1 0.2 0.2 0.3 0.3 0.4 0.4 0.5 0.5\n", encoding="utf-8")
    ann3 = read_yolo_txt(txt2, img, prefer_type=LabelType.DETECT)
    assert not ann3.labels and len(ann3.raw_lines) == 1, "DETECT 数据集中的多边形行应被保留"
    # 撤销快照 JSON 往返保留 raw_lines
    ann4 = ImageAnnotation.from_dict(ann.to_dict())
    assert ann4.raw_lines == ann.raw_lines
    print("  [OK] 坏行逐行容错且写回保留")


def test_coco_to_yolo():
    """COCO 转换：category_id 映射为连续索引；分割+关键点标注保留关键点。"""
    import json as json_mod
    from app.core.format_convert import coco_to_yolo
    tmp = Path(tempfile.mkdtemp())
    coco = {
        "categories": [
            {"id": 12, "name": "cat"},
            {"id": 90, "name": "dog"},
            {"id": 3, "name": "person"},
        ],
        "images": [{"id": 1, "width": 100, "height": 100, "file_name": "a.jpg"}],
        "annotations": [
            # 非连续 id：cat=12 应映射为 0
            {"image_id": 1, "category_id": 12, "bbox": [10, 10, 20, 20]},
            # dog=90 → 1：同时含分割与关键点，应输出 pose 行（保留关键点）
            {"image_id": 1, "category_id": 90, "bbox": [30, 30, 40, 40],
             "segmentation": [[40.0, 40.0, 60.0, 40.0, 60.0, 60.0, 40.0, 60.0]],
             "keypoints": [50.0, 50.0, 2, 55.0, 55.0, 1, 60.0, 60.0, 0]},
            # person=3 → 2：纯分割
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 10, 10],
             "segmentation": [[1.0, 1.0, 5.0, 1.0, 5.0, 5.0, 1.0, 5.0]]},
            # 未知类别应被跳过
            {"image_id": 1, "category_id": 999, "bbox": [0, 0, 5, 5]},
        ],
    }
    json_path = tmp / "coco.json"
    json_path.write_text(json_mod.dumps(coco), encoding="utf-8")
    out_dir = tmp / "yolo_labels"
    ok, fail, classes = coco_to_yolo(json_path, out_dir)
    assert ok == 1 and fail == 0
    assert classes == ["cat", "dog", "person"]
    lines = (out_dir / "a.txt").read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 3, f"未知类别应被跳过，实际 {len(lines)} 行"
    # cat: 索引 0，检测行
    assert lines[0].startswith("0 ")
    # dog: 索引 1，pose 行 = 1 + box(4) + 3*3 关键点 = 14 个 token
    dog_tokens = lines[1].split()
    assert dog_tokens[0] == "1" and len(dog_tokens) == 14, \
        f"分割+关键点标注应输出关键点，实际 {len(dog_tokens)} token"
    assert dog_tokens[7] == "2" and dog_tokens[10] == "1" and dog_tokens[13] == "0", \
        "关键点可见性错误"
    # person: 索引 2，分割行
    person_tokens = lines[2].split()
    assert person_tokens[0] == "2" and len(person_tokens) == 9
    print("  [OK] COCO 类别映射 + 关键点保留")


def test_remap_class_ids():
    """类别删除/重排必须重映射所有标注文件的 class_id。"""
    tmp = Path(tempfile.mkdtemp())
    labels = tmp / "labels"
    (labels / "sub").mkdir(parents=True)
    (labels / "a.txt").write_text(
        "0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n2 0.5 0.5 0.2 0.2\n",
        encoding="utf-8")
    (labels / "sub" / "b.txt").write_text(
        "0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\ngarbage\n", encoding="utf-8")

    usage = count_class_usage(labels)
    assert usage == {0: 2, 1: 2, 2: 1}, f"类别统计错误: {usage}"

    # 删除类别 1：其行删除，id>1 的前移
    rewritten, removed = remap_class_ids(labels, {1: None, 2: 1})
    assert removed == 2 and rewritten == 2
    a_lines = (labels / "a.txt").read_text(encoding="utf-8").strip().splitlines()
    assert a_lines[0].startswith("0 ") and a_lines[1].startswith("1 ")
    assert len(a_lines) == 2, "被删除类别的行应移除"
    b_text = (labels / "sub" / "b.txt").read_text(encoding="utf-8")
    assert b_text.strip().splitlines()[0].startswith("0 "), "未受影响文件的行不应变化"
    assert "garbage" in b_text, "坏行必须原样保留"

    # 交换 0/1（上移/下移）
    (labels / "a.txt").write_text(
        "0 0.1 0.1 0.2 0.2\n1 0.3 0.3 0.2 0.2\n", encoding="utf-8")
    remap_class_ids(labels, {0: 1, 1: 0})
    a_lines = (labels / "a.txt").read_text(encoding="utf-8").strip().splitlines()
    assert a_lines[0].startswith("1 ") and a_lines[1].startswith("0 "), "交换重映射错误"

    # 删除最后一个类别（数据集仅剩 0 号类）
    (labels / "a.txt").write_text("0 0.1 0.1 0.2 0.2\n", encoding="utf-8")
    remap_class_ids(labels, {0: None})
    assert (labels / "a.txt").read_text(encoding="utf-8") == "", "删除唯一类别后应清空"
    print("  [OK] 类别删除/重排重映射")


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
    print("== 测试格式判断/坏行/COCO/类别重映射 ==")
    test_detect_format_prefer()
    test_read_bad_lines_preserved()
    test_coco_to_yolo()
    test_remap_class_ids()
    print("\n全部核心逻辑测试通过！")
