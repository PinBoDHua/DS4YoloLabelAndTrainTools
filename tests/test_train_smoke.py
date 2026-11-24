"""训练链路冒烟测试：创建数据集 → 划分 → 训练 1 epoch → 验证权重生成。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.core.dataset_manager import create_dataset
from app.core.label_data import LabelItem, LabelType, ImageAnnotation
from app.core.label_io import write_yolo_txt
from app.core.train_engine import TrainWorker
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

app = QCoreApplication(sys.argv)

# 构造数据集：2 类，12 张图
tmp = Path(tempfile.mkdtemp())
ds = create_dataset(tmp / "train_smoke", ["cat", "dog"], LabelType.DETECT)
import random
random.seed(7)
for i in range(12):
    p = ds.root / "images" / f"img{i}.jpg"
    Image.new("RGB", (320, 320), (100 + i * 8, 90, 140)).save(p)
# 每张图 1-3 个随机框
for i in range(12):
    img = str(ds.root / "images" / f"img{i}.jpg")
    labels = []
    for j in range(random.randint(1, 3)):
        x, y = random.randint(10, 200), random.randint(10, 200)
        w, h = random.randint(30, 100), random.randint(30, 100)
        labels.append(LabelItem(LabelType.DETECT, random.randint(0, 1),
                                points=[[x, y], [x + w, y + h]]))
    write_yolo_txt(ImageAnnotation(img, labels), ds.label_path(img))
ds.scan_images()

# 划分 8:1:1
counts = ds.split((0.8, 0.1, 0.1), 42, copy=True)
print(f"划分: {counts}")
yaml_path = ds.write_data_yaml()

# 训练 1 epoch
worker = TrainWorker(
    model="yolov8n.pt", data_yaml=str(yaml_path), epochs=1, batch=4,
    imgsz=320, lr0=0.01, device="0",
    project=str(ds.root), name="smoke_train",
)

out = {}


def on_epoch(epoch, total, metrics):
    print(f"epoch {epoch}/{total}: {list(metrics.keys())[:6]}")


def on_finish(out_dir, final):
    out["dir"] = out_dir
    print(f"训练完成: {out_dir}")
    print(f"最终指标: {final}")
    best = Path(out_dir) / "weights" / "best.pt"
    assert best.exists(), f"best.pt 未生成: {best}"
    print(f"[OK] best.pt 已生成: {best} ({best.stat().st_size / 1024 / 1024:.1f} MB)")
    app.quit()


def on_fail(err):
    print(f"[FAIL] 训练失败: {err}")
    app.exit(1)


worker.epoch_done.connect(on_epoch)
worker.finished_ok.connect(on_finish)
worker.failed.connect(on_fail)
worker.start()

QTimer.singleShot(600000, lambda: (print("[FAIL] 训练超时"), app.exit(1)))
sys.exit(app.exec())
