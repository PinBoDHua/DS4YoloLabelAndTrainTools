"""中文路径训练验证：ultralytics 在中文数据集路径下能否正常训练。"""
import sys
import random
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from app.core.dataset_manager import create_dataset
from app.core.label_data import ImageAnnotation, LabelItem, LabelType
from app.core.label_io import write_yolo_txt
from app.core.train_engine import TrainWorker
from PySide6.QtCore import QCoreApplication, QTimer

app = QCoreApplication(sys.argv)

# 中文路径数据集
tmp = Path(tempfile.mkdtemp()) / "中文训练数据集"
ds = create_dataset(tmp, ["猫", "狗"], LabelType.DETECT)
random.seed(7)
for i in range(12):
    Image.new("RGB", (320, 320), (100 + i * 8, 90, 140)).save(ds.root / "images" / f"图{i}.jpg")
for i in range(12):
    img = str(ds.root / "images" / f"图{i}.jpg")
    labels = []
    for j in range(random.randint(1, 3)):
        x, y = random.randint(10, 200), random.randint(10, 200)
        w, h = random.randint(30, 100), random.randint(30, 100)
        labels.append(LabelItem(LabelType.DETECT, random.randint(0, 1),
                                points=[[x, y], [x + w, y + h]]))
    write_yolo_txt(ImageAnnotation(img, labels), ds.label_path(img))
ds.scan_images()

counts = ds.split((0.8, 0.1, 0.1), 42, copy=True)
print(f"划分: {counts}")
yaml_path = ds.write_data_yaml()

worker = TrainWorker(
    model="yolov8n.pt", data_yaml=str(yaml_path), epochs=1, batch=4,
    imgsz=320, lr0=0.01, device="0",
    project=str(ds.root), name="中文训练测试",
)


def on_epoch(epoch, total, metrics):
    print(f"epoch {epoch}/{total} 正常")


def on_finish(out_dir, final):
    best = Path(out_dir) / "weights" / "best.pt"
    if best.exists():
        print(f"[OK] 中文路径训练成功！best.pt 已生成")
    else:
        print("[FAIL] best.pt 未生成")
    app.quit()


def on_fail(err):
    print(f"[FAIL] 中文路径训练失败: {err[:500]}")
    app.exit(1)


worker.epoch_done.connect(on_epoch)
worker.finished_ok.connect(on_finish)
worker.failed.connect(on_fail)
worker.start()

QTimer.singleShot(600000, lambda: (print("[FAIL] 超时"), app.exit(1)))
sys.exit(app.exec())
