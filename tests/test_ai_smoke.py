"""AI 辅助链路冒烟测试：SAM 加载 + 点提示分割 + 模型预标注。"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw

from app.core.sam_engine import SamEngine
from PySide6.QtCore import QCoreApplication, QEventLoop, QTimer

app = QCoreApplication(sys.argv)

# 合成测试图：白底 + 两个不同颜色的圆
img = Image.new("RGB", (640, 480), "white")
d = ImageDraw.Draw(img)
d.ellipse([100, 100, 300, 300], fill=(255, 0, 0))     # 红圆
d.ellipse([400, 200, 560, 420], fill=(0, 0, 255))     # 蓝圆
img.save("_test_sam_input.png")
bgr = np.asarray(img)[:, :, ::-1].copy()  # RGB -> BGR

engine = SamEngine(None)
loaded = {}


def on_loaded(ok, err):
    loaded["ok"] = ok
    loaded["err"] = err
    if not ok:
        print(f"[FAIL] SAM 加载失败: {err}")
        app.exit(1)
        return
    print("[OK] SAM 模型加载成功")
    # 用红圆中心点做正点提示
    engine.segment_done.connect(on_segment)
    engine.segment(bgr, points=[[200, 200]], point_labels=[1], box=None, request_id=1)


def on_segment(req_id, polygon):
    if polygon is None:
        print("[FAIL] SAM 分割返回空结果")
        app.exit(1)
        return
    print(f"[OK] SAM 分割成功: {len(polygon)} 个顶点")
    # 验证多边形位于红圆区域附近
    cx = sum(p[0] for p in polygon) / len(polygon)
    cy = sum(p[1] for p in polygon) / len(polygon)
    print(f"  多边形质心: ({cx:.0f}, {cy:.0f})（红圆中心约 200,200）")
    assert 100 < cx < 300 and 100 < cy < 300, "多边形不在红圆区域"
    print("[OK] 多边形位置正确（位于红圆内）")
    # 测试预标注（完成后退出）
    test_prelabel()


def test_prelabel():
    global _prelabel_worker  # 保持引用防止被 GC 回收
    from app.core.prelabel_engine import PrelabelWorker
    # 用 yolov8n.pt 对合成图推理（COCO 类别，红圆蓝圆可能检出为 sports ball）
    tmp = Path(tempfile.mkdtemp())
    _prelabel_worker = PrelabelWorker("yolov8n.pt", [str(Path("_test_sam_input.png").resolve())],
                                      str(tmp), conf=0.1, skip_existing=False)
    worker = _prelabel_worker
    result = {}

    def on_finish(ok, skip):
        result["ok"] = ok
        out = tmp / "_test_sam_input.txt"
        if out.exists():
            lines = out.read_text(encoding="utf-8").strip().splitlines()
            print(f"[OK] 预标注完成: {ok} 张, 生成 {len(lines)} 个检测结果")
            for line in lines[:5]:
                print(f"  {line}")
        else:
            print(f"[OK] 预标注完成: {ok} 张（未检出目标，输出为空文件）")
        assert ok == 1
        print("\nAI 辅助链路测试全部通过！")
        app.quit()

    def on_fail(err):
        print(f"[FAIL] 预标注失败: {err}")
        app.exit(1)

    worker.finished_ok.connect(on_finish)
    worker.failed.connect(on_fail)
    worker.start()


engine.ensure_loaded(on_loaded)
QTimer.singleShot(300000, lambda: (print("[FAIL] 超时"), app.exit(1)))
sys.exit(app.exec())
