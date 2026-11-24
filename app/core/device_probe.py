"""设备探测：后台线程检查 GPU/CUDA 可用性与型号，供训练/推理面板显示。

探测在线程中执行（import torch 较慢，避免阻塞界面启动），
结果缓存于模块级变量，多个面板共享一次探测。
"""
from __future__ import annotations

import logging

from PySide6.QtCore import QThread, Signal

logger = logging.getLogger(__name__)

_result: tuple[bool, str] | None = None     # (cuda 可用, 描述文本)
_callbacks: list = []
_running = False
_thread: QThread | None = None              # 保持引用防止 GC


def probe_gpu_sync() -> tuple[bool, str]:
    """同步探测（须在线程中调用）。"""
    try:
        import torch
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            mem = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
            return True, f"{name}（{mem:.1f} GB）"
        return False, "未检测到可用 GPU"
    except Exception as e:
        logger.debug("GPU 探测失败: %s", e)
        return False, "GPU 探测失败"


class _GpuProbeThread(QThread):
    done = Signal(bool, str)

    def run(self) -> None:
        global _result
        _result = probe_gpu_sync()
        self.done.emit(*_result)


def _on_done(ok: bool, desc: str) -> None:
    global _running
    _running = False
    for cb in list(_callbacks):
        try:
            cb(ok, desc)
        except Exception:
            pass
    _callbacks.clear()


def shutdown_probe() -> None:
    global _running
    if _thread is not None and _thread.isRunning():
        _thread.wait()
    _running = False


def ensure_probe(callback) -> None:
    """确保 GPU 探测已启动；完成后在 GUI 线程回调 callback(ok, desc)。"""
    global _running, _thread
    if _result is not None:
        callback(*_result)
        return
    _callbacks.append(callback)
    if not _running:
        _running = True
        _thread = _GpuProbeThread()
        _thread.done.connect(_on_done)
        _thread.start()
