"""通用辅助：日志初始化、文件工具、下载加速等。"""
from __future__ import annotations

import logging
import sys
import traceback
from logging.handlers import RotatingFileHandler
from pathlib import Path

from app.config import LOGS_DIR

# GitHub 下载加速镜像（依次尝试）
GH_MIRRORS = [
    "https://ghfast.top/",
    "https://gh-proxy.com/",
    "https://ghproxy.net/",
]


def patch_opencv_imread() -> None:
    """全局补丁：cv2.imread 支持中文路径（回退 PIL 读取）。

    背景：OpenCV 的 imread 不支持含中文的路径，且当前环境 numpy 2.x 的
    np.fromfile 对中文路径同样失败，唯一可靠方案是 PIL 读取后转 numpy。
    ultralytics 训练/推理内部均调用 cv2.imread，打补丁后中文路径数据集
    可正常训练。
    """
    try:
        import cv2
    except ImportError:
        return

    if getattr(cv2, "__imread_patched__", False):
        return

    _orig_imread = cv2.imread

    def _patched_imread(path, flags=cv2.IMREAD_COLOR):
        img = _orig_imread(path, flags)
        if img is not None:
            return img
        # 原生读取失败（多为中文路径）：PIL 回退
        try:
            from PIL import Image
            import numpy as np
            pil = Image.open(path)
            if flags == cv2.IMREAD_GRAYSCALE:
                return np.asarray(pil.convert("L"))
            if flags == cv2.IMREAD_UNCHANGED:
                return np.asarray(pil)
            return np.asarray(pil.convert("RGB"))[:, :, ::-1].copy()  # RGB→BGR
        except Exception:
            return None

    cv2.imread = _patched_imread
    cv2.__imread_patched__ = True


def download_with_fallback(url: str, target: Path, progress_cb=None,
                           timeout: int = 60) -> bool:
    """带镜像回退的文件下载：先直连，失败后依次尝试 GitHub 加速镜像。

    支持断点续传（.part 临时文件）。返回是否成功。
    """
    import requests

    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        return True
    tmp = target.with_suffix(target.suffix + ".part")

    urls = [url]
    if url.startswith("https://github.com/"):
        urls += [m + url for m in GH_MIRRORS]

    last_err = None
    for try_url in urls:
        try:
            headers = {}
            if tmp.exists():
                headers["Range"] = f"bytes={tmp.stat().st_size}-"
            with requests.get(try_url, stream=True, timeout=timeout,
                              headers=headers) as r:
                if r.status_code in (200, 206):
                    total = int(r.headers.get("content-length", 0)) + tmp.stat().st_size \
                        if r.status_code == 206 else int(r.headers.get("content-length", 0))
                    mode = "ab" if r.status_code == 206 else "wb"
                    done = tmp.stat().st_size if r.status_code == 206 else 0
                    with open(tmp, mode) as f:
                        for chunk in r.iter_content(chunk_size=1024 * 256):
                            f.write(chunk)
                            done += len(chunk)
                            if progress_cb and total:
                                progress_cb(done, total)
                elif r.status_code == 416:  # Range 超出（已完整）
                    pass
                else:
                    r.raise_for_status()
            if tmp.exists() and tmp.stat().st_size > 0:
                tmp.replace(target)
                return True
        except Exception as e:
            last_err = e
            logger = logging.getLogger(__name__)
            logger.warning("下载失败 %s: %s", try_url[:80], e)
            continue
    tmp.unlink(missing_ok=True)
    if last_err:
        logging.getLogger(__name__).warning("全部下载源失败: %s", last_err)
    return False


def setup_logging() -> None:
    """初始化全局日志：控制台 + 滚动文件（logs/app.log）。"""
    LOGS_DIR.mkdir(exist_ok=True)
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(name)s: %(message)s", "%H:%M:%S"
    )
    root = logging.getLogger()
    root.setLevel(logging.INFO)

    # 避免重复添加
    if not root.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(fmt)
        root.addHandler(sh)

        fh = RotatingFileHandler(
            LOGS_DIR / "app.log", maxBytes=5 * 1024 * 1024,
            backupCount=3, encoding="utf-8",
        )
        fh.setFormatter(fmt)
        root.addHandler(fh)


def install_excepthook() -> None:
    """全局异常钩子：未捕获异常弹窗提示并写日志，避免程序静默崩溃。"""
    def _hook(exc_type, exc_value, exc_tb):
        tb_text = "".join(traceback.format_exception(exc_type, exc_value, exc_tb))
        logging.getLogger(__name__).error("未捕获异常:\n%s", tb_text)
        try:
            from PySide6.QtWidgets import QApplication, QMessageBox
            app = QApplication.instance()
            if app is not None:
                QMessageBox.critical(
                    None, "程序错误",
                    f"发生未预期的错误，详细信息已写入 logs/app.log：\n\n{exc_value}",
                )
        except Exception:
            pass
        sys.__excepthook__(exc_type, exc_value, exc_tb)

    sys.excepthook = _hook
