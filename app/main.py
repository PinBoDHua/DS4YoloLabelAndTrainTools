"""YOLO 标注训练工具 - 程序入口。"""
from __future__ import annotations

import logging
import sys

from app.utils import install_excepthook, patch_opencv_imread, setup_logging

setup_logging()
install_excepthook()
patch_opencv_imread()  # cv2.imread 中文路径补丁（训练/预标注/SAM 依赖）
logger = logging.getLogger("app.main")


def main() -> int:
    import faulthandler
    faulthandler.enable()  # Python 层崩溃时输出栈信息

    from PySide6.QtCore import Qt, qInstallMessageHandler
    from PySide6.QtCore import QtMsgType as QMT
    from PySide6.QtWidgets import QApplication

    # 捕获 Qt 内部消息（qWarning/qFatal 等）写入日志，便于定位 C++ 层崩溃原因
    qt_logger = logging.getLogger("qt")

    def _qt_message_handler(mode, context, message):
        if mode in (QMT.QtDebugMsg,):
            qt_logger.debug("[%s] %s", context.file or "qt", message)
        elif mode in (QMT.QtWarningMsg, QMT.QtCriticalMsg):
            qt_logger.warning("[%s] %s", context.file or "qt", message)
        elif mode == QMT.QtFatalMsg:
            qt_logger.error("[FATAL] [%s] %s", context.file or "qt", message)
        else:
            qt_logger.info("[%s] %s", context.file or "qt", message)

    qInstallMessageHandler(_qt_message_handler)

    # Windows 高分屏适配
    try:
        from PySide6.QtGui import QGuiApplication
        QGuiApplication.setHighDpiScaleFactorRoundingPolicy(
            Qt.HighDpiScaleFactorRoundingPolicy.PassThrough)
    except Exception:
        pass

    app = QApplication(sys.argv)
    app.setApplicationName("YOLO 标注训练工具")
    app.setOrganizationName("YoloTools")

    from app.config import config
    from app.theme import theme
    theme.apply(app)  # 恢复上次主题（默认浅色）

    from app.ui.main_window import MainWindow
    win = MainWindow()

    # 恢复窗口几何
    geo = config.get("window")
    if geo:
        try:
            win.setGeometry(geo["x"], geo["y"], geo["w"], geo["h"])
        except Exception:
            pass
    win.show()

    # 后台探测 GPU（供训练/推理面板显示设备信息）
    def _on_gpu_probe(ok: bool, desc: str) -> None:
        win.train_panel.set_gpu_info(ok, desc)
        win.test_panel.set_gpu_info(ok, desc)

    from PySide6.QtCore import QTimer
    from app.core.device_probe import ensure_probe
    QTimer.singleShot(300, lambda: ensure_probe(_on_gpu_probe))

    # 自动打开上次数据集
    if config.get("auto_open_last", True):
        last = config.get("last_dataset")
        if last:
            from pathlib import Path
            if Path(last).is_dir():
                QTimer.singleShot(200, lambda: win._open_dataset(last))

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
