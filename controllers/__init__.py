import sys
from pathlib import Path

# 添加项目根目录到Python路径
PROJECT_ROOT = str(Path(__file__).resolve().parent.parent)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from .base_controller import BaseController
from .ik_controller import IKController

__all__ = ["BaseController", "IKController"]
