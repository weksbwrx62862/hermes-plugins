"""pytest 配置 — 设置 Python 路径。"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
