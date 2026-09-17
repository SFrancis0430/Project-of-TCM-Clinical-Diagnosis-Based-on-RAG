import sys
from pathlib import Path

# 确保项目根目录在 sys.path 中，使 `import config` / `from src...` 在 pytest 下可用
sys.path.insert(0, str(Path(__file__).resolve().parent))
