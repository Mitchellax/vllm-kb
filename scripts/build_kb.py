"""入口脚本：python scripts/build_kb.py [--config ...] [--skip-pull] [--limit N] [--rebuild]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vllm_kb.pipeline import main  # noqa: E402

if __name__ == "__main__":
    main()
