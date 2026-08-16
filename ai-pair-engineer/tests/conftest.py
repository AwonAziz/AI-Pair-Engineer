"""Conftest to make ai_pair_engineer importable for tests."""
import sys
from pathlib import Path

# Add the project root (parent of tests/) to Python path
project_root = Path(__file__).parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))