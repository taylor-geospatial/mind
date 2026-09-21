import importlib
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class ImportTests(unittest.TestCase):
    def test_imports(self) -> None:
        modules = ["mind", "mind_standalone", "hubconf"]
        for base, folder in ((ROOT / "src", "mind"), (ROOT, "scripts")):
            modules.extend(
                ".".join(path.relative_to(base).with_suffix("").parts)
                for path in sorted((base / folder).rglob("*.py"))
                if path.name != "__init__.py"
            )
        for module in modules:
            with self.subTest(module=module):
                importlib.import_module(module)
