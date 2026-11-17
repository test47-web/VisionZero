import importlib.util
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "prepare_rdd2022.py"
spec = importlib.util.spec_from_file_location("prepare_rdd2022", SCRIPT)
module = importlib.util.module_from_spec(spec)
assert spec and spec.loader
spec.loader.exec_module(module)


def test_class_mapping():
    assert module.normalise_class("D00") == 0
    assert module.normalise_class("D10") == 1
    assert module.normalise_class("D20") == 2
    assert module.normalise_class("D40") == 3
    assert module.normalise_class("D50") is None


def test_group_assignment_is_deterministic():
    groups = [f"India:{i}" for i in range(20)]
    first = module.assign_groups(groups, 42)
    second = module.assign_groups(groups, 42)
    assert first == second
    assert set(first.values()).issubset({"train", "val", "test"})
