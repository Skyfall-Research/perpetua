"""Every torch-gated test carries the ``torch`` marker the torch CI jobs select by."""

import importlib.util
import re
import subprocess
import sys
from pathlib import Path

TESTS = Path(__file__).resolve().parent
# The module that imports torch, then every marked function, marked parameter
# and guide with a torch block.
BRIDGE = "tests/test_torch_bridge.py::"
MARKED = {
    "tests/test_scripts.py::test_baseline_rejects_minibatches_that_do_not_divide_the_batch",
    "tests/test_scripts.py::test_baseline_progress_is_valid_json_before_the_first_episode_ends",
    "tests/test_measurement.py::test_baseline_script_reports_throughput_when_torch_is_available",
    "tests/test_donation.py::test_every_runner_reports_whether_it_consumes_its_input[TorchRunner]",
    "tests/test_docs_api.py::test_documented_modules_import[perpetua.torch_bridge]",
    "tests/test_docs_api.py::test_every_module_declares_names_that_exist[perpetua.torch_bridge]",
    "tests/test_docs_api.py::test_every_declared_name_is_documented_or_listed_with_a_reason"
    "[perpetua.torch_bridge]",
    "tests/test_tutorials.py::test_guide_code_blocks_execute_in_order[docs/training.md]",
}


def test_torch_is_gated_by_the_marker_only() -> None:
    gate = re.compile(r"""importorskip\(\s*["']torch|find_spec\(["']torch["']\)""")
    allowed = {"conftest.py", "test_torch_bridge.py", Path(__file__).name}
    gated = [path.name for path in TESTS.glob("*.py") if gate.search(path.read_text())]
    assert set(gated) <= allowed, "gate these tests with pytest.mark.torch instead"


def test_the_marker_selects_every_torch_gated_test() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-m", "torch"],
        cwd=TESTS.parent,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    ids = [line for line in result.stdout.splitlines() if "::" in line]
    assert {line for line in ids if not line.startswith(BRIDGE)} == MARKED
    # Without torch the bridge's test module skips at import and collects nothing.
    bridge = importlib.util.find_spec("torch") is not None
    assert any(line.startswith(BRIDGE) for line in ids) == bridge
