"""Unit tests for the p4harness harness (no LLM calls — mock the nodes)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


class NSH:
    def isValid(self):
        return True

    spi = 100
    si = 255


class Headers:
    def __init__(self):
        self.nsh = NSH()


def test_state_schema_imports():
    from agents.state import P4HarnessState
    state: P4HarnessState = {
        "task": "add firewall VNF",
        "p4_code": "",
        "plan": "",
        "review_notes": [],
        "test_output": "",
        "review_passed": False,
        "tests_passed": False,
        "approved_for_push": False,
        "round": 0,
    }
    assert state["task"] == "add firewall VNF"


def test_model_router_loads_config():
    from agents.model_router import _load_config
    cfg = _load_config()
    assert "models" in cfg
    assert "planner" in cfg["models"]
    assert "p4_coder" in cfg["models"]
    assert "reviewer" in cfg["models"]


def test_graph_compiles():
    from agents.graph import build_graph
    g = build_graph()
    assert g is not None


def test_swap_model_updates_config(tmp_path):
    """Test model swap without touching real config."""
    import yaml

    cfg = {
        "models": {"planner": {"model": "gemma3:27b", "temperature": 0.3}},
        "ollama": {"base_url": "http://localhost:11434"},
        "fallbacks": {},
    }
    config_path = tmp_path / "models.yaml"
    with open(config_path, "w") as f:
        yaml.dump(cfg, f)

    # Patch the config path
    import agents.model_router as router
    original = router._CONFIG_PATH
    router._CONFIG_PATH = config_path
    router._config = None

    router.swap_model("planner", "llama3.1:8b")

    with open(config_path) as f:
        updated = yaml.safe_load(f)
    assert updated["models"]["planner"]["model"] == "llama3.1:8b"

    # Restore
    router._CONFIG_PATH = original
    router._config = None


def test_review_pass_detection():
    """Gate logic: review_passed AND tests_passed → approved."""
    from agents.nodes import gate_node
    state = {
        "review_passed": True,
        "tests_passed": True,
        "compile_passed": True,
        "round": 1,
    }
    result = gate_node(state)
    assert result["approved_for_push"] is True


def test_review_fail_blocks_push():
    from agents.nodes import gate_node
    state = {
        "review_passed": False,
        "tests_passed": True,
        "round": 1,
    }
    result = gate_node(state)
    assert result["approved_for_push"] is False
