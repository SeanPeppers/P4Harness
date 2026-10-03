"""Model router — loads model config and returns a LangChain-Ollama LLM."""
from __future__ import annotations
import json
import urllib.request
import urllib.error
import yaml
from pathlib import Path
from langchain_ollama import ChatOllama

_GRAMMAR_PATH = Path(__file__).parent.parent / "configs" / "p4_fragment.gbnf"
_grammar_cache: str | None = None


def _load_grammar() -> str:
    global _grammar_cache
    if _grammar_cache is None and _GRAMMAR_PATH.exists():
        _grammar_cache = _GRAMMAR_PATH.read_text()
    return _grammar_cache or ""


_P4_JSON_SCHEMA = {
    "type": "object",
    "properties": {
        "p4_code": {
            "type": "string",
            "description": "Complete P4_16 fragment: register declarations, actions, tables, apply block.",
        }
    },
    "required": ["p4_code"],
    "additionalProperties": False,
}


def generate_with_grammar(role: str, system: str, prompt: str, temperature: float | None = None) -> str:
    """Generate P4 code using Ollama's JSON schema format constraint.

    Ollama 0.24+ supports JSON schema via the `format` field in /api/generate.
    This prevents prose contamination — the entire response is a JSON object
    with a `p4_code` key.  Falls back gracefully (returns "") on any failure.

    Note: GBNF grammar (configs/p4_fragment.gbnf) is the formal spec for the
    syntax layer; it requires llama.cpp direct API or a future Ollama version
    that re-exposes the `grammar` field.  JSON schema is the current practical
    approximation.
    """
    cfg = _load_config()
    model_cfg = cfg["models"][role]
    base_url = cfg["ollama"]["base_url"].rstrip("/")
    model = model_cfg["model"]
    temp = temperature if temperature is not None else model_cfg["temperature"]

    structured_prompt = (
        f"{system}\n\n"
        f"{prompt}\n\n"
        "Output ONLY valid JSON with a single key 'p4_code' whose value is the P4_16 fragment. "
        "No markdown fences inside the JSON value."
    )
    payload = {
        "model": model,
        "prompt": structured_prompt,
        "stream": False,
        "format": _P4_JSON_SCHEMA,
        "options": {"temperature": temp, "num_predict": model_cfg.get("num_predict", 512)},
    }

    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{base_url}/api/generate",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            result = json.loads(resp.read())
            raw = result.get("response", "")
            parsed = json.loads(raw)
            code = parsed.get("p4_code", "")
            if code:
                print(f"[router] JSON-schema generation succeeded ({len(code)} chars).")
            return code
    except Exception as exc:
        print(f"[router] JSON-schema generate failed ({exc}), caller will use ChatOllama fallback.")
        return ""

_CONFIG_PATH = Path(__file__).parent.parent / "configs" / "models.yaml"
_config: dict | None = None


def _load_config() -> dict:
    global _config
    if _config is None:
        with open(_CONFIG_PATH) as f:
            _config = yaml.safe_load(f)
    return _config


def get_llm(role: str, temperature: float | None = None, num_predict: int | None = None) -> ChatOllama:
    """Return a ChatOllama instance for the given agent role.

    temperature and num_predict override the config values when provided.
    """
    cfg = _load_config()
    model_cfg = cfg["models"][role]
    base_url = cfg["ollama"]["base_url"]
    return ChatOllama(
        model=model_cfg["model"],
        temperature=temperature if temperature is not None else model_cfg["temperature"],
        base_url=base_url,
        num_predict=num_predict if num_predict is not None else model_cfg.get("num_predict", 512),
    )


def swap_model(role: str, new_model: str) -> None:
    """Hot-swap a model for a role in the running config (no restart needed)."""
    cfg = _load_config()
    cfg["models"][role]["model"] = new_model
    with open(_CONFIG_PATH, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False)
    # Invalidate cache so next get_llm picks up the change
    global _config
    _config = None
    print(f"[router] {role} → {new_model}")
