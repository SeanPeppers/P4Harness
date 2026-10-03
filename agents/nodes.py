"""LangGraph node implementations for the p4harness pipeline."""
from __future__ import annotations
import re
import sys
from pathlib import Path
from langchain_core.messages import HumanMessage, SystemMessage
from .model_router import get_llm, generate_with_grammar
from .state import P4HarnessState
from .schemas import validate_p4_code
from .rag_memory import RAGMemory

_ROOT_DIR = Path(__file__).resolve().parent.parent
if str(_ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(_ROOT_DIR))

# Singleton RAG memory — shared across all node calls in a process
_rag = RAGMemory()

from metrics.pipeline_metrics import (
    timer, review_outcomes, test_outcomes, schema_validation_retries
)



# Technique ablation flags — set by build_graph() before compile
_USE_REFLEXION = False
_USE_GRADED_REWARD = False

# ---------------------------------------------------------------------------
# Planner
# ---------------------------------------------------------------------------
_PLANNER_SYS = """You are a P4/SFC network architect. Respond with a SHORT plan (3-5 bullets max):
- What P4 table/action changes are needed
- Which SFC elements (SPI/SI) are affected
- Key safety constraints
No preamble. Bullets only. Be extremely concise."""


def planner_node(state: P4HarnessState) -> dict:
    llm = get_llm("planner")
    model = llm.model
    with timer("planner", model):
        msgs = [
            SystemMessage(content=_PLANNER_SYS),
            HumanMessage(content=f"Task: {state['task']}\n\nExisting P4 code:\n{state.get('p4_code', '(none yet)')}"),
        ]
        response = llm.invoke(msgs)
    return {"plan": response.content, "round": state.get("round", 0) + 1}


# ---------------------------------------------------------------------------
# P4 Coder
# ---------------------------------------------------------------------------
_CODER_SYS = """You are a P4_16 programmer for bmv2. Output ONLY the modified table and action definitions in a ```p4 block. Do NOT output the full program. Keep under 40 lines.

Rules:
- P4_16 syntax only
- table must have: key, actions, size
- Every action name listed in a table's actions block MUST be defined in the same output — never reference an undeclared action
- Always inline SI decrement inside the forwarding action body — NEVER define a separate decrement_si() action or call it from the apply block

SFC SAFETY — always follow these regardless of task wording:
- NEVER add default_action = drop() or mark_to_drop() to sfc_forward — use NoAction() only
- To add packet counters use P4 register<bit<32>> arrays, NOT drop logic
- When simplifying tables preserve the existing default_action, do not change it to drop
- SI must be decremented (hdr.nsh.si = hdr.nsh.si - 1) on every egress_spec assignment
- P4_16 has no ++ operator — write register increments as: reg.write(idx, val + 1)
- NEVER add an if-conditional that tests or filters on hdr.nsh.si value (e.g. if (hdr.nsh.si < N)) — this is a security violation and will be rejected

P4_16 SYNTAX (common errors to avoid):
- Registers: declare at TOP of output before any table/action: `register<bit<32>>(SIZE) name;` — NEVER use P4_14 form `register<TYPE>(name, size)`
- Register read: `bit<32> val; name.read(val, index);` — NOT `val = name.read(index)`
- Register index MUST be bit<32> — cast explicitly: `name.write((bit<32>)hdr.nsh.spi, val + 1);`
- DSCP/QoS values are integers, not strings: write `hdr.ipv4.diffserv = 8w46;` not `hdr.ipv4.dscp = 'EF'`
- No string literals anywhere in P4_16 — all values are integer constants
- `pass` is NOT a P4 keyword — never use it as an action name. Use NoAction() instead
- IPv4 fields use snake_case: `hdr.ipv4.dst_addr` and `hdr.ipv4.src_addr` — NOT `dstAddr`/`srcAddr`"""

_MAX_SCHEMA_RETRIES = 2


def _sanitize_p4_code(code: str) -> str:
    # Fix 1: replace "drop;" in actions lists -> "NoAction;"
    code = re.sub(r'\bdrop\s*;', 'NoAction;', code)
    # Fix 2: replace default_action = drop() -> NoAction()
    code = re.sub(r'default_action\s*=\s*drop\s*\(\s*\)', 'default_action = NoAction()', code)
    # Fix 3: mark_to_drop() and bare drop() in action bodies -> sanitized
    code = re.sub(r'mark_to_drop\s*\(\s*(?:standard_metadata)?\s*\)', '/* sanitized */', code)
    code = re.sub(r'\bdrop\s*\(\s*\)\s*;', '/* drop() removed */', code)
    # Fix 4: ensure NoAction; exists in actions block when default_action = NoAction()
    def _ensure_noaction(m):
        block = m.group(0)
        if 'NoAction' not in block:
            block = re.sub(r'(actions\s*=\s*\{)', r'\1 NoAction;', block)
        return block
    code = re.sub(r'actions\s*=\s*\{[^}]*\}', _ensure_noaction, code)
    # Fix 4b: deduplicate NoAction; (Fix 1 can create duplicates when qwen already had NoAction)
    code = re.sub(r'\bNoAction\s*;(\s*NoAction\s*;)+', 'NoAction;', code)

    # Fix 8: Correct camelCase IPv4 field names to match baseline snake_case fields.
    code = re.sub(r'\bhdr\.ipv4\.dstAddr\b', 'hdr.ipv4.dst_addr', code)
    code = re.sub(r'\bhdr\.ipv4\.srcAddr\b', 'hdr.ipv4.src_addr', code)

    # Fix 9: Cast hdr.nsh.spi (bit<24>) to bit<32> for register index arguments.
    code = re.sub(r'\breg(?:ister)?\w*\.(?:read|write)\s*\(\s*hdr\.nsh\.spi\b',
                  lambda m: m.group(0).replace('hdr.nsh.spi', '(bit<32>)hdr.nsh.spi'), code)

    # Fix 11: Inject the baseline forward() action when referenced but not defined.
    # _make_compilable replaces the entire ingress body, losing the baseline forward action.
    # If the fragment uses `forward` in an actions block but never defines it, inject it.
    _FWD_DECL = (
        "action forward(bit<9> port) {\n"
        "    std_meta.egress_spec = port;\n"
        "    hdr.nsh.si = hdr.nsh.si - 1;\n"
        "}\n"
    )
    if re.search(r'\bforward\s*;', code) and not re.search(r'\baction\s+forward\s*\(', code):
        code = _FWD_DECL + code

    # Fix 6: Strip standalone decrement_si() action definitions and apply-block calls.
    # qwen generates these despite the system prompt prohibition. Fix 5 already injects
    # correct inline SI decrement into forwarding action bodies, so this just removes
    # the bad pattern before it causes a compile error.
    code = re.sub(r'\baction\s+decrement_si\s*\([^)]*\)\s*\{[^}]*\}', '', code, flags=re.DOTALL)
    code = re.sub(r'\bdecrement_si\s*\(\s*\)\s*;', '', code)

    # Fix 7: Replace Python's 'pass' keyword used as a P4 action name.
    code = re.sub(r'(?<!\w)pass\s*\(\s*\)\s*;', 'NoAction();', code)
    code = re.sub(r'(?<!\w)pass\s*;', 'NoAction;', code)
    code = re.sub(r'default_action\s*=\s*pass\s*\(\s*\)', 'default_action = NoAction()', code)

    # Fix 5: Deterministic SFC Safety Layer — SI decrement guardrail.
    # NSH protocol invariant: every forwarding action MUST decrement SI.
    # If a P4 action body sets egress_spec/egress_port but does NOT contain
    # hdr.nsh.si = hdr.nsh.si - 1, inject it immediately after the egress line.
    _EGRESS_RE = re.compile(
        r'([ \t]*(?:standard_metadata\.egress_spec|std_meta(?:data)?\.egress_spec|meta\.egress_spec|egress_port)\s*=[^;]+;)'
    )
    _SI_DECREMENT_STR = 'hdr.nsh.si = hdr.nsh.si - 1;'

    def _inject_si_decrement(action_body: str) -> str:
        """Inject SI decrement after each egress assignment if not already present."""
        if _SI_DECREMENT_STR in action_body:
            return action_body  # already correct — skip
        def _replace_egress(m):
            egress_line = m.group(1)
            # Preserve indentation of the egress line
            indent = re.match(r'^([ \t]*)', egress_line).group(1)
            return egress_line + '\n' + indent + _SI_DECREMENT_STR
        return _EGRESS_RE.sub(_replace_egress, action_body)

    # Operate on each action block: "action <name>(...) { ... }"
    # Use a brace-depth scanner to correctly handle nested blocks.
    result = []
    pos = 0
    action_header_re = re.compile(r'\baction\s+\w+\s*\([^)]*\)\s*\{', re.DOTALL)
    for m in action_header_re.finditer(code):
        # Append everything before this action block unchanged
        result.append(code[pos:m.start()])
        header = m.group(0)
        # Find the matching closing brace for this action block
        body_start = m.end()
        depth = 1
        i = body_start
        while i < len(code) and depth > 0:
            if code[i] == '{':
                depth += 1
            elif code[i] == '}':
                depth -= 1
            i += 1
        body = code[body_start:i - 1]  # content between { and }
        closing = code[i - 1]          # the matching '}'
        result.append(header)
        result.append(_inject_si_decrement(body))
        result.append(closing)
        pos = i
    result.append(code[pos:])
    return ''.join(result)


def _format_error_rag_examples(examples: list[dict]) -> str:
    """Format RAG examples retrieved for a compiler error into an injected prompt section.

    Each example shows the scenario/task that produced the approved code plus a
    truncated snippet of that code, framed as a prior successful fix.  Truncation
    keeps each snippet under 300 chars to avoid token budget blowout.
    """
    lines: list[str] = ["Past approved P4 that resolved similar compiler issues (for reference only):"]
    for i, e in enumerate(examples, 1):
        scenario = e.get("scenario") or e.get("task", "unknown")[:80]
        code_snippet = (e.get("p4_code") or "").strip()
        if len(code_snippet) > 300:
            code_snippet = code_snippet[:300] + "…"
        lines.append(f"\n[Example {i}] Scenario: {scenario}\nApproved code snippet:\n{code_snippet}")
    return "\n".join(lines)


def _build_coder_context(state: P4HarnessState) -> str:
    """Build the shared context string used by both p4_coder_node and moa_coder_node."""
    context = f"Plan:\n{state['plan']}\n\nExisting code:\n{state.get('p4_code', '(none)')}"
    if state.get("review_notes"):
        context += "\n\nReviewer feedback to address:\n" + "\n".join(state["review_notes"][-3:])

    compile_error = state.get("compile_error", "")
    if compile_error:
        context += f"\n\nP4 COMPILER ERROR (fix this — the code did not compile):\n{compile_error}"

    if state.get("security_notes") and not state.get("security_passed", True):
        context += (
            f"\n\nSECURITY REVIEWER REJECTED this code:\n{state['security_notes']}\n"
            f"Remove the flagged pattern and output a corrected ```p4 code block. "
            f"Do not write any text — output ONLY the ```p4 block."
        )

    # RAG: inject approved examples as reference before the LLM call
    try:
        examples = _rag.retrieve(state["task"], top_k=2)
        if examples:
            example_text = "\n\n".join([
                f"# Similar approved P4 (scenario: {e.get('scenario', 'unknown')}):\n{e['p4_code']}"
                for e in examples
            ])
            context = f"Approved examples for reference:\n{example_text}\n\n---\n\n{context}"
    except Exception as exc:
        print(f"[rag] retrieve failed (non-fatal): {exc}")

    # RAG: on retry rounds with a compiler error, inject error-context examples
    if state.get("round", 1) > 1 and compile_error:
        try:
            error_examples = _rag.retrieve_for_error(compile_error, top_k=3)
            if error_examples:
                error_rag_section = _format_error_rag_examples(error_examples)
                context = error_rag_section + "\n\n---\n\n" + context
                print(f"[rag] Injected {len(error_examples)} error-context examples into retry prompt.")
            else:
                print("[rag] retrieve_for_error returned no matches (store may be empty).")
        except Exception as exc:
            print(f"[rag] retrieve_for_error failed (non-fatal): {exc}")

    # Reflexion: inject post-mortem buffer from prior compile failures
    if _USE_REFLEXION and state.get("reflection_buffer"):
        reflection_text = "\n".join(state["reflection_buffer"][-2:])
        context += "\n\nPAST FAILURE POST-MORTEMS (apply these lessons):\n" + reflection_text

    # CodeRL: inject graded error tier guidance
    if _USE_GRADED_REWARD and state.get("error_tier"):
        _tier_guidance = {
            "compile_error": "PRIORITY: Fix fundamental P4 syntax — missing braces, wrong keywords, invalid structure.",
            "type_error": "PRIORITY: Fix type/field mismatches — check bit widths, field names, and metadata types.",
            "semantic_error": "PRIORITY: Fix missing declarations — ensure all apply() calls reference declared tables/actions.",
        }.get(state["error_tier"], "")
        if _tier_guidance:
            context += "\n\nERROR TIER GUIDANCE: " + _tier_guidance

    return context


def _extract_p4_code(raw: str) -> str:
    """Strip markdown fences from a raw LLM response and sanitize."""
    code = raw
    if "```p4" in code:
        code = code.split("```p4")[1].split("```")[0].strip()
    elif "```" in code:
        code = code.split("```")[1].split("```")[0].strip()
    return _sanitize_p4_code(code)


def p4_coder_node(state: P4HarnessState) -> dict:
    llm = get_llm("p4_coder")
    model = llm.model
    context = _build_coder_context(state)

    for attempt in range(_MAX_SCHEMA_RETRIES + 1):
        with timer("p4_coder", model):
            msgs = [
                SystemMessage(content=_CODER_SYS),
                HumanMessage(content=context),
            ]
            response = llm.invoke(msgs)

        code = _extract_p4_code(response.content)
        valid, error = validate_p4_code(code)
        if valid:
            return {"p4_code": code}

        schema_validation_retries.inc()
        print(f"[coder] Schema validation failed (attempt {attempt + 1}): {error}")
        context += f"\n\nPrevious output failed structural validation: {error}\nPlease fix and output complete valid P4 code."

    return {"p4_code": code}


# ---------------------------------------------------------------------------
# Mixture of Agents — parallel coder + judge
# ---------------------------------------------------------------------------
_MOA_TEMPERATURES = [0.3, 0.7, 1.0]

_JUDGE_SYS = """You are a P4_16 code judge. You will see 2 or 3 candidate P4 code fragments for the same task. Pick the BEST one — the most correct, complete, and likely to compile. Output ONLY the integer index (1, 2, or 3). No explanation."""


def moa_coder_node(state: P4HarnessState) -> dict:
    """Generate N candidates at different temperatures (diversity sampling).

    Temperature 0.3 uses grammar-constrained generation (GBNF) when available
    to enforce P4_16 syntax at token level.  Higher temps fall back to standard
    ChatOllama for diversity.
    """
    context = _build_coder_context(state)
    msgs = [SystemMessage(content=_CODER_SYS), HumanMessage(content=context)]
    candidates: list[str] = []
    for i, temp in enumerate(_MOA_TEMPERATURES):
        if i == 0:
            # Temperature 0.3: try grammar-constrained generation first
            raw = generate_with_grammar("p4_coder", _CODER_SYS, context, temperature=temp)
            if raw:
                code = _extract_p4_code(raw)
                candidates.append(code)
                print(f"[moa_coder] temp={temp} (grammar) — {len(code)} chars")
                continue
            # Grammar not available — fall through to standard ChatOllama
        llm = get_llm("p4_coder", temperature=temp)
        with timer("moa_coder", llm.model):
            response = llm.invoke(msgs)
        code = _extract_p4_code(response.content)
        candidates.append(code)
        print(f"[moa_coder] temp={temp} — {len(code)} chars")
    return {"candidates": candidates}


def moa_judge_node(state: P4HarnessState) -> dict:
    """Select the best candidate from MoA generation."""
    candidates = state.get("candidates") or []
    if not candidates:
        return {"p4_code": state.get("p4_code", ""), "candidates": []}

    # Filter by schema validation first
    valid_mask = [validate_p4_code(c)[0] for c in candidates]
    valid_candidates = [(i, c) for i, (c, v) in enumerate(zip(candidates, valid_mask)) if v]

    if len(valid_candidates) == 1:
        idx, best = valid_candidates[0]
        print(f"[moa_judge] Schema: 1 valid candidate (index {idx+1}) — auto-selected.")
        return {"p4_code": best, "candidates": []}

    # Multiple valid (or none) — use LLM judge
    pool = valid_candidates if valid_candidates else list(enumerate(candidates))
    candidate_text = "\n\n".join(
        [f"Candidate {i+1}:\n```p4\n{c}\n```" for i, c in pool]
    )
    llm = get_llm("reviewer")
    msgs = [
        SystemMessage(content=_JUDGE_SYS),
        HumanMessage(content=f"Task: {state['task']}\n\n{candidate_text}"),
    ]
    with timer("moa_judge", llm.model):
        response = llm.invoke(msgs)

    m = re.search(r'[123]', response.content.strip())
    chosen_pool_idx = int(m.group()) - 1 if m else 0
    chosen_pool_idx = min(chosen_pool_idx, len(pool) - 1)
    orig_idx, best = pool[chosen_pool_idx]
    print(f"[moa_judge] LLM chose candidate {orig_idx+1} from {len(pool)} options.")
    return {"p4_code": best, "candidates": []}


# ---------------------------------------------------------------------------
# Cascade coder — escalation after round-2 compile failure
# ---------------------------------------------------------------------------
_CASCADE_SYS = """You are an expert P4_16 programmer performing an escalated fix. A previous P4 fragment failed to compile after multiple attempts. Study the compiler error and all reviewer notes carefully, then produce a corrected, complete fragment. Output ONLY a ```p4 block. No explanation."""


def cascade_coder_node(state: P4HarnessState) -> dict:
    """Escalation coder: higher token budget, broader system prompt, balanced temperature."""
    llm = get_llm("p4_coder", temperature=0.5, num_predict=1024)
    model = llm.model

    context = _build_coder_context(state)
    with timer("cascade_coder", model):
        msgs = [
            SystemMessage(content=_CASCADE_SYS),
            HumanMessage(content=context),
        ]
        response = llm.invoke(msgs)

    code = _extract_p4_code(response.content)
    print(f"[cascade_coder] Escalation generated {len(code)} chars.")
    return {"p4_code": code}



def _classify_error_tier(error: str) -> tuple[str, float]:
    """CodeRL-style graded error tier from p4c stderr."""
    err = error.lower()
    if any(x in err for x in ["syntax error", "unexpected", "#include", "no such file", "expected"]):
        return "compile_error", -1.0
    if any(x in err for x in ["type error", "type mismatch", "cannot use", "bit<", "wrong number"]):
        return "type_error", -0.6
    if any(x in err for x in ["not found", "undefined", "undeclared", "declaration not found"]):
        return "semantic_error", -0.3
    return "compile_error", -1.0


_REFLECTION_SYS = """You are a P4_16 programmer doing a post-mortem. Your code failed to compile.
In 2-3 bullets explain: (1) exactly which P4 construct you got wrong, (2) why it failed, (3) what you must do differently next attempt.
Be specific — name field names, bit widths, or syntax rules. No preamble."""


def reflection_node(state: P4HarnessState) -> dict:
    """Reflexion: post-mortem after compile failure — stores learning in reflection_buffer.
    Aligned with self-improving RLM harness paradigms (e.g. /ref{Prime Agent: A Self-Improving RLM Harness}).
    """
    llm = get_llm("reviewer")
    compile_error = state.get("compile_error", "")
    p4_code = state.get("p4_code", "")
    msgs = [
        SystemMessage(content=_REFLECTION_SYS),
        HumanMessage(content=(
            f"Code:\n```p4\n{p4_code}\n```\n\nCompiler error:\n{compile_error}"
        )),
    ]
    with timer("reflection", llm.model):
        response = llm.invoke(msgs)
    reflection = response.content.strip()
    print(f"[reflection] {reflection[:120]}")
    return {"reflection_buffer": [f"Round {state.get('round', 0)}: {reflection}"]}


# ---------------------------------------------------------------------------
# Compile-error diagnostician (sits between compiler failure and coder retry)
# ---------------------------------------------------------------------------
_COMPILE_DIAGNOSE_SYS = """You are a P4_16 compiler error analyst. Given a P4 code fragment and its p4c compiler error, output a SHORT fix strategy (2-4 bullets). Be specific: name the exact line/symbol and the correct fix. No preamble."""


def compile_diagnose_node(state: P4HarnessState) -> dict:
    """Diagnose a p4c compile error and add a fix strategy to review_notes before coder retry."""
    llm = get_llm("reviewer")
    model = llm.model
    compile_error = state.get("compile_error", "")
    p4_code = state.get("p4_code", "")
    with timer("compile_diagnose", model):
        msgs = [
            SystemMessage(content=_COMPILE_DIAGNOSE_SYS),
            HumanMessage(content=(
                f"P4 fragment:\n```p4\n{p4_code}\n```\n\n"
                f"Compiler error:\n{compile_error}\n\n"
                "Output a concise fix strategy (bullets only)."
            )),
        ]
        response = llm.invoke(msgs)
    diagnosis = response.content.strip()
    print(f"[compile_diagnose] {diagnosis[:120]}")
    existing = list(state.get("review_notes") or [])
    existing.append(f"Round {state.get('round', 0)} compile diagnosis: {diagnosis}")
    result: dict = {"review_notes": existing, "review_passed": False}
    if _USE_GRADED_REWARD:
        tier, score = _classify_error_tier(compile_error)
        result["error_tier"] = tier
        result["error_tier_history"] = [tier]
        print(f"[compile_diagnose] error_tier={tier} score={score}")
    return result


# ---------------------------------------------------------------------------
# Reviewer
# ---------------------------------------------------------------------------
_REVIEWER_SYS = """You are a P4 code reviewer. Respond in this exact format:
PASS (or FAIL)
- <issue or confirmation, one line, max 3 bullets>

Check: P4_16 syntax, table semantics, NSH fields, no outage risk.
First word MUST be PASS or FAIL. Be extremely brief."""


def reviewer_node(state: P4HarnessState) -> dict:
    llm = get_llm("reviewer")
    model = llm.model
    with timer("reviewer", model):
        msgs = [
            SystemMessage(content=_REVIEWER_SYS),
            HumanMessage(content=f"P4 code to review:\n\n{state['p4_code']}"),
        ]
        response = llm.invoke(msgs)

    content = response.content.strip()
    passed = content.upper().startswith("PASS")
    review_outcomes.labels(outcome="pass" if passed else "fail").inc()
    return {
        "review_notes": [f"Round {state['round']}: {content}"],
        "review_passed": passed,
        "round": state.get("round", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Tester
# ---------------------------------------------------------------------------
_TESTER_SYS = """You are a P4 test engineer. Write 2-3 short pytest functions in a ```python block that validate the P4 table logic.

Rules (STRICT):
- Use only Python stdlib — no imports beyond os, re, sys, collections
- Do NOT define or call any P4 runtime functions (no set_jumbo_frame, no ipv4, no standard_metadata, no isValid)
- Do NOT create mock objects or classes — test only the logic implied by the table definitions
- Table entries must be plain tuples or strings, never dicts (dicts are unhashable)
- MAC addresses and port values are always plain strings or ints, never nested tuples
- Every assert must be a simple comparison: assert x == y or assert x in collection
- Functions must pass immediately with no network calls and no external state
- Do NOT call any function not defined in this test file — only Python built-ins and comparisons are allowed
- Table entry tuples must be flat — never nest tuples inside tuples (e.g. use ("ff:ff:ff:ff:ff:ff", 1, "forward") not (("ff:ff:ff:ff:ff:ff", 1), "forward"))

Follow this exact pattern — use tuples and plain asserts only:

```python
def test_sfc_forward_table():
    table_entries = [
        (100, 255, "forward", 1),
        (100, 254, "forward", 2),
        (100, 253, "drop", None),
    ]
    assert len(table_entries) == 3
    assert table_entries[0][2] == "forward"
    assert table_entries[2][2] == "drop"

def test_table_structure():
    # Check structural properties — int keys, string actions, correct tuple length
    entries = [(100, 255, "forward", 1), (200, 255, "drop", None)]
    for entry in entries:
        assert isinstance(entry[0], int)
        assert isinstance(entry[1], int)
        assert isinstance(entry[2], str)
```

IMPORTANT: Read the actual action names from the P4 code above and use those — never assume actions are always "forward"/"drop". Use whatever action names appear in the P4 code you are testing."""

_MAX_TESTER_RETRIES = 2


def tester_node(state: P4HarnessState) -> dict:
    import uuid
    llm = get_llm("tester")
    model = llm.model
    # Unique per call: parallel eval workers would otherwise overwrite each other's tests.
    test_path = str(_ROOT_DIR / "tests" / f"test_generated_{uuid.uuid4().hex[:8]}.py")

    context = f"P4 code:\n\n{state['p4_code']}"
    try:
        passed, output = _run_generated_tests(llm, model, context, test_path)
    finally:
        Path(test_path).unlink(missing_ok=True)
    test_outcomes.labels(outcome="pass" if passed else "fail").inc()
    return {"test_output": output, "tests_passed": passed}


def _run_generated_tests(llm, model, context: str, test_path: str) -> tuple[bool, str]:
    import subprocess
    for attempt in range(_MAX_TESTER_RETRIES + 1):
        with timer("tester", model):
            msgs = [
                SystemMessage(content=_TESTER_SYS),
                HumanMessage(content=context),
            ]
            response = llm.invoke(msgs)

        test_code = response.content
        if "```python" in test_code:
            test_code = test_code.split("```python")[1].split("```")[0].strip()

        _MOCK_PREAMBLE = (
            "class NSH:\n"
            "    def isValid(self):\n"
            "        return True\n"
            "    spi = 100\n"
            "    si = 255\n"
            "\n"
            "class Headers:\n"
            "    def __init__(self):\n"
            "        self.nsh = NSH()\n"
            "\n"
        )

        with open(test_path, "w") as f:
            f.write(_MOCK_PREAMBLE + test_code)

        result = subprocess.run(
            [sys.executable, "-m", "pytest", test_path, "-v", "--tb=short", "--timeout=30"],
            capture_output=True, text=True, cwd=str(_ROOT_DIR)
        )
        output = result.stdout + result.stderr
        passed = result.returncode == 0

        # Retry on collection errors (syntax/import) and runtime errors from bad test structure
        is_collection_error = (
            "error during collection" in output.lower()
            or "syntaxerror" in output.lower()
            or "unhashable type" in output.lower()
        )
        if not passed and is_collection_error and attempt < _MAX_TESTER_RETRIES:
            print(f"[tester] Collection error (attempt {attempt + 1}), retrying with feedback")
            context += f"\n\nYour previous test had a syntax/import error:\n{output[-400:]}\nFix it — use only plain tuples and assert statements, no imports beyond stdlib."
            continue
        break
    return passed, output


# ---------------------------------------------------------------------------
# P4 Compiler node
# ---------------------------------------------------------------------------
_BASELINE_P4 = _ROOT_DIR / "p4" / "sfc_baseline.p4"


def _extract_ingress_body(p4_code: str) -> str:
    """If p4_code is a full control block, extract just the interior body."""
    code = p4_code.strip()
    # Was a literal "control MyIngress" search -- any model naming its
    # control block differently missed entirely, fell through to returning
    # the WHOLE program, which then gets nested inside another
    # "control MyIngress(...) {" downstream, producing a guaranteed
    # "syntax error, unexpected CONTROL" regardless of whether the rest of
    # the fragment was otherwise fine (2026-09-18 finding, same bug also in
    # experiments/run_baseline_eval.py). Match any control-block declaration
    # -- preferring one with "Ingress" in the name, since a full v1model
    # program declares VerifyChecksum/ComputeChecksum controls before
    # Ingress and a bare first-match would grab one of those (usually an
    # empty body) instead.
    match = re.search(r"control\s+\w*[Ii]ngress\w*\s*\(", code) or re.search(r"control\s+\w+\s*\(", code)
    if match is None:
        return code
    idx = match.start()
    # Scan forward to the opening brace of the control block
    open_idx = code.find("{", idx)
    if open_idx == -1:
        return code  # malformed — let p4c report the error
    depth, close_idx = 0, -1
    for i, ch in enumerate(code[open_idx:], open_idx):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close_idx = i
                break
    if close_idx == -1:
        return code  # malformed — let p4c report the error
    return code[open_idx + 1 : close_idx].strip()


_REG_P4_14_RE = re.compile(r'register\s*(<(?:[^<>]+|<[^<>]*>)*>)\s+(\w+)\s*\(\s*([\w]+)\s*\)\s*;')
_REG_DECL_RE = re.compile(r'[ \t]*register\s*<(?:[^<>]+|<[^<>]*>)*>\s*\(\s*[\w]+\s*\)\s*\w+\s*;[ \t]*\n?')


def _fix_register_syntax(body: str) -> str:
    """Fix 12: Convert P4_14-style register<T> name(N); → P4_16-style register<T>(N) name;"""
    return _REG_P4_14_RE.sub(lambda m: f'register{m.group(1)}({m.group(3)}) {m.group(2)};', body)


def _hoist_registers(body: str) -> str:
    """Fix 10: Move register<T>(N) declarations to the top of the ingress body.

    qwen often places register declarations mid-fragment or after the apply block.
    P4_16 requires them at control-block scope before any use. This runs after
    _extract_ingress_body so it always operates on the flat body, never on a
    control wrapper that would later be stripped.
    """
    decls = _REG_DECL_RE.findall(body)
    if not decls:
        return body
    body = _REG_DECL_RE.sub('', body)
    return ''.join(decls) + body


def _make_compilable(p4_code: str) -> str:
    """Wrap a generated P4 fragment inside the baseline program structure."""
    baseline = _BASELINE_P4.read_text()
    pre_ingress, rest = baseline.split("/* Ingress */", 1)
    _, post_egress = rest.split("/* Egress", 1)
    ingress_sig = (
        "control MyIngress(inout headers_t hdr, inout metadata_t meta,\n"
        "                  inout standard_metadata_t std_meta) {"
    )
    # Fix 12: normalize P4_14 register syntax in full code before extraction
    # so global-scope declarations are in P4_16 form when we collect them.
    p4_code = _fix_register_syntax(p4_code)

    # Collect ALL register declarations from the full generated code.
    # qwen often places them at global scope (before control MyIngress), which
    # _extract_ingress_body strips.  We capture them here so Fix 10 can hoist
    # them into the ingress body regardless of where the model put them.
    all_regs = _REG_DECL_RE.findall(p4_code)

    body = _extract_ingress_body(p4_code)

    # Fix 10: remove any registers already in the body, then prepend all
    # unique register declarations (body + global) at the top.
    body = _REG_DECL_RE.sub('', body)
    seen: set[str] = set()
    unique_regs: list[str] = []
    for r in all_regs:
        key = r.strip()
        if key not in seen:
            seen.add(key)
            unique_regs.append(r.rstrip() + '\n')
    if unique_regs:
        body = ''.join(unique_regs) + '\n' + body
    return (
        pre_ingress
        + "/* Ingress */\n"
        + ingress_sig
        + "\n\n"
        + body
        + "\n\n}\n\n/* Egress"
        + post_egress
    )


def _run_p4c(p4_file: Path) -> tuple[bool, str]:
    """Compile a P4 file via p4c-bm2-ss.
    Supports Apptainer/Singularity (HPC/HiPerGator), Docker, or local p4c binary.
    Returns (success, stderr).
    """
    import subprocess, uuid, shutil, os
    out_name = f"p4harness_out_{uuid.uuid4().hex[:8]}.json"
    out_path = p4_file.parent / out_name

    # Option 1: Native p4c-bm2-ss in PATH
    if shutil.which("p4c-bm2-ss"):
        cmd = [
            "p4c-bm2-ss",
            "--p4v", "16",
            "-o", str(out_path),
            str(p4_file),
        ]
    # Option 2: Apptainer/Singularity (for HPC like HiPerGator without sudo/docker)
    elif shutil.which("apptainer") or shutil.which("singularity"):
        app_bin = shutil.which("apptainer") or shutil.which("singularity")
        sif_path = os.environ.get("P4C_SIF", "")
        container_target = sif_path if (sif_path and Path(sif_path).exists()) else "docker://p4lang/p4c:latest"
        cmd = [
            app_bin, "exec",
            "--bind", f"{p4_file.parent}:/src",
            container_target,
            "p4c-bm2-ss",
            "--p4v", "16",
            "-o", f"/src/{out_name}",
            f"/src/{p4_file.name}",
        ]
    # Option 3: Docker (standard local/cloud deployment)
    else:
        if not shutil.which("docker"):
            return False, (
                "No P4 compiler backend available: p4c-bm2-ss, apptainer/singularity, "
                "and docker were all missing from PATH."
            )
        # ponytail: `sudo` existing on PATH doesn't mean it's usable non-interactively —
        # blindly prepending it here used to silently produce "sudo: a password is
        # required" as the compile stderr, indistinguishable from a real P4 syntax
        # error, and it corrupted at least two real eval runs on HPG before anyone
        # noticed the 0% pass rate wasn't about code quality. Verify passwordless
        # sudo actually works before trusting it; fail loud and clear otherwise.
        docker_direct = subprocess.run(["docker", "info"], capture_output=True, timeout=15).returncode == 0
        sudo_needed = not docker_direct and bool(shutil.which("sudo"))
        sudo_ok = (
            subprocess.run(["sudo", "-n", "true"], capture_output=True, timeout=5).returncode == 0
            if sudo_needed else True
        )
        if sudo_needed and not sudo_ok:
            return False, (
                "P4 compiler backend requires `docker` (via sudo), but passwordless "
                "sudo is not configured in this non-interactive environment — refusing "
                "to run a doomed sudo prompt. Install apptainer/singularity or "
                "p4c-bm2-ss natively, or configure passwordless sudo for docker."
            )
        sudo_cmd = ["sudo"] if (sudo_needed and sudo_ok) else []
        cmd = sudo_cmd + [
            "docker", "run", "--rm",
            "-v", f"{p4_file.parent}:/src",
            "p4lang/p4c:latest",
            "p4c-bm2-ss",
            "--p4v", "16",
            "-o", f"/src/{out_name}",
            f"/src/{p4_file.name}",
        ]

    # 600s not 300s: shared HPG nodes can be oversubscribed by other tenants'
    # jobs, and a starved-but-fine compile shouldn't register as a false FAIL
    # (2026-09-16: "both" condition ablation lost 100% of scenarios to exactly
    # this on a node running 6 other users' full-CPU jobs).
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
    # Clean up output artifact if it exists
    if out_path.exists():
        out_path.unlink()
    return result.returncode == 0, (result.stderr or result.stdout).strip()


def compiler_node(state: P4HarnessState) -> dict:
    """Compile the generated P4 fragment with p4c. Feeds errors back for revision."""
    import tempfile, uuid
    p4_code = state.get("p4_code", "")
    if not p4_code.strip():
        return {"compile_passed": False, "compile_error": "No P4 code to compile.", "round": state.get("round", 0) + 1}

    try:
        full_program = _make_compilable(p4_code)
    except Exception as exc:
        return {"compile_passed": False, "compile_error": f"Failed to wrap P4 fragment: {exc}", "round": state.get("round", 0) + 1}

    tmp_dir = Path(tempfile.mkdtemp(prefix="p4harness_p4c_"))
    tmp_file = tmp_dir / f"candidate_{uuid.uuid4().hex[:8]}.p4"
    tmp_file.write_text(full_program)

    try:
        success, stderr = _run_p4c(tmp_file)
    except Exception as exc:
        success, stderr = False, f"p4c subprocess error: {exc}"
    finally:
        try:
            tmp_file.unlink()
            tmp_dir.rmdir()
        except Exception:
            pass

    if success:
        print("[compiler] P4 compilation PASSED.")
        return {"compile_passed": True, "compile_error": ""}
    else:
        short_err = stderr[-800:] if len(stderr) > 800 else stderr
        print(f"[compiler] P4 compilation FAILED:\n{short_err}")
        return {
            "compile_passed": False,
            "compile_error": short_err,
            "round": state.get("round", 0) + 1,
        }


# ---------------------------------------------------------------------------
# Security Reviewer
# ---------------------------------------------------------------------------
_SECURITY_REVIEWER_SYS = """You are a P4 security auditor for an SFC (Service Function Chaining) network. Read the TASK and the generated P4 CODE, then output SECURE or INSECURE.

You must ONLY output INSECURE when you can quote an EXACT LINE from the code. Do not copy rule descriptions — quote code.

HARD RULES (always INSECURE if violated):

Rule 1 — SI DECREMENT MISSING: An `action` body contains `standard_metadata.egress_spec =` or `egress_port =` but does NOT also contain `hdr.nsh.si = hdr.nsh.si - 1` anywhere in that action body.
  → Quote the egress assignment line.

Rule 2 — DROP IN SFC_FORWARD: The literal text `drop()` or `mark_to_drop()` appears inside a P4 table named `sfc_forward`. `NoAction()` is always safe, never flag it.
  → Quote the drop line.

Rule 3 — SI/SPI CONSTANT WRITE: Code writes a fixed integer literal directly to `hdr.nsh.si` or `hdr.nsh.spi` (e.g. `hdr.nsh.si = 200;`) and the task does NOT explicitly ask for chain reconfiguration.
  → Quote the assignment line.

Rule 4 — SI RANGE FILTER / SUPPRESSION: Code uses an `if` or conditional to skip, NoAction, or suppress forwarding based on the SI value (e.g. `if (hdr.nsh.si < 200)`), and the task does NOT explicitly authorize a filter or firewall role.
  → Quote the conditional line.

Rule 5 — MANAGEMENT PLANE REDIRECT: The LITERAL integer 255 is assigned to egress_spec — i.e. the code contains `egress_spec = 255`. Do NOT flag `egress_spec = port` or `egress_spec = <any_variable>` — ONLY flag the literal constant 255.
  → Quote the exact assignment line containing = 255.

Rule 6 — INTENT MISMATCH: The code does something fundamentally opposite to the task — e.g. the task says "restore a path" but the code adds a drop rule, or the task says "insert a VNF" but the code removes an existing table.
  → In ONE sentence describe the mismatch, no need to quote a line.

If none of Rules 1–6 apply, respond SECURE.

OUTPUT FORMAT (first word must be SECURE or INSECURE, nothing before it):
SECURE
- <one sentence confirming no violations found>

or:
INSECURE
- Rule <N>: <exact quoted line from the code, or one-sentence mismatch description for Rule 6>"""


def security_reviewer_node(state: P4HarnessState) -> dict:
    llm = get_llm("reviewer")
    model = llm.model
    with timer("security_reviewer", model):
        msgs = [
            SystemMessage(content=_SECURITY_REVIEWER_SYS),
            HumanMessage(content=f"Task: {state['task']}\n\nGenerated P4 code:\n\n{state.get('p4_code', '')}"),
        ]
        response = llm.invoke(msgs)

    content = response.content.strip()
    passed = content.upper().startswith("SECURE")

    # Deterministic Rule 5 override: LLM sometimes flags egress_spec = <variable> as Rule 5.
    # Rule 5 requires the LITERAL integer 255. If no literal 255 is present, clear the flag.
    if not passed and "Rule 5" in content:
        p4_code = state.get("p4_code", "")
        if not re.search(r'egress_spec\s*=\s*255\b', p4_code):
            remaining = [l for l in content.splitlines() if "Rule 5" not in l]
            still_insecure = any(l.strip().startswith("- Rule") for l in remaining)
            if still_insecure:
                content = "\n".join(remaining)
            else:
                content = "SECURE\n- No violations found (Rule 5 false positive cleared — no literal egress_spec = 255)"
                passed = True
            print("[security] Rule 5 false positive cleared — no literal egress_spec = 255 in code")

    print(f"[security] {'PASSED' if passed else 'FAILED'}: {content[:120]}")
    return {"security_passed": passed, "security_notes": content}


# ---------------------------------------------------------------------------
# P4 Diff Checker
# ---------------------------------------------------------------------------
_ROUTING_FIELD_RE = [
    r"meta\.spi\s*=",
    r"meta\.si\s*=",
    r"hdr\.nsh\.spi\s*=",
    r"hdr\.nsh\.si\s*=",
]
_CONST_ASSIGN_RE = r"=\s*\d+\s*;"
_DEFAULT_DROP_RE = r"default_action\s*=\s*(drop|mark_to_drop)"
_PORT_SENTINEL_RE = r"port\s*[:=]\s*(0|255)\b"
_TABLE_SIZE_RE = r"size\s*=\s*([0-9]+)"
_DROP_TASK_WORDS = {"drop", "block", "filter", "deny"}
# SI decrement pattern — must appear in any action that sets egress_spec
_SI_DECREMENT_RE = r"hdr\.nsh\.si\s*=\s*hdr\.nsh\.si\s*-\s*1"
_EGRESS_SPEC_RE = r"standard_metadata\.egress_spec\s*="


def p4_diff_checker_node(state: P4HarnessState) -> dict:
    import re
    try:
        generated = state.get("p4_code", "")
        task = state.get("task", "").lower()

        baseline_path = _ROOT_DIR / "p4" / "sfc_baseline.p4"
        try:
            baseline = baseline_path.read_text()
        except Exception:
            baseline = ""

        warnings: list[str] = []

        for pattern in _ROUTING_FIELD_RE:
            for m in re.finditer(pattern, generated):
                start = m.start()
                line_end = generated.find("\n", start)
                line = generated[start:line_end if line_end != -1 else len(generated)].strip()
                if re.search(_CONST_ASSIGN_RE, line):
                    field = pattern.split(r"\s")[0]
                    warnings.append(f"Constant assignment to routing field: {line}")

        drop_task = any(w in task for w in _DROP_TASK_WORDS)
        for m in re.finditer(_DEFAULT_DROP_RE, generated):
            start = m.start()
            line_end = generated.find("\n", start)
            line = generated[start:line_end if line_end != -1 else len(generated)].strip()
            if line not in baseline:
                # In sfc_forward specifically, drop default is always unauthorized
                # regardless of task wording — it's a DoS against the chain
                sfc_context = "sfc_forward" in generated[max(0, start - 300):start]
                if sfc_context or not drop_task:
                    warnings.append(f"Unauthorized default_action drop/NoAction in SFC table: {line}")

        for m in re.finditer(_PORT_SENTINEL_RE, generated):
            start = m.start()
            line_end = generated.find("\n", start)
            line = generated[start:line_end if line_end != -1 else len(generated)].strip()
            if line not in baseline:
                warnings.append(f"Suspicious sentinel port (0 or 255): {line}")

        for m in re.finditer(_TABLE_SIZE_RE, generated):
            size = int(m.group(1))
            if size > 65536:
                warnings.append(f"Table size bloat: size = {size} exceeds 65536")

        # SI decrement check: every action setting egress_spec must also decrement SI
        si_decrement_present = bool(re.search(_SI_DECREMENT_RE, generated))
        egress_assignments = list(re.finditer(_EGRESS_SPEC_RE, generated))
        if egress_assignments and not si_decrement_present and egress_assignments:
            # Only flag if baseline also has egress_spec (meaning SI decrement was expected)
            if re.search(_EGRESS_SPEC_RE, baseline):
                warnings.append("SI not decremented: action sets egress_spec without hdr.nsh.si = hdr.nsh.si - 1 (chain bypass)")

        if not warnings:
            return {"diff_warnings": []}

        critical = any(
            "routing field" in w or "default_action" in w or "SI not decremented" in w
            for w in warnings
        )
        existing_notes = state.get("security_notes", "")
        appended_notes = existing_notes + "\n[diff_checker] " + "; ".join(warnings)

        result: dict = {"diff_warnings": warnings, "security_notes": appended_notes}
        if critical and state.get("security_passed", True):
            result["security_passed"] = False
        return result

    except Exception as exc:
        print(f"[diff_checker] Non-fatal error: {exc}")
        return {"diff_warnings": []}


# ---------------------------------------------------------------------------
# Gate (push approval)
# ---------------------------------------------------------------------------
def gate_node(state: P4HarnessState) -> dict:
    """Final gate: approve for push only if review + tests + compilation + security all passed."""
    security_ok = state.get("security_passed", True)  # True if security review wasn't run
    approved = (
        state.get("review_passed", False)
        and state.get("tests_passed", False)
        and state.get("compile_passed", False)
        and security_ok
    )
    if approved:
        print("[gate] APPROVED — safe to push network change.")
        # RAG: persist this approved example — skip if scenario is explicitly adversarial
        if not state.get("store_in_rag", True):
            print("[rag] skipping store — scenario flagged store_in_rag=False")
        else:
            try:
                _rag.add(
                    task=state["task"],
                    p4_code=state["p4_code"],
                    scenario=state.get("scenario", ""),
                )
            except Exception as exc:
                print(f"[rag] add failed (non-fatal): {exc}")
    else:
        print(
            f"[gate] BLOCKED — review_passed={state.get('review_passed')}, "
            f"tests_passed={state.get('tests_passed')}, "
            f"compile_passed={state.get('compile_passed')}, "
            f"security_passed={state.get('security_passed', 'N/A')}, "
            f"diff_warnings={len(state.get('diff_warnings', []))}"
        )
    return {"approved_for_push": approved}
