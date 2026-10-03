# P4Harness

[![CI](https://github.com/SeanPeppers/P4Harness/actions/workflows/ci.yml/badge.svg)](https://github.com/SeanPeppers/P4Harness/actions/workflows/ci.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

**Compiler-gated multi-agent LLM pipeline for P4 network policy synthesis.**

This repository is the research artifact for the paper *P4Harness: Autonomous P4 Network Policy Management and Verification Using Compiler-Gated Multi-Agent Pipelines* (manuscript in preparation). It contains the LangGraph orchestration pipeline, the evaluation scenario suites, the benchmark datasets, and the container definitions needed to rerun the paper's experiments.

P4Harness turns a natural-language network intent (for example, *"port 1 is down, reroute SPI 100 to port 2"*) into a P4_16 program for the bmv2 `simple_switch` target. LLM agents plan, write, and review the change. A deterministic layer then decides whether the change ships: the code must compile with `p4c-bm2-ss`, pass a security review, and pass a structural diff check against the baseline program. LLM approval alone never deploys a change.

```
intent ─► planner ─► coder (MoA: 3 candidates + judge) ─► reviewer ─► tester
                        ▲                                               │
                        │                                               ▼
              compile_diagnose ◄──── FAIL ──── p4c-bm2-ss compiler gate
              (stderr + error-indexed RAG)                              │ PASS
                                                                        ▼
                                  gate ◄── p4_diff_checker ◄── security_reviewer
                                   │
                                   ▼
                          approved P4 program (max 4 revision rounds)
```

## Contents

- [Repository layout](#repository-layout)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick check (no GPU)](#quick-check-no-gpu)
- [Reproducing the paper](#reproducing-the-paper)
- [Datasets](#datasets)
- [Determinism and expected variance](#determinism-and-expected-variance)
- [Citation](#citation)
- [License](#license)

## Repository layout

```
agents/          LangGraph pipeline: graph, agent nodes, model router, RAG memory
  graph.py           full P4Harness pipeline (build_graph flags drive the ablations)
  graph_baselines.py zero-shot, NetBuddy-style, and Confucius-style baselines
  nodes.py           planner, MoA coder/judge, reviewer, tester, compiler gate,
                     compile_diagnose, security reviewer, diff checker, gate
configs/         models.yaml (agent-to-model routing), GBNF grammar, topology template
containers/      Apptainer definitions for p4c-bm2-ss and Ollama
data/            eval_set.jsonl (50 frozen tasks), verified_corpus.jsonl (3,745 examples)
experiments/     one module per experiment suite, plus ablation and evaluation drivers
metrics/         Prometheus counters emitted by the pipeline
p4/              sfc_baseline.p4, the NSH service-function-chaining program every run edits
scripts/         fine-tuning driver, Slurm example, result analysis, RAG seeding
tests/           unit tests (no LLM needed) and an environment preflight check
run.py           CLI: run the pipeline on one intent, or swap an agent's model
```

## Requirements

| Component | Paper setup | Minimum |
|---|---|---|
| OS | Linux (RHEL-based HPC cluster) | Linux |
| Python | 3.12 | 3.12 |
| GPU | 1× NVIDIA L4, 24 GB | any GPU that runs a 14B Q4 model in Ollama (~10 GB VRAM); CPU works, slowly |
| LLM runtime | [Ollama](https://ollama.com) with `qwen2.5-coder:14b` for every agent role | same |
| P4 compiler | `p4c-bm2-ss` from the `p4lang/p4c` image, run through Apptainer | any one of: native `p4c-bm2-ss`, Apptainer/Singularity, or Docker |
| Fine-tuning only | 1× L4 24 GB, [llama.cpp](https://github.com/ggml-org/llama.cpp) | 24 GB GPU |

The pipeline looks for a compiler in this order: `p4c-bm2-ss` on `PATH`, then Apptainer/Singularity (using `$P4C_SIF` if set, otherwise `docker://p4lang/p4c:latest`), then Docker.

## Installation

```bash
git clone https://github.com/SeanPeppers/P4Harness.git
cd P4Harness
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# LLM backend
ollama serve &                       # or use containers/ollama.def on a cluster
ollama pull qwen2.5-coder:14b

# P4 compiler (pick one)
docker pull p4lang/p4c:latest                                   # Docker
apptainer build p4c.sif containers/p4c.def && export P4C_SIF=$PWD/p4c.sif   # HPC

python tests/preflight_check.py      # checks Python deps, compiler, pytest-timeout, Ollama
```

> **Security note:** the tester agent writes LLM-generated `pytest` files and runs them on the host. Run experiments in a container, VM, or disposable cluster job, not on a machine that holds credentials.

## Quick check (no GPU)

These run in under a minute and need neither Ollama nor a compiler:

```bash
python -m pytest tests/test_harness.py -v         # pipeline routing and gate logic, LLM mocked
python experiments/runner.py run all --dry-run    # prints every scenario in every suite
```

With Ollama and a compiler available, run the pipeline on a single intent:

```bash
python run.py run "Port 1 is DOWN. Reroute SPI=100 SI=255 to port 2." --p4 p4/sfc_baseline.p4 --dry-run
```

## Reproducing the paper

Every experiment writes a timestamped JSON log to `experiments/results/`. Run `python experiments/runner.py report` for an approval summary of the per-scenario pipeline logs, or `python scripts/analyze_results.py --mode audit` to recompile the approved programs from the exp1–3 and ablation logs with `p4c-bm2-ss` (same compiler backends as the pipeline).

Experiment numbers in the manuscript changed during revision, so the table below goes by paper section. The **Suite ID** column gives the name the code uses.

| Paper section / table | Suite ID | Command | Scale |
|---|---|---|---|
| Functional results: link failure recovery | `exp1` | `python experiments/runner.py run exp1` | 6 scenarios |
| Functional results: dynamic VNF insertion | `exp2` | `python experiments/runner.py run exp2` | 7 scenarios |
| Functional results: impairment adaptation | `exp3` | `python experiments/runner.py run exp3` | 9 scenarios |
| Construct-level probes | `exp8` | `python experiments/runner.py exp8 --repeat 3` | 5 scenarios × 3 trials |
| Scenario-breadth expansion | `exp9` | `python experiments/runner.py exp9 --repeat 3` | 15 scenarios × 3 trials |
| Compiler gate vs. LLM-only gate (baselines) | `exp7` | `python experiments/runner.py exp7` | 22 scenarios × 4 systems |
| Architecture ablation | `ablation` | `python experiments/runner.py ablation` | 22 scenarios × 3 configs |
| Technique ablation | `ablation_techniques` | `python experiments/ablation_techniques.py --parallel 4` | 50 tasks × 4 conditions |
| Security evaluation, full pipeline | `exp5` | `python experiments/runner.py red-team` | 18 attacks |
| Security evaluation, reduced gates | `exp5` | `python experiments/run_redteam_configs.py both` | 18 attacks × 2 configs |
| Domain adaptation (QLoRA / DoRA) | `finetune` | see [below](#domain-adaptation) | 4 adapters, 50-task eval |
| Zero-shot base-model comparison | `zero_shot` | `python experiments/zero_shot_model_sweep.py` | 50 tasks × 4 models¹ |

¹ Pull the extra base models first: `for m in qwen2.5-coder:7b deepseek-coder:6.7b codellama:13b; do ollama pull $m; done`.

Measured wall-clock time per scenario on one L4 in the paper's runs: median 725 s (max 1,284 s) for a functional scenario through the full multi-round pipeline, 280 s for a red-team scenario, and 149 s for a single zero-shot generation. Multiply by the run counts above to plan a job.

For a smoke test, `runner.py run`, `exp7`, `exp8`, `exp9`, and `red-team` accept `--dry-run`, and the eval-set drivers accept `--limit N`.

To score a red-team log, run `python experiments/runner.py validate-red-team experiments/results/<file>.json`.

### Domain adaptation

The paper fine-tunes `Qwen/Qwen2.5-Coder-14B-Instruct` on `data/verified_corpus.jsonl`, swaps the result in as the coder model, and scores it on the 50-task eval set. Each adapter is one run:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements-finetune.txt
export LLAMA_CPP=~/llama.cpp     # built checkout of llama.cpp

CONFIG_NAME=qlora_r8  LORA_R=8  USE_DORA=0 scripts/run_finetune.sh
CONFIG_NAME=qlora_r16 LORA_R=16 USE_DORA=0 scripts/run_finetune.sh
CONFIG_NAME=dora_r8   LORA_R=8  USE_DORA=1 scripts/run_finetune.sh
CONFIG_NAME=dora_r16  LORA_R=16 USE_DORA=1 scripts/run_finetune.sh
```

The script trains, merges the adapter, quantizes to Q5_K_M GGUF, imports the model into Ollama, evaluates it, and switches the coder back to the base model when it exits.

### Running on a Slurm cluster

The paper's runs used a Slurm GPU cluster with Apptainer. [`scripts/slurm_example.sbatch`](scripts/slurm_example.sbatch) is a template for the functional suite. Edit the `#SBATCH` lines for your site, build the two containers, then run `sbatch scripts/slurm_example.sbatch`.

### Changing models

`configs/models.yaml` maps each agent role to an Ollama model. To swap one without editing the file:

```bash
python run.py swap p4_coder qwen2.5-coder:32b
```

## Datasets

| File | Records | Fields | Used by |
|---|---|---|---|
| `data/eval_set.jsonl` | 50 | `task`, `p4_code`, `scenario`, `source`, `approved` | technique ablation, domain adaptation, zero-shot comparison |
| `data/verified_corpus.jsonl` | 3,745 | same | QLoRA/DoRA training corpus |

Both sets are LLM-generated P4_16 SFC tasks. Every `p4_code` entry passed a `p4c-bm2-ss` compile check when the corpus was built, and the eval set was sampled from the same verified pool (seed 42). Neither file contains personal data or third-party source code. The scenario suites (Exp 1–3, construct probes, breadth expansion, red team) are defined inline in their `experiments/exp*.py` modules.

## Determinism and expected variance

LLM sampling is stochastic. Temperatures are fixed per role in `configs/models.yaml` (planner 0.3, coder 0.1, reviewer 0.2, tester 0.0), and the MoA coder samples three candidates at 0.3, 0.7, and 1.0. Ollama does not guarantee identical output across GPU types or driver versions, so per-scenario outcomes can flip between runs even with these settings. The paper reports construct probes and breadth expansion over 3 independent trials, with 95% Wilson score intervals. Compare reruns to the paper at the level of approval rates and intervals, not individual scenarios.

The compiler gate itself is deterministic: a given P4 program either compiles under `p4c-bm2-ss` or it does not.

## Citation

If you use P4Harness, please cite the paper ([`CITATION.cff`](CITATION.cff) has the same entry):

```bibtex
@unpublished{peppers2026p4harness,
  title  = {P4Harness: Autonomous P4 Network Policy Management and Verification
            Using Compiler-Gated Multi-Agent Pipelines},
  author = {Peppers, Sean and Gao, Tiying and Calyam, Prasad and Qu, Chengyi},
  note   = {Manuscript in preparation},
  year   = {2026}
}
```

## License

Code is released under the [MIT License](LICENSE). The datasets in `data/` are released under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) (see [`data/LICENSE`](data/LICENSE)).

Questions and bug reports: please [open an issue](https://github.com/SeanPeppers/P4Harness/issues).
