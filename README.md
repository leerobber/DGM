# DGM — Darwin Gödel Machine

> *An AI agent that improves its own source code. Generation by generation. No ceiling.*

[![Python](https://img.shields.io/badge/Python-3.11+-3776ab?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![Sovereign Core](https://img.shields.io/badge/Sovereign_Core-connected-00ff88?style=flat-square)](https://github.com/leerobber/sovereign-core)
[![Architecture](https://img.shields.io/badge/Architecture-Gödel_Machine-7c3aed?style=flat-square)](https://github.com/leerobber/DGM)

---

## What This Is

DGM is an implementation of the **Darwin Gödel Machine** — a self-improving AI agent that modifies its own code to solve progressively harder problems.

Named after two ideas that changed how we think about intelligence:

- **Darwin** — keep good solutions, build on them, improve generation by generation. Don't just keep the best — keep every good-enough stepping stone, because the path to breakthrough often goes through mediocre.
- **Gödel** — any system powerful enough to do mathematics can also reason about itself. If you can reason about yourself, you can rewrite yourself to be better.

Put those together: an agent that reads its own code, understands what it's doing, proposes improvements, tests them empirically, and keeps the ones that work.

---

## How It Works

```
┌────────────────────────────────────────────────────────┐
│                    DGM LOOP                            │
│                                                        │
│  1. Take a problem from the benchmark                  │
│  2. Generate a solution in code                        │
│  3. Run the solution — measure the score               │
│  4. If score > current best → new stepping stone       │
│  5. Propose modifications to the solution code         │
│  6. Test modifications empirically                     │
│  7. Keep improvements, discard regressions             │
│  8. Repeat from the new stepping stone                 │
│                                                        │
│  Result: each generation starts ahead of the last     │
└────────────────────────────────────────────────────────┘
```

**The key insight:** Don't just keep the single best solution. Keep every good-enough stepping stone. Sometimes the path to a breakthrough goes through something mediocre. Discard the mediocre and you discard the bridge.

---

## The Stepping Stones Archive

Every solution that beats the previous best gets archived — not replaced. The archive grows. Each new run can start from *any* stepping stone, not just the most recent one.

This creates **diversity of evolutionary paths** — which is what prevents getting trapped in local optima.

```
Generation 1: Score 0.45 ──► archived
Generation 2: Score 0.52 ──► archived  
Generation 3: Score 0.61 ──► archived
Generation 4: Score 0.58 ──► below threshold, discarded
Generation 5: Score 0.74 ──► archived ← branches from Gen 3, not Gen 4
Generation 6: Score 0.89 ──► Elite status
```

---

## Sovereign Core Integration

DGM uses the **Sovereign Core gateway** for all LLM inference — no cloud API required.

```python
# .env
SOVEREIGN_GATEWAY_URL=http://localhost:8000

# Inference routes through:
# RTX 5050 (Qwen2.5)    → primary
# Radeon 780M           → fallback
# Ryzen CPU             → last resort
```

All reasoning, code generation, and self-modification happens on local hardware.

---

## Quickstart

```bash
git clone https://github.com/leerobber/DGM
cd DGM
pip install -r requirements.txt

# Configure sovereign gateway
cp .env.example .env
# Set SOVEREIGN_GATEWAY_URL=http://localhost:8000

# Run the evolution loop
python main.py
```

---

## Architecture

```
dgm/
├── main.py                  # Entry point — runs the evolution loop
├── agent/
│   ├── dgm_agent.py         # Core agent — proposes and applies modifications
│   └── llm_withtools_sovereign.py  # Sovereign Core LLM adapter
├── evaluation/
│   └── evaluator.py         # Scores solutions empirically
├── archive/
│   └── stepping_stones.py   # Manages the stepping stone archive
└── benchmarks/
    └── tasks/               # Problem sets for the agent to solve
```

---

## Philosophy

The difference between a system that improves and a system that just runs is the feedback loop.

DGM closes the loop: the agent produces code, the code gets scored, the score determines what survives, what survives becomes the foundation for the next attempt.

No human in the loop. No manual intervention. Just selection pressure — the computational equivalent of evolution.

> *"I'm not trying to build the smartest system today. I'm building the one that gets smarter fastest over time."*

---

## Part of the Sovereign Stack

DGM is one node in a larger system:

| Repo | Role |
|------|------|
| [sovereign-core](https://github.com/leerobber/sovereign-core) | Gateway + KAIROS engine |
| **DGM** | Self-improving coding agent — stepping stones feed KAIROS |
| [HyperAgents](https://github.com/leerobber/HyperAgents) | Self-referential swarm agents |
| [Honcho](https://github.com/leerobber/Honcho) | Mission control dashboard |
| [contentai-pro](https://github.com/leerobber/contentai-pro) | Multi-agent content engine |

---

## Built By

**Terry Lee** — Douglasville, GA  
Self-taught systems architect. No team. No institution. Just architecture.

*Self-taught. Self-funded. Self-improving — just like the systems I build.*
