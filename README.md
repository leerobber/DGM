# Darwin Gödel Machine (DGM) — Sovereign Core Edition

> Open-ended self-improvement of coding agents via evolutionary search.
> Now running entirely on local GPU hardware through the Sovereign Core gateway.

---

## What This Is

The Darwin Gödel Machine is a system that:
1. Takes a **coding agent** as the base unit
2. Uses that agent to **improve itself** (generate diffs to its own code)
3. Evaluates the improved agent against SWE-bench coding tasks
4. If it scores better → adds it to an **archive** (stepping stones, not just the best)
5. Future generations bootstrap from the archive — not just the current best

This is open-ended evolution. The archive grows. The system gets smarter.

---

## Sovereign Core Integration

DGM now routes all inference through your local GPU cluster instead of Anthropic/OpenAI.

```
coding_agent.py  →  llm_withtools_sovereign.py  →  /v1/chat/completions
                                                      │
                                              Sovereign Core Gateway :8000
                                                      │
                                    ┌─────────────────┼─────────────────┐
                                    │                 │                 │
                               RTX 5050         Radeon 780M         Ryzen 7
                            Qwen2.5:14b      DeepSeek-Coder:6.7b   Mistral:7B
                            (proposer)         (verifier)           (fallback)
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/leerobber/DGM
cd DGM

# Configure
cp .env.example .env
# Set SOVEREIGN_GATEWAY_URL to your TatorTot IP

# Verify gateway connection
curl http://localhost:8000/health

# Run DGM (self-improvement loop)
python DGM_outer.py \
  --output_dir ./output \
  --selfimprove_size 4 \
  --n_generations 10

# Or with Sovereign-aware agent
USE_SOVEREIGN=1 python DGM_outer.py --output_dir ./output
```

---

## Architecture

### DGM_outer.py — Evolution Loop
```
Initialize archive = ['initial']
For each generation:
  Choose parent candidates from archive (by score + diversity)
  Spawn N self-improvement workers (ThreadPoolExecutor)
    Each worker: coding_agent.py mutates the agent code
                 Run SWE-bench evaluation
                 If score > threshold → add to archive
  Update DGM metadata (dgm_metadata.jsonl)
```

### coding_agent.py — The Self-Improving Agent
```
Given: a repo with a bug (SWE-bench instance)
1. Read issue description
2. Explore the codebase
3. Generate a patch via LLM (now → Sovereign Core)
4. Apply patch
5. Run tests
6. Evaluate score
```

### llm_withtools_sovereign.py — The LLM Bridge
```
Priority chain:
  1. Sovereign Core gateway /v1/chat/completions
  2. Direct Ollama (if gateway down)
  3. Original llm_withtools (cloud fallback, if Ollama down)
```

---

## Integration with KAIROS

DGM-generated improvements can feed into the KAIROS SAGE loop:

```bash
# Run DGM, then push best agent to KAIROS
python scripts/dgm_to_kairos.py \
  --dgm_output ./output \
  --gateway http://localhost:8000
```

The KAIROS archive then evolves the agent using the 4-agent SAGE loop,
combining DGM's evolutionary search with SAGE's adversarial refinement.

---

## Performance Notes

- **RTX 5050 (8GB VRAM)**: Handles Qwen2.5:14B comfortably. For 32B you'll need 4-bit quant.
- **Radeon 780M (4GB VRAM)**: Runs DeepSeek-Coder:6.7B for code verification steps.
- **Ryzen 7**: CPU fallback for Mistral:7B / smaller models.
- **Gateway routing**: Automatically selects best available backend per request.

---

## Files

| File | Purpose |
|------|---------|
| `DGM_outer.py` | Main evolution loop |
| `coding_agent.py` | Single-language coding agent |
| `coding_agent_polyglot.py` | Multi-language coding agent |
| `llm_withtools_sovereign.py` | ← **Sovereign Core LLM adapter** |
| `self_improve_step.py` | One self-improvement step |
| `analysis/` | Visualization scripts |
| `initial/` | Baseline agent (SWE-bench initial run) |

---

## License

MIT (following Meta's original DGM license)
