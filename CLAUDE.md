# zeoRAG — Claude working notes

## Kernels

Two kernels are in use in this repo — pick per-notebook, not per-repo:

| Kernel | Torch | When to use | Env path |
|---|---|---|---|
| `mrigi_tor190_v8` | **1.12.1** | **Default for every notebook**, including 23j / 30b / 30c and anything historical (23e, 30, 30a, …) | `/home/synthesisproject/miniforge3/envs/mrigi_tor190_v8` |
| `mrigi_tor190_v9` | **2.8.0+cu128** | **Only notebook 23k** (needs `peft` to load LoRA adapters) | `/home/synthesisproject/miniforge3/envs/mrigi_tor190_v9` |

v9 was created (2026-07-21) as `conda create --name mrigi_tor190_v9 --clone mrigi_tor190_v8 -y`, then `pip install "peft<0.4"`. ⚠️ The pip install **silently upgraded torch from 1.12.1 → 2.8.0+cu128** (peft's dep resolver pulled a newer torch). v9 is therefore NOT a strict superset of v8 — the torch difference matters. So far it has caused no issues for 23k, but keep it in mind. Registered as a Jupyter kernel via `python -m ipykernel install --user --name=mrigi_tor190_v9 --display-name "Python (mrigi_tor190_v9)"`.

Kernelspec metadata (Jupyter matches by `name`, not `display_name`):

```json
"kernelspec": {"display_name": "Python (mrigi_tor190_vN)", "language": "python", "name": "mrigi_tor190_vN"}
```

Every notebook has a plain-text banner at the top saying which kernel to select.

## Environment / tokens

Tokens are loaded from `/home/jupyter/Mrigi/env.sh` (parsed at the top of each notebook, since `!source` runs in a subprocess and doesn't persist into Python):

- `HF_TOKEN_BESTE` — HuggingFace token used by generation notebooks (23x)
- `OPENAI_API_TOKEN` — GPT-4.1 API token used by judge notebooks (30x)

## Vector index

`zeoRAG/faiss_index/` holds `index.faiss` (~2.1 GB) + `index.pkl` (~1.3 GB) built from the zeolite corpus. Loaded via `FAISS.load_local(..., allow_dangerous_deserialization=True)` with `SentenceTransformerEmbeddings("all-MiniLM-L6-v2")`.

## GPU / OOM

The kernel ships **torch 1.12.1**, which does NOT support `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` (that option was added in torch ≥ 2.1). Setting it raises `Unrecognized CachingAllocator option: expandable_segments` inside `AutoModelForCausalLM.from_pretrained` and blocks every model from loading. Notebook 23i still has this line at the top and needs it removed before it can run again; 23j and 23k are already patched.

Rely on the aggressive per-question `gc.collect() + torch.cuda.empty_cache()` and full-unload-between-models pattern already in 23i/23j/23k for OOM mitigation.

## HuggingFace cache vs download

- **23j** loads with `local_files_only=True` — used weights **must** already be in `~/.cache/huggingface/`. If not, the model gets logged as `load_error` in the checkpoint and skipped.
- **23k** loads the 7 `_COT` **LoRA adapters** via `peft.PeftModel.from_pretrained(base, <local_snapshot_dir>)`. Two subtleties:
  1. **Adapter file format:** the 7 adapter repos only publish `adapter_model.safetensors`. peft 0.3.x can only load `adapter_model.bin`. A one-shot conversion (`safetensors.torch.load_file` → `torch.save`) has been run against the local cache — every adapter's snapshot directory now has both `.safetensors` and a converted `.bin` alongside it. If the cache is ever wiped or a fresh adapter is added, this conversion must be re-run before 23k will load it.
  2. **Local path, not repo id:** passing `PeftModel.from_pretrained(base, "aleynabeste/…")` makes peft 0.3 call `hf_hub_download("adapter_model.bin")`, which 404s (only `.safetensors` exists on the hub). Instead, 23k resolves each adapter's repo id to its local snapshot directory (`~/.cache/huggingface/hub/models--aleynabeste--…/snapshots/<sha>/`) and passes that path; peft then reads local files without touching the Hub.

## Result-file lineage (open-ended track)

```
zeolite_openended_100.xlsx  ─┐
                              ├─ 23j (7 non-COT models, kernel v8/v9)  ─┐
                              ├─ 23k (7 COT LoRA adapters, kernel v9)   ─┴─→  results_23j_checkpoint.json (14 models merged)
                                                                                │
                                                                                └─→ 30b (dual judge: gpt-4.1 + gpt-4o, kernel v9)
                                                                                      └─→ judge_results_23j_dual_final_<ts>.json
                                                                                             └─→ 30c (per-FT-model win/loss + topic clustering, kernel v9)
```
