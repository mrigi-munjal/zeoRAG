# zeoRAG

Retrieval-Augmented Generation benchmarking for zeolite synthesis literature.

**Kernels used in this repo:**
- **`mrigi_tor190_v8`** — default for everything (torch 1.12.1).
- **`mrigi_tor190_v9`** — used by notebooks **23k** and **23L** (v8 clone + `peft<0.4` for loading LoRA adapters).

Every notebook that needs a specific kernel has a plain-text banner at the top telling you which one to select. See [zeoRAG/CLAUDE.md](CLAUDE.md) for env details.

**Key notebooks:**
- **21b** — retrieval-only benchmarking (title → DOI, FAISS vs BM25 vs Hybrid)
- **23e** — generation benchmarking (MCQ, 90 questions, Llama + GPT-4.1, multi-category)
- **23i / 23j / 23k / 23L** — multi-model 100-question benchmarks: **23i** = MCQ, **23j** = open-ended, **23k** = recovery run for the `_COT` LoRA adapters, **23L** = Q67 backfill for question-set parity
- **30** — LLM-as-judge quality evaluation of 23e outputs
- **30a** — LLM-as-judge for 23i (MCQ)
- **30b** — **dual-judge** (GPT-4.1 + GPT-4o) for 23j+23k open-ended answers, with judge-agreement views
- **30c** — FT-vs-base analysis: per-FT-model win/loss lists vs Llama base + GPT-4.1 autoclustered topic breakdown
- **30d** — aggregate/visualize an existing single-judge results file without re-judging
- **30e** — ⭐ **paper figures**, one set per judge (Table S1 labels, read-only, no API calls)
- **30f** — GPT-5.2 no-context generation + judging (SI reference point only)
- **30g** — ⭐ **fixes the answer-extraction bug and re-judges** — produces the results reported below

**Run order for the open-ended track:** `23j` → `23k` → **`23L`** → **`30g`** → `30e`
(`30f` for the SI reference point; `30b` is superseded by `30g`.)

**Jump to:** [Methods](#methods-open-ended-evaluation) · [Results](#results-open-ended-dual-judge-evaluation)

---

# Methods: Open-Ended Evaluation

> Everything in this section describes the **open-ended** track: generation in **23j** (+ **23k** for the LoRA adapters, **23L** for question-set parity), judging in **30g**, figures in **30e**. The older MCQ track (23e/23i → 30/30a) is documented in the notebook index further down.

## 1. Corpus and retrieval index

| Component | Value |
|---|---|
| Source corpus | `filtered_records_on_zeolite.json` (1.4 GB), zeolite-filtered scientific literature |
| Embedding model | `sentence-transformers/all-MiniLM-L6-v2` (384-dim) |
| Vector store | FAISS (`faiss_index/`) |
| Indexed chunks | **1,474,439** vectors |
| Chunk size | mean 883 chars, median 788 (range 58–5,223) |
| Chunk metadata | `doi`, `source` |

**Retrieval method evaluated:** Maximal Marginal Relevance (MMR) via `FAISS.max_marginal_relevance_search(query, k=15)`. Retrieved chunks are concatenated with `\n\n` separators to form the context block. A **no-context** condition (the model answers from parametric knowledge alone) serves as the ablation baseline.

## 2. Question set

100 open-ended questions derived from `zeolite_mcq_selected_combined__modified_beste.xlsx`:

- Started from the 100-item "Selected MCQs" sheet (GPT-generated questions grounded in specific papers).
- **14 items flagged by domain experts** (marked with red cell fill) were removed and replaced with the first 14 items from the "replacements" sheet.
- **Q67 was subsequently swapped** — the original item (Pd/ZSM-22 hydroisomerization) produced a context block that triggered CUDA OOM on every model at k=15; it was replaced with the previously-unused 15th replacement item (Al organization in SSZ-13). Pre-swap file preserved as `zeolite_openended_100_pre_q67swap.xlsx`. Because the swap happened
  *between* the 23j and 23k runs, the two groups of models were briefly evaluated on different
  question sets; **notebook 23L** regenerates Q67 for the affected models so all 9 Table S1 variants
  share one set.
- **Conversion to open-ended:** only the question stem is presented to the model. The A–E answer options are discarded; the text of the correct option becomes the **reference (gold) answer** supplied to the judge.

Resulting set (`zeolite_openended_100.xlsx`): 100 questions, **99 unique DOIs**, question length mean 230 chars (127–367), gold answer length mean 174 chars.

## 3. Models evaluated

All open-weight models are Llama-3-8B derivatives, evaluated in fp16 on a single GPU with
sequential load/unload. **These 10 variants — and only these — are reported in the paper (Table S1).**

| # | Table S1 label | Internal name | Base model | DAPT corpus | CoT FT |
|---|---|---|---|---|---|
| 1 | GPT-5.2 | *(API)* | GPT-5.2 | None | No |
| 2 | Llama-3-8B-Instruct | `Llama-3-8B-Instruct` | Llama-3-8B-Instruct | None | No |
| 3 | Llama-3-8B-Instruct + CoT FT | `base_llama_COT` | Llama-3-8B-Instruct | None | Yes |
| 4 | Llama-3-8B-Instruct, broad zeolite abstracts DAPT | `DAPT_LR1e5` | Llama-3-8B-Instruct | Broad zeolite abstracts | No |
| 5 | Llama-3-8B-Instruct, broad zeolite abstracts DAPT + CoT FT | `DAPT_LR1e5_COT` | Llama-3-8B-Instruct | Broad zeolite abstracts | Yes |
| 6 | Llama-3-8B-Instruct, synthesis abstracts DAPT | `synv2V2_step80` | Llama-3-8B-Instruct | Synthesis abstracts | No |
| 7 | Llama-3-8B-Instruct, synthesis abstracts DAPT + CoT FT | `synv2V2_step80_COT` | Llama-3-8B-Instruct | Synthesis abstracts | Yes |
| 8 | Llama-3-8B **base**, synthesis abstracts DAPT + CoT FT | `synv2_base_step80_COT` | Llama-3-8B **base** | Synthesis abstracts | Yes |
| 9 | Llama-3-8B-Instruct, synthesis full-paper DAPT | `fullpaper_120M_LR1e5` | Llama-3-8B-Instruct | Synthesis full papers | No |
| 10 | Llama-3-8B-Instruct, synthesis full-paper DAPT + CoT FT | `fullpaper_120M_COT` | Llama-3-8B-Instruct | Synthesis full papers | Yes |

Row 1 (GPT-5.2) is an API model reported in the
[SI section](#supplementary--gpt-52-reference-point-si-only-not-a-main-paper-result) only; rows
2–10 are the nine open-weight variants in every Results table and figure.

> **Not reported.** The pipeline also contains `synv2V2_final`, `synv2V2_final_COT`,
> `synv2_base_final`, `synv2_base_final_COT` and `synv2_base_step80`. They were generated and in
> some cases judged, but are **excluded from the paper** and from every table and figure here.

> **Note on the stored generations.** The raw 23j/23k outputs contain the echoed prompt; the
> answers used for all reported results are the re-extracted ones written by **30g** to
> `results_23j_clean_*.json`. See [The extraction bug](#the-extraction-bug-and-what-it-changed).

`_COT` variants are **LoRA adapters** (PEFT, `r=8`, `lora_alpha=16`, `lora_dropout=0.05`, target modules `q_proj`+`v_proj`) applied on top of their respective base checkpoints — not standalone models. Each adapter's base is declared in its `adapter_config.json`.

**Generation parameters:** `max_new_tokens=400`, `temperature=0.1`, `do_sample=True`, `torch_dtype=float16`. Prompts are applied via the Llama-3 chat template; DAPT checkpoints that ship without a `chat_template` are backfilled with the stock Llama-3 template.

**Generation prompt (with context):**
```
You are an expert on zeolite synthesis, chemistry, and catalysis. Answer the following
question using the provided context together with your own knowledge. Give a clear,
focused answer in 3–5 sentences (or fewer if the question is simple). Do not invent
citations. Do not restate the question.

Context:
{context}

Question: {question}

Answer:
```
The no-context prompt is identical with the context block and its instruction clause removed.

## 4. LLM-as-judge protocol

Two independent judges — **`gpt-4.1`** and **`gpt-4o`** — score every response on four dimensions, each 1–10 with a written rationale. Judge calls use `temperature=0.1`, `max_tokens=200`, `response_format={"type": "json_object"}`. Both judges receive **byte-identical prompts**; the only difference is the model name. Judges are called sequentially within a question; parallelism (8 workers) is across questions.

> **All four tasks are computed and stored, but only correctness and completeness are reported** in the Results figures. Context relevance cannot separate models (identical retriever for all), and faithfulness has the weakest inter-judge agreement — see [Why only two metrics are reported](#why-only-two-metrics-are-reported).

### 4.1 What each metric measures, and what the judge sees

Which inputs a judge receives is the operational definition of each metric — it determines what the score *can* be sensitive to.

| Metric | Operational definition | Inputs shown to judge | Deliberately **not** shown |
|---|---|---|---|
| **Context Relevance** | Is the retrieved passage set useful for answering this query? Property of the **retriever**, not the generator. | query, context | model's answer, gold answer |
| **Correctness** | Does the answer match the gold answer *in scientific substance, not wording*? | query, **gold answer**, response | context |
| **Completeness** | Does the answer address what the question actually asks? | query, response | gold answer, context |
| **Faithfulness** | Are the claims grounded in the context / uncontested zeolite science, i.e. free of hallucination? | query, context, response | gold answer |

Three design decisions worth noting for the write-up:

- **Correctness is the only metric that sees the gold answer.** Completeness and Faithfulness are deliberately gold-blind so they measure *response quality* independent of whether the answer happens to be right.
- **Faithfulness is explicitly decoupled from Correctness** by prompt instruction ("Do NOT reward or penalize based on whether the response is correct"). A confidently wrong but well-grounded answer should score high on faithfulness and low on correctness. This lets us separate *hallucination* from *error*.
- **Correctness does not see the context.** This prevents the judge from crediting an answer merely for echoing retrieved text.

Handling of undefined cases: Context Relevance is undefined for the no-context condition → recorded as `-1` and excluded from all means. Generator `ERROR`/`INVALID` outputs → all four scores `-1`, excluded. Context is truncated to 3,000 chars before being shown to a judge.

### 4.2 Shared system prompt

```
You are an expert evaluator for a Retrieval-Augmented Generation (RAG) system
focused on zeolite synthesis, catalysis, and environmental applications.
You will evaluate the quality of retrieved contexts and open-ended generated responses
against a reference (gold) answer.
Always respond in valid JSON format.
```

### 4.3 Task 1 — Context Relevance

```
Given the following query and retrieved context, rate how relevant the context is
for answering the query.

**Scoring Rubric (1-10):**
- 1-2: Completely irrelevant — context has no connection to the query topic
- 3-4: Mostly irrelevant — context touches on the general domain but doesn't address
       the specific question
- 5-6: Partially relevant — context contains some useful information but misses key aspects
- 7-8: Mostly relevant — context addresses the main topic and provides useful information
       for answering
- 9-10: Highly relevant — context directly addresses the query with specific, useful information

**Query:**
{query}

**Retrieved Context:**
{context}

Respond with ONLY a JSON object in this exact format:
{"score": <integer 1-10>, "reasoning": "<one to two sentences explaining your score>"}
```

### 4.4 Task 2 — Correctness

```
Given the following query, a reference (gold) answer, and a model's open-ended response,
rate how correct the response is.

Judge on scientific substance, not wording. The response may phrase things differently,
be more or less detailed, or add related information — that is fine as long as the core
claim matches the reference and any additional claims are accurate.

**Scoring Rubric (1-10):**
- 1-2: Completely incorrect — asserts something that contradicts the reference answer
- 3-4: Mostly incorrect — misses the main point, but shows partial understanding of the topic
- 5-6: Partially correct — captures part of the reference answer but is missing or muddling
       the key idea
- 7-8: Mostly correct — captures the key idea of the reference answer, with minor gaps or
       imprecise phrasing
- 9-10: Fully correct — captures the reference answer's key idea clearly and accurately
        (extra correct detail is fine)

**Query:**
{query}

**Reference (Gold) Answer:**
{gold_answer}

**Model's Open-Ended Response:**
{response}

Respond with ONLY a JSON object in this exact format:
{"score": <integer 1-10>, "reasoning": "<one to two sentences explaining your score>"}
```

### 4.5 Task 3 — Completeness

```
Given the following query and a model's open-ended response, rate how completely the
response addresses the question.

Judge how much of what the question is asking for is present — the main mechanism, the
reason, or the required piece of reasoning. Do not penalize concise answers if they cover
the essential point; do penalize hand-wavy or partial answers that skip the substantive part.

**Scoring Rubric (1-10):**
- 1-2: No meaningful response — empty, garbled, refuses, or completely off-topic
- 3-4: Minimal response — mentions the topic but does not really answer the question
- 5-6: Partial response — addresses part of the question but leaves the key aspect unexplained
- 7-8: Good response — addresses the question with a clear, relevant explanation of the main point
- 9-10: Excellent response — fully addresses the question with a thorough, well-reasoned explanation

**Query:**
{query}

**Model's Open-Ended Response:**
{response}

Respond with ONLY a JSON object in this exact format:
{"score": <integer 1-10>, "reasoning": "<one to two sentences explaining your score>"}
```

### 4.6 Task 4 — Faithfulness

```
Given the following query, retrieved context, and a model's open-ended response, rate how
faithful the response is — i.e., how well its claims are grounded in the retrieved context
or in uncontested, well-known zeolite science, and how free it is of hallucinated /
invented details.

Do NOT reward or penalize based on whether the response is correct (that is scored
separately). Score only whether the response invents specific facts, cites nonexistent
papers/values, or makes claims that neither the context nor common knowledge in zeolite
science would support.

**Scoring Rubric (1-10):**
- 1-2: Response is dominated by hallucinated / fabricated claims (invented numbers,
       nonexistent references, made-up mechanisms) not supported by context or standard knowledge
- 3-4: Response has multiple unsupported / suspicious claims mixed with grounded ones
- 5-6: Response has a few unsupported or borderline claims but is largely grounded
- 7-8: Response is well-grounded, with at most one minor unsupported detail
- 9-10: Every substantive claim is either directly supported by the context or is
        uncontroversial zeolite-science common knowledge

If no context was provided (i.e., this is a no-context run), judge purely against
uncontroversial zeolite-science common knowledge.

**Query:**
{query}

**Retrieved Context (may be empty):**
{context}

**Model's Open-Ended Response:**
{response}

Respond with ONLY a JSON object in this exact format:
{"score": <integer 1-10>, "reasoning": "<one to two sentences explaining your score>"}
```

When no context is available, the `{context}` slot is filled with the literal string
`No context — judge purely on standard zeolite-science common knowledge.`

---

# Results: Open-Ended Dual-Judge Evaluation

**Source data:** `judge_results_23j_clean_20260803_201225.json` (written by **30g**; judges `gpt-4o` + `gpt-4.1`, prompt-stripped answers).
**Figures and tables:** notebook **30e**, run `20260803_212837` → `figures_30e/`.

> Every number in this section is reproduced verbatim from that 30e run — specifically
> `figures_30e/paper_table_gpt-4o_20260803_212837.csv` (primary) and
> `paper_table_gpt-4.1_20260803_212837.csv` (robustness check). Re-running 30e against the same
> judge file regenerates them exactly; re-running it after a new 30g pass will supersede them.
**Primary judge: `gpt-4o`.** `gpt-4.1` scored every response independently and is reported throughout as a **robustness check**, not as a co-equal result.
**Metrics:** correctness and completeness (see [Why only two metrics](#why-only-two-metrics-are-reported)).
**Scope:** the 9 open-weight Table S1 variants, 100 questions, 2 retrieval conditions. GPT-5.2 is SI-only.

> ### ⚠ These numbers supersede all earlier versions of this file
> Results published before 2026-08-03 were computed on **contaminated judge inputs**. See
> [The extraction bug](#the-extraction-bug-and-what-it-changed) — every number below has been
> recomputed on corrected data, and two conclusions changed materially.

---

## The extraction bug, and what it changed

Notebooks 23j/23k stored the **entire prompt** in `model_answer`, not the model's answer.
`extract_completion()` split on the tokenised tag `<|start_header_id|>assistant<|end_header_id|>`,
but the HuggingFace pipeline decodes special tokens away before that string exists — so the tag
appeared in **0 of 100** records for every model, the split target was never found, and the
function returned its input unchanged. Prompt + retrieved context + answer was then judged as
"the response".

| Condition | Fraction of judged text that was the actual answer |
|---|---|
| no context | ~50% (rest is the instruction and question) |
| **RAG (MMR k=15)** | **~3%** (rest is 15 retrieved chunks) |

Contamination correlated with score at p < 1e-5 for **both** metrics (r ≈ 0.14–0.36 correctness,
0.35–0.52 completeness). **Notebook 30g** re-extracts every answer with a prompt-anchored
extractor and re-judges; **notebook 23L** separately fixed a question-set mismatch (below).

**What changed once corrected:**

| | Before (contaminated) | After (corrected) |
|---|---|---|
| Finding 2 — DAPT-only + retrieval | ambiguous; 2 of 3 models ≈ 0 under `gpt-4o` | **unambiguous: all 3 negative under both judges** |
| Judge agreement (correctness) | r = 0.880, 92% within ±1 | **r = 0.913, 95% within ±1** |
| "no context beats RAG on completeness" | apparent effect | **artefact — gone** |

The completeness "finding" in earlier drafts was an artefact: a 15,000-character wall of retrieved
passages reads as unfocused to a gold-blind judge, and the effect was largest for the models whose
real answers were shortest.

**A second data defect, fixed by 23L.** The models had been evaluated on *two different question
sets*, differing at index 67: the 23j run used the original Pd/ZSM-22 item (which OOM'd at MMR,
scoring only 99 questions), while the 23k run used the SSZ-13 replacement. 23L regenerates Q67 for
the affected models so all 9 share one 100-question set.

---

## Main results

Mean ± SEM. Indented rows are the `+ CoT FT` variant of the row above.

### Primary judge: `gpt-4o`

| Model (Table S1) | Corr. no context | Corr. RAG | Compl. no context | Compl. RAG |
|---|---|---|---|---|
| Llama-3-8B-Instruct | 7.41 ± 0.16 | 7.41 ± 0.18 | 8.63 ± 0.05 | 8.17 ± 0.10 |
| &nbsp;&nbsp;+ CoT FT | 7.94 ± 0.13 | 8.08 ± 0.14 | 8.99 ± 0.01 | 8.91 ± 0.04 |
| Broad zeolite abstracts DAPT | 7.26 ± 0.18 | 6.52 ± 0.20 | 8.56 ± 0.06 | 6.93 ± 0.16 |
| &nbsp;&nbsp;+ CoT FT | 7.68 ± 0.17 | 8.04 ± 0.16 | 8.90 ± 0.05 | 8.85 ± 0.06 |
| Synthesis abstracts DAPT | 7.33 ± 0.18 | 6.89 ± 0.18 | 8.65 ± 0.06 | 7.31 ± 0.14 |
| &nbsp;&nbsp;+ CoT FT | 7.76 ± 0.14 | **8.21 ± 0.12** | 8.98 ± 0.02 | 8.92 ± 0.04 |
| Llama-3-8B **base**, synth. abstracts DAPT + CoT FT | 4.97 ± 0.34 | 7.53 ± 0.19 | 5.73 ± 0.39 | 8.29 ± 0.15 |
| Synthesis full-paper DAPT | 7.08 ± 0.19 | 6.48 ± 0.19 | 8.41 ± 0.09 | 7.03 ± 0.16 |
| &nbsp;&nbsp;+ CoT FT | 7.83 ± 0.14 | 8.13 ± 0.14 | 9.00 ± 0.00 | 8.84 ± 0.05 |

> **Figures 1–2 vs this table — stock Llama-3-8B-Instruct.** The table reports the original run
> (7.41 / 7.41 correctness), so that every model in it comes from one consistent generation pass.
> **Figures 1 and 2 instead show the 23M replicate for that row** (7.26 no-context / 7.30 RAG,
> `gpt-4o` correctness only), because the replicate is the more recent measurement of that model.
> The two differ by ~0.15 points, which *is* the useful number — see
> [Run-to-run stability](#run-to-run-stability-notebook-23m).

### Robustness check: `gpt-4.1`

Same data, second judge. Reported to show the conclusions do not depend on judge choice.


| Model (Table S1) | Corr. no context | Corr. RAG | Compl. no context | Compl. RAG |
|---|---|---|---|---|
| Llama-3-8B-Instruct | 7.42 ± 0.15 | 7.56 ± 0.17 | 7.95 ± 0.09 | 7.58 ± 0.12 |
| &nbsp;&nbsp;+ CoT FT | 8.34 ± 0.13 | **8.76 ± 0.14** | 9.06 ± 0.09 | 8.93 ± 0.09 |
| Broad zeolite abstracts DAPT | 7.40 ± 0.16 | 6.51 ± 0.17 | 8.02 ± 0.10 | 6.54 ± 0.14 |
| &nbsp;&nbsp;+ CoT FT | 8.16 ± 0.16 | 8.53 ± 0.15 | 8.97 ± 0.11 | 8.78 ± 0.10 |
| Synthesis abstracts DAPT | 7.40 ± 0.16 | 6.96 ± 0.16 | 7.94 ± 0.10 | 6.83 ± 0.13 |
| &nbsp;&nbsp;+ CoT FT | 8.44 ± 0.13 | 8.75 ± 0.13 | 9.22 ± 0.07 | 9.00 ± 0.08 |
| Llama-3-8B **base**, synth. abstracts DAPT + CoT FT | 5.07 ± 0.35 | 7.87 ± 0.19 | 5.65 ± 0.38 | 8.10 ± 0.17 |
| Synthesis full-paper DAPT | 7.29 ± 0.17 | 6.56 ± 0.15 | 7.81 ± 0.12 | 6.66 ± 0.14 |
| &nbsp;&nbsp;+ CoT FT | 8.46 ± 0.14 | 8.57 ± 0.15 | 9.31 ± 0.08 | 8.83 ± 0.11 |

Under retrieval, **every `+ CoT FT` row outranks every non-CoT row** — under the primary judge and confirmed by the robustness check.

---

## Finding 1 — CoT fine-tuning is the dominant effect, and retrieval amplifies it

All 16 pairwise comparisons (4 pairs × 2 conditions × 2 judges) are significant; 15 of 16 at
p < 0.001. Paired per-question differences, Wilcoxon signed-rank:

**Primary judge `gpt-4o`**

| Pair | Corr. no context | Corr. RAG | Compl. no context | Compl. RAG |
|---|---|---|---|---|
| Llama-3-8B-Instruct (no DAPT) | +0.53 \*\* | **+0.67** \*\*\* | +0.36 \*\*\* | **+0.74** \*\*\* |
| Broad zeolite abstracts DAPT | +0.42 \* | **+1.52** \*\*\* | +0.34 \*\*\* | **+1.92** \*\*\* |
| Synthesis abstracts DAPT | +0.43 \* | **+1.32** \*\*\* | +0.33 \*\*\* | **+1.61** \*\*\* |
| Synthesis full-paper DAPT | +0.75 \*\*\* | **+1.65** \*\*\* | +0.59 \*\*\* | **+1.81** \*\*\* |

**Robustness check `gpt-4.1`** — same direction, larger magnitudes throughout

| Pair | Corr. no context | Corr. RAG | Compl. no context | Compl. RAG |
|---|---|---|---|---|
| Llama-3-8B-Instruct (no DAPT) | +0.92 \*\*\* | **+1.20** \*\*\* | +1.11 \*\*\* | **+1.35** \*\*\* |
| Broad zeolite abstracts DAPT | +0.76 \*\*\* | **+2.02** \*\*\* | +0.95 \*\*\* | **+2.24** \*\*\* |
| Synthesis abstracts DAPT | +1.04 \*\*\* | **+1.79** \*\*\* | +1.28 \*\*\* | **+2.17** \*\*\* |
| Synthesis full-paper DAPT | +1.17 \*\*\* | **+2.01** \*\*\* | +1.50 \*\*\* | **+2.17** \*\*\* |

Pooled over the 4 pairs, **primary judge `gpt-4o`**:

| Metric | Condition | n | Mean Δ | Cohen's d | p |
|---|---|---|---|---|---|
| Correctness | no context | 400 | +0.53 | 0.29 | 2.1 × 10⁻⁸ |
| Correctness | **RAG** | 400 | **+1.29** | **0.71** | 2.6 × 10⁻³² |
| Completeness | no context | 400 | +0.41 | 0.54 | 1.4 × 10⁻²³ |
| Completeness | **RAG** | 400 | **+1.52** | **1.05** | 8.5 × 10⁻⁵² |

<sub>Pooling both judges (n = 800) gives +0.75 / **+1.52** correctness and +0.81 / **+1.75**
completeness — same pattern, larger magnitudes, since `gpt-4.1` scores higher throughout.</sub>

**The CoT advantage roughly doubles once retrieval is available** (d = 0.29 → 0.71 correctness;
0.54 → 1.05 completeness under the primary judge). CoT is not simply adding knowledge — it is teaching the model to *use
context it is given*. **Robust: replicates under the robustness-check judge on every pair and both metrics.**

## Finding 2 — Domain-adaptive pretraining alone makes models *worse* at using retrieval

Change in correctness when MMR context is added, with paired Wilcoxon signed-rank tests on
matched questions:

| Model | **`gpt-4o`** (primary) | `gpt-4.1` (check) |
|---|---|---|
| Llama-3-8B-Instruct (no DAPT) | +0.00 n.s. | +0.14 n.s. |
| **Broad zeolite abstracts DAPT** | **−0.74** \*\* | **−0.89** \*\*\* |
| **Synthesis abstracts DAPT** | **−0.44** \* | **−0.44** \* |
| **Synthesis full-paper DAPT** | **−0.60** \* | **−0.73** \*\*\* |
| Llama-3-8B-Instruct + CoT FT | +0.14 n.s. | +0.42 \* |
| Broad abstracts DAPT + CoT FT | +0.36 n.s. | +0.37 n.s. |
| Synthesis abstracts DAPT + CoT FT | +0.45 \* | +0.31 n.s. |
| Synthesis full-paper DAPT + CoT FT | +0.30 n.s. | +0.11 n.s. |
| Llama-3-8B base + CoT FT | **+2.62** \*\*\* | **+2.88** \*\*\* |

**All three DAPT-only checkpoints are significantly harmed by retrieval, under both judges** — the
only significant *negative* effects in the table. No CoT variant is harmed. Completeness is starker
still: the DAPT-only models drop 1.2–1.7 points when context is added.

> **On stock Llama-3-8B-Instruct.** Its +0.00 under `gpt-4o` is a coincidence of a balanced split,
> not evidence that retrieval does nothing to individual answers: **81 of 100 per-question scores
> change** (45 up, 36 down), and the score distribution shifts noticeably (RAG produces both more
> 9s — 39 vs 29 — and more 3–5s). What is genuine is that the gains and losses cancel, so there is
> **no significant net effect** (p = 0.96 under `gpt-4o`, p = 0.45 under `gpt-4.1`). Retrieval
> reshuffles which questions it answers well without improving it overall.

Reading: DAPT on raw zeolite text strengthens the parametric prior while eroding
instruction-following, so a long retrieved passage becomes a distraction rather than evidence. CoT
fine-tuning restores the ability to condition on context. **DAPT and CoT FT are not independent
contributions — DAPT only pays off when paired with CoT.**

> This finding was *ambiguous* before the extraction fix (two of three models sat at ≈ 0 under
> `gpt-4o`). On corrected data it is consistent and unambiguous across both judges — the bug had
> been masking it.

## Finding 3 — The best single configuration is not resolved by this data

| Judge | 1st | 2nd | 3rd |
|---|---|---|---|
| **`gpt-4o`** (primary) | Synth. abstracts DAPT + CoT (**8.21**) | Full-paper DAPT + CoT (8.13) | Instruct + CoT (8.08) |
| `gpt-4.1` (check) | Instruct + CoT (8.76) | Synth. abstracts DAPT + CoT (8.75) | Full-paper DAPT + CoT (8.57) |

Under the primary judge the best configuration is **Synthesis abstracts DAPT + CoT FT (8.21)** —
but the top three are separated by only **0.13 points**, and the robustness check reorders them
(placing Instruct + CoT first, by 0.01 over Synth. abstracts DAPT + CoT). **The margin is inside
judge-disagreement noise, so no claim about which single configuration is best is supported.** What *is* supported: all top slots are `+ CoT FT`
variants, and CoT-on-stock-Llama is competitive with DAPT + CoT, so the domain corpora add no
clear benefit over CoT alone on this benchmark.

## Finding 4 — Instruction tuning is a prerequisite

`Llama-3-8B base, synthesis abstracts DAPT + CoT FT` — the only variant built on the **base**
rather than **Instruct** checkpoint — is by far the weakest without context (**4.97 / 5.07** vs
7.68–7.94 and 8.16–8.46 for its Instruct-based siblings), with 2× the standard error. Retrieval
recovers it substantially (**+2.56 / +2.80**, the largest retrieval gain of any model) but it still
trails. CoT fine-tuning does not substitute for instruction tuning. **Robust under both judges.**

---

## Run-to-run stability (notebook 23M)

Generation uses `temperature=0.1, do_sample=True`, so the pipeline is stochastic. **23M** re-ran
stock `Llama-3-8B-Instruct` on the identical 100 questions, with the fixed extractor, and re-judged
correctness with `gpt-4o`:

| | no context | RAG (MMR k=15) | gap |
|---|---|---|---|
| Original run | 7.41 | 7.41 | **+0.00** |
| 23M replicate | 7.26 | 7.30 | **+0.04** |

Two things follow:

1. **The null result reproduces.** The no-context vs RAG gap is +0.00 in one run and +0.04 in the
   other — retrieval has no net effect on stock Llama-3-8B-Instruct under either.
2. **Single-run noise is ~0.15 points.** Both condition means moved by roughly that much between
   two identical runs. **Any reported difference smaller than ~0.15 points should not be
   interpreted** — which is precisely the situation in
   [Finding 3](#finding-3--the-best-single-configuration-is-not-resolved-by-this-data), where the
   top three configurations are separated by 0.13.

This is a single-model estimate; a full multi-model replication would tighten it.

## Judge reliability — `gpt-4.1` as robustness check

| Metric | Pearson r (per question) | Agree within ±1 | n |
|---|---|---|---|
| Correctness | **0.913** | 95% | 1,790 |
| Completeness | 0.874 | 92% | 1,790 |

Model-level correlation on correctness is **r = 0.983**. Agreement improved measurably after the
extraction fix (correctness r 0.880 → 0.913) — unsurprising, since the judges now read a ~600
character answer rather than a 15,000-character blob.

`gpt-4.1` scores systematically **higher** than the primary judge `gpt-4o` (visible as points above
the y = x line in Figure 5), so absolute values in the primary tables are the more conservative of
the two. Findings 1, 2 and 4 replicate under the check; Finding 3 does not, and is reported as
unresolved.

**Why `gpt-4o` is primary:** it is the more conservative scorer, and it is the judge used for the
figures in `30e` (`PRIMARY_JUDGE = 'gpt-4o'`). There is no principled reason to treat either model
as ground truth, which is why both are reported rather than one being justified over the other.

## Why only two metrics are reported

**Context relevance** scores the *query ↔ context* match. Every model receives identical retrieved
context from one shared MMR retriever, so the model never enters the computation — across all 9
models it varied by **0.098 points**. A useful retriever sanity check (~7.8/10); useless for
comparing models.

**Faithfulness** had the weakest inter-judge agreement of the four tasks (r = 0.656), spanned only
1.08 points across all models under RAG, and correlated r = 0.62 with correctness. Both were
computed and remain in the source JSON.

## Data-quality notes for the SI

**1. `Llama-3-8B base + CoT FT` scores over 91/100 questions under RAG.** Nine generations
(indices 3, 25, 28, 35, 42, 43, 58, 77, 82) produced **no answer at all** — the raw output ends
exactly at the prompt's `Answer:` token with nothing following. Diagnosis: as a non-instruction-tuned
base LM, it treats the prompt as text to continue rather than an instruction to follow, and under
the long RAG context it exhausted its 400-token budget continuing the passage — in several cases
generating further *question* text — before reaching an answer. Evidence this is model behaviour
and not a pipeline defect: the three Instruct-based CoT models have **zero** invalid records on the
same questions with the same contexts, and the same model has only 1 invalid without context
(where the prompt is ~600 chars instead of ~15,000). Its valid answers are also the longest of any
model (960 chars mean vs 717 for its Instruct sibling), consistent with a model that does not know
when to stop. Its RAG means are therefore over a 91-question subset; all paired comparisons use
matched questions and are unaffected.

**2. `gpt-4o` saturates on completeness.** For `Synthesis full-paper DAPT + CoT FT` without
context, `gpt-4o` assigned **exactly 9 to all 100 questions** (SEM 0.00), while `gpt-4.1` spread
across 6–10 on the same answers ({6:2, 7:2, 8:2, 9:51, 10:43}). That cell carries no variance and
cannot support a comparison. Completeness generally shows less spread under `gpt-4o` — a reason to
lead with correctness, which discriminates under both judges.

## Result completeness

Every reported cell is **100/100 questions** for both judges and both metrics, with one documented
exception:

| Model | no context | RAG |
|---|---|---|
| Llama-3-8B **base**, synthesis abstracts DAPT + CoT FT | 99/100 | **91/100** |
| *all other 8 open-weight models* | 100/100 | 100/100 |
| GPT-5.2 (no-context only) | 100/100 | — |

The shortfall is a genuine property of that model, not missing data — see the
[SI data-quality notes](#data-quality-notes-for-the-si). Both judges fail on exactly the same
questions, and all paired comparisons use matched questions, so the shortfall does not bias any
delta reported here.

## Limitations

- **Correctness and completeness correlate at r ≈ 0.78** — not independent evidence. Completeness
  partly rewards thoroughness, which CoT models produce by construction.
- **Finding 3 is unresolved** — under the primary judge the best configuration is Synthesis
  abstracts DAPT + CoT FT, but the top three sit within judge-disagreement noise and the check
  judge reorders them.
- **One retriever, one k** (MMR, k = 15) and **one question set** (100 items, no cross-dataset
  replication).
- **The question set is LLM-generated** from paper abstracts, with the correct MCQ option as the
  gold answer; judges are OpenAI models. Relative comparisons among open-weight models are
  unaffected (identical setup), but see the SI caveat on GPT-5.2.

## Supplementary — GPT-5.2 reference point (SI only, not a main-paper result)

> **Framing.** This belongs in the **SI, not the main paper.** The objective of this work is *not*
> to beat a frontier closed model, but to characterise how DAPT, CoT FT and retrieval interact in
> an **8B open-weight model runnable locally on a single GPU** over a private literature corpus.
> GPT-5.2 is reported only to locate the benchmark on an absolute scale. Cost, latency, data
> residency and reproducibility differ by orders of magnitude, so a head-to-head ranking would be a
> category error.

Notebook **30f** generated GPT-5.2 answers with the verbatim no-context prompt from 23j; **30g**
judged them under the identical clean protocol. GPT-5.2 was given **no retrieval**.

| | **`gpt-4o`** (primary) | `gpt-4.1` (check) |
|---|---|---|
| **GPT-5.2 correctness (no context)** | **9.14** | **9.91** |
| **GPT-5.2 completeness (no context)** | **9.03** | **10.00** |
| Best open-weight, no context | 7.94 | 8.46 |
| Best open-weight, with RAG | 8.21 | 8.76 |

**Two caveats that matter more than the ranking:**

1. **The benchmark is saturated at that capability level.** 9.14–9.91 out of 10, and a perfect
   10.00 completeness under `gpt-4.1`, means this question set no longer discriminates among
   frontier models. It discriminates well among 8B variants (4.97–8.21), which is what it was built
   for.
2. **Possible circularity.** The questions were LLM-generated, the gold answers come from that same
   generation step, and both judges are OpenAI models. A frontier OpenAI model answering
   OpenAI-written questions graded by OpenAI judges may benefit from shared conventions an 8B Llama
   derivative does not share.

## Pending

| Item | Status |
|---|---|
| Extraction fix + re-judge (30g) | **Done** — both judges, all 9 models |
| Q67 question-set parity (23L) | **Done** — all 9 models on one 100-question set |
| GPT-5.2 (SI) | **Done** — 30f generation, 30g clean judging |
| Third judge to resolve Finding 3 | Recommended if the "best configuration" claim matters |

## Corpus Analysis & Data Preparation

### Notebook 22 — ZeoSyn DOIs vs Corpus DOIs (Overlap Analysis)

Analyses the intersection between the **ZeoSyn** structured database (`ZEOSYN.xlsx`) and the two zeolite literature JSON corpora (`filtered_records_on_zeolite.json` and `filtered_records_on_additionalkeywords.json`). This is a prerequisite for understanding which frameworks can actually be benchmarked with ground-truth DOI labels.

**Inputs:** `ZEOSYN.xlsx`, `filtered_records_on_zeolite.json`, `filtered_records_on_additionalkeywords.json`

**Analysis performed:**
- Counts unique DOIs in each source
- Computes pairwise and three-way intersections (Excel only, JSON only, zeolite JSON only, all three)
- Filters ZeoSyn to only rows whose DOIs are present in the zeolite corpus

**Output:** `filtered_ZEOSYN.xlsx` — ZeoSyn entries restricted to DOIs that exist in the indexed corpus. Used downstream by notebooks 21a, 26, 27, 28, 29.

---

## Retrieval Benchmarking (No Generation)

### Notebook 21a — ZeoSyn Top-K Retrieval Quality (Framework-Aware)

This notebook benchmarks retrieval using **ZeoSyn** — a database mapping zeolite framework types to DOIs. Instead of using paper titles as queries, it uses zeolite synthesis questions and checks whether the retrieved documents match the correct **framework-specific DOIs**.

**Dataset:** `ZEOSYN.xlsx` (framework-to-DOI mapping) + `zeolite_synthesis_questions.txt`

**Retrieval method:** FAISS (semantic similarity search)

**K values evaluated:** Hit@1, Hit@3, Hit@5

**Key distinction:** Correctness is judged by whether the retrieved DOI belongs to the right zeolite framework type, not just an exact DOI match. Results saved to `zeosyn_benchmark_details.json` and plotted as a bar chart.

### Notebook 21b — Benchmarking for Retrieval Quality (Title → DOI) ⭐ *Best retrieval benchmark*

Benchmarks **retrieval-only** performance using paper **titles as queries** and checking whether the correct **DOI** is recovered from the vector database.

**Test sets:** n=100 and n=200 (randomly sampled zeolite papers with valid title–DOI pairs from `filtered_records_on_zeolite.json`)

**Retrieval methods compared:**
- FAISS (semantic similarity search)
- BM25 (keyword-based search)
- Hybrid (BM25 + MMR combined)

**K values evaluated:** Top-1, Top-5, Top-10, Top-15, Top-20

**Outputs:**
- Grouped bar charts comparing all 3 methods at each k, for both n=100 and n=200
- Retrieval accuracy (% of queries where the correct DOI appears in the top-k results)

---
> ⚠️ **Historical — superseded by 21b:**
>
> **Notebook 21** — Original retrieval benchmark. FAISS only, smaller test set (`enriched_records`), also includes exploratory RAG query cells using Llama. Superseded by 21b which adds BM25/Hybrid comparison and larger n.

---

## Generation / MCQ Benchmarking

### Notebook 23d — Generation Quality: Llama, 7 Retrieval Methods, k=10 & k=20

Evaluates **Llama-3-8B-Instruct** on MCQs about zeolite synthesis from `questionbank(unedited).xlsx` (Synthesis questions only).

**Model:** Llama-3-8B-Instruct (GPU 3)

**Retrieval methods (7 total, k=10):**
Abstract baseline, No context, FAISS, MMR, BM25, Hybrid, LLM-Optimized

**Structure:**
- Phase 1: Abstract, No Context, FAISS
- Phase 2: MMR, BM25, Hybrid, LLM-Optimized
- Phase 3: k=20 for all retrieval methods (per-question GPU cache clearing to avoid OOM)

**Notable feature:** Includes an **LLM-Optimized** retrieval method — Llama rewrites the MCQ into a keyword search query before retrieval. Also compares k=10 vs k=20 with regression analysis (questions correct at k=10 but wrong at k=20).

**Outputs:** Bar charts, combined k=10 vs k=20 comparison chart, regression analysis, CSV export. Run with kernel v6.

### Notebook 23b — Generation Quality: Mistral vs Llama, Multi-K

Compares **two models** at **two k values** side by side on Synthesis questions from `questionbank(unedited).xlsx`.

**Models compared:**
- Mistral-7B-Instruct-v0.1 (GPU 3)
- Llama-3-8B-Instruct (GPU 2)

**Retrieval methods (7 total):** Abstract baseline, No context, FAISS, MMR, BM25, Hybrid, LLM-Optimized

**K values:** k=10 and k=20 (per-question GPU cache clearing for k=20)

**Outputs:** Grouped bar charts comparing both models, combined k=10 vs k=20 results, CSV export.

### Notebook 23e — Generation Quality: 90 Questions, 3 Categories, Llama + GPT-4.1 ⭐ *Primary generation benchmark*

Evaluates LLM performance on the full **QuestionBank_GreenandNew_90.xlsx** (90 MCQs across Synthesis, Chem Con & Catalysis, and Envir. Apps/DAC).

**Models tested:**
- Llama-3-8B-Instruct (GPU 3) — all retrieval methods
- GPT-4.1 (OpenAI API) — no-context baseline only

**Retrieval methods (at k=10 and k=15):**
Abstract baseline, No context (Llama), No context (GPT-4.1), FAISS, MMR, BM25, Hybrid

**Structure:**
- Phase 1: Abstract, No Context (Llama + GPT-4.1), FAISS
- Phase 2: MMR, BM25, Hybrid
- Phase 3: k=15 for all retrieval methods

Accuracy reported **per category** and **overall**. Results saved as JSON and CSV. All plots saved as **SVG**. Run with kernel v6.

### Notebook 23f — Information Gain: Retriever Accuracy vs k (Synthesis Only)

Measures how retriever accuracy changes as k increases, using the **30 Synthesis questions** from `QuestionBank_GreenandNew_90.xlsx`.

**Baselines (k-independent):** Abstract, No context (Llama), No context (GPT-4.1)

**Retrievers swept:** FAISS, MMR, BM25, Hybrid

**k values:** 5, 10, 15, 20, 25, 30

**Outputs:** Information gain line chart (accuracy vs k, baselines as horizontal lines), results table, CSV, OOM/INVALID/ERROR summary. All plots saved as **SVG**.

### Notebook 23k — Open-Ended Recovery of the 7 `_COT` LoRA-adapter models

Companion to 23j. The 7 `_COT` fine-tunes are **LoRA/PEFT adapters** (r=8, α=16, `q_proj`+`v_proj`) — not full HF models — so 23j's `AutoModelForCausalLM.from_pretrained` couldn't load them. 23k runs on the new `mrigi_tor190_v9` kernel (v8 + `peft<0.4`) and uses `PeftModel.from_pretrained(base, adapter)`: for each adapter it reads `adapter_config.json` → discovers `base_model_name_or_path` → loads the base from the local HF cache (`local_files_only=True`) → applies the adapter. All 7 base models are already cached, so no downloads.

Uses `zeolite_openended_100.xlsx` with **Q67 swapped** (original OOM'd on every model at mmr; backup at `zeolite_openended_100_pre_q67swap.xlsx`). Merges into `results_23j_checkpoint.json` so downstream 30b sees a single unified 14-model file. Same early-abort / per-method-checkpoint / progress-log tracking as patched 23j.

### Notebook 23j — Generation Quality: OPEN-ENDED, 14 Models, 100 Questions ⭐ *Open-ended companion to 23i*

Sister notebook to 23i (MCQ), but the model sees **only the question stem** — no A/B/C/D/E options and no letter-format instructions. Answers are stored as free-form prose and graded downstream by **notebook 30b** (GPT-4.1 judge).

**Question set:** `zeolite_openended_100.xlsx` (100 questions) — built from `zeolite_mcq_selected_combined__modified_beste.xlsx`: 86 kept rows from "Selected MCQs" (14 red-flagged rows dropped) + first 14 rows from the "replacements" tab. A `correct_answer_text` column is added (text of the correct option) as the gold answer used by the judge.

**Models tested (14):** Same as 23i — Llama-3-8B-Instruct + 13 fine-tunes on `cuda:0` (`DAPT_LR1e5[_COT]`, `synv2V2_step80[_COT]`, `synv2V2_final[_COT]`, `synv2_base_step80[_COT]`, `synv2_base_final[_COT]`, `fullpaper_120M[_COT]`, `base_llama_COT`).

**Retrieval methods:** `no_context`, `mmr` (k=15). `max_new_tokens=400` (vs 100 for MCQ in 23i).

**Outputs:** `results_23j_checkpoint.json` + `results_23j_complete_<ts>.json` + `results_23j_<ts>.csv` (per-model valid-response rate — real quality scores come from 30b).

### Notebook 23i — Generation Quality: MCQ, 14 Models, 100 Questions ⭐ *Direct MCQ counterpart to 23j*

Same 14-model, single-GPU-swap sweep as 23j but in **MCQ format** using `zeolite_mcq_selected_combined_v2.xlsx`. Uses the strict `Answer: [LETTER] - [One sentence explanation]` prompt from earlier 23-series notebooks. Retrieval methods: `no_context` + `mmr` (k=15). Outputs `results_23h_checkpoint.json` / `results_23h_complete_<ts>.json` (23h-prefixed filenames are legacy — this is the 23i notebook). Graded by notebook 30a.

### Notebook 23g — Generation Quality: ZeoDapModelLR1e5 (Fine-Tuned Zeolite Model)

Evaluates the fine-tuned zeolite domain model **ZeoDapModelLR1e5** (`aleynabeste/ZeoDapModelLR1e5`) on the **30 Synthesis questions** from `QuestionBank_GreenandNew_90.xlsx`, compared against a GPT-4.1 no-context baseline.

**Models tested:**
- ZeoDapModelLR1e5 (GPU 3) — all retrieval methods
- GPT-4.1 (OpenAI API) — no-context baseline only

**Retrieval methods (at k=10 and k=15):**
Abstract baseline, No context (ZeoDap), No context (GPT-4.1), FAISS, MMR, BM25, Hybrid

Accuracy reported for Synthesis category only. All plots saved as **SVG**. Run with kernel v6.

---
> ⚠️ **Historical — superseded by 23d/23e:**
>
> **Notebook 23** — Earliest generation notebook. Mistral only, FAISS only, no structured phases. Two modes: open-ended RAG QA with a self-judging Mistral LLM, and MCQ with single-letter extraction. Precursor to the LLM-judge pattern used in notebook 30.
>
> **Notebook 23a** — First structured MCQ benchmark. Mistral only, k=10 only, `questionbank(unedited).xlsx` Synthesis sheet. No unified `evaluate_method` function. Superseded by 23d (Llama) and 23e (multi-category, better question bank).

---

## LLM-as-Judge Evaluation

### Notebook 30b — Dual-Judge (GPT-4.1 + GPT-4o) for 23j+23k (Open-Ended) ⭐

Grades the open-ended answers from the merged 14-model file (23j + 23k) using **two judges in parallel** — `gpt-4.1` and `gpt-4o` — so both scores can be compared. Input: `results_23j_checkpoint.json` (after 23k merges).

**Four scoring tasks per judge (each 1–10 with reasoning):**
1. **Correctness** — free-text answer vs `correct_answer_text` gold answer (judged on scientific substance, not wording)
2. **Completeness** — how fully the response addresses the question
3. **Context Relevance** — how relevant the retrieved MMR context was (skipped for `no_context`)
4. **Faithfulness** — hallucination check: does the response avoid inventing values, citations, or claims not supported by context or standard zeolite science

**Judge comparison views added:** Pearson correlation, mean absolute score difference, and % agreement within ±1 point per (model, method, task); scatter plot of gpt-4.1 vs gpt-4o scores pooled across all models.

Storage schema is nested: each per-question record has a `judges` sub-dict keyed by judge model name, so downstream (30c) can dispatch on `schema == 'dual-judge-v1'`.

`ThreadPoolExecutor(max_workers=8)` parallelism across questions; judges called sequentially within a question so we don't multiply request rate. Per-model intermediate saves.

**Outputs:** `judge_results_23j_dual_intermediate_<ts>.json`, `judge_results_23j_dual_final_<ts>.json`, `judge_summary_23j_dual_<ts>.csv`, `judge_agreement_23j_dual_<ts>.csv`, plus per-task grouped bar charts (method × judge), per-(judge, method) heatmaps, and gpt-4.1-vs-gpt-4o scatter plot as SVG.

### Notebook 30e — Paper figures from the dual-judge evaluation ⭐ *Read-only, no API calls*

Turns `judge_results_23j_dual_final_20260802_184737.json` into publication figures, labelled with the **Table 1** model names. Covers **9 of 10 Table 1 rows** — GPT-5.2 was never run through the generation pipeline and is excluded from every figure.

| Figure | Content |
|---|---|
| 1 | Correctness by model × retrieval condition (horizontal bars, ±1 SEM) |
| 2 | All four judge metrics as small multiples |
| 3 | CoT effect per base checkpoint (dumbbell: base → +CoT FT) |
| 4 | Paired CoT deltas by metric, with Wilcoxon significance |
| 5 | Judge agreement, GPT-4.1 vs GPT-4o (model-level r and per-question ±1 agreement) |

Colours come unchanged from the validated data-viz reference palette (categorical slots 1–2 for the two retrieval conditions). Every figure is written as both `.svg` (for LaTeX) and 300-dpi `.png` into `figures_30e/`, alongside `paper_table_<ts>.csv`/`.tex` (mean ± SEM per model) and `cot_paired_deltas_<ts>.csv`.

### Notebook 30d — Aggregate an existing single-judge `judge_results_final_*.json` (no re-judging)

Consumes an existing GPT-4.1 single-judge JSON produced by notebook **30** (the 23e-consumer: single model = Llama-3-8B-Instruct, k × category × method × question, tasks = relevance / correctness / completeness — no faithfulness, no second judge). Runs the "aggregate → CSV → visualizations → final printout" flow from 30b's `## Aggregate to a summary DataFrame + CSV` section onward, but adapted to this file's schema (`results[k_label][category][method] → [per-question dicts]`).

Default input: `judge_results_final_20260306_162420.json` (override `JUDGE_FILE` at top of the file). Auto-discovers which tasks are present, so it also works on older files that lack faithfulness.

**Outputs:** `judge_summary_30d_<ts>.csv` (one row per k × category × method with mean/n per task) + per-`(task, k)` bar charts and heatmaps (methods × categories) as SVG.

### Notebook 30c — FT-vs-Base Analysis (per-FT wins/losses + topic autocluster) ⭐

Consumes `judge_results_23j_dual_final_*.json`. For each FT model separately (per user request — not just the aggregate):

1. Computes per-question `correctness_score(FT) − correctness_score(Llama-3-8B-Instruct)` for each (method, judge).
2. Ranks and reports the top-10 wins (FT beats base) and top-10 losses (FT worse than base) per FT model per (method, judge).
3. Uses GPT-4.1 to cluster the 100 questions into 6–8 zeolite-science topics (returns labels + assignments as JSON), then per-topic FT-vs-base mean correctness delta heatmaps (rows = FT models, cols = topics), one heatmap per (method, judge).

**Outputs:** `ft_vs_base_win_loss_<ts>.csv`, `topic_assignments_<ts>.csv`, `ft_vs_base_by_topic_<ts>.csv`, `ft_vs_base_overall_<ts>.csv`, and diverging-colormap topic heatmaps (`ft_vs_base_topic_heatmap_*_<ts>.svg`).

### Notebook 30a — LLM-as-Judge for 23i (MCQ)

Same pipeline as 30b but for MCQ outputs: 3 tasks (relevance, correctness, completeness — no faithfulness), correctness prompt compares model's letter answer against the correct letter. Input: `results_23h_checkpoint.json`. Outputs: `judge_results_23i_final_<ts>.json` + `judge_summary_23i_<ts>.csv`.

### Notebook 30 — LLM-as-Judge: GPT-4.1 Evaluation of RAG Generation Quality ⭐

Uses **GPT-4.1 as a judge** to evaluate the quality of Llama-3-8B-Instruct responses from notebook **23e**, going beyond MCQ accuracy to assess reasoning and retrieval quality.

**Input:** `results_23e_complete_*.json` (per-question `query`, `context`, `model_answer`, `correct_answer`, `full_response`, `explanation`)

**Three evaluation tasks (each scored 1–10 with reasoning):**
1. **Context Relevance** — How relevant is the retrieved context to the query?
2. **Response Correctness** — How correct is the model's answer given the context?
3. **Response Completeness** — How fully does the response address the question?

**Coverage:**
- 6 methods: abstract, no_context, FAISS, MMR, BM25, Hybrid
- 3 categories: Synthesis (30), ChemCon (30), DAC (30) = 90 questions
- k=10 and k=15; Context Relevance skipped for `no_context`
- ERROR/INVALID responses recorded with score -1

**Outputs:**
- Per-question JSON scores with reasoning (`judge_results_final_*.json`)
- Summary CSV (`judge_summary_*.csv`)
- Per-category grouped bar charts, k=10 vs k=15 comparison bar charts (PNG)
- Heatmap: methods × categories for each task (PNG)
- Radar/spider chart: overall quality profile per method (PNG)
- Box plots: score distributions per method (PNG)
- Scatter plots: MCQ accuracy vs judge scores (PNG)

---

## ZeoSyn Synthesis Generation

These notebooks generate synthesis descriptions for zeolite frameworks in the ZeoSyn database by querying the RAG pipeline once per framework, rather than evaluating against a fixed question bank.

### Notebook 27 — ZeoSyn Synthesis Generation (MMR)

Generates synthesis descriptions using **MMR retrieval** (k=5, fetch_k=25, λ=0.5) and Mistral-7B-Instruct-v0.1. Intermediate step between notebook 26 (FAISS only) and 29 (Hybrid).

**Model:** Mistral-7B-Instruct-v0.1 (GPU 3)

**Retrieval:** MMR (diversity-aware semantic search, k=5)

**Dataset:** `ZEOSYN.csv`

**Query template (structured):** `"Extract the following synthesis information for {code} or {code}-like frameworks: [crystallization temperature], [crystallization time], [precursor1, precursor2, ..], [OSDA1, OSDA2, etc]"`

**Key differences:** Uses MMR instead of plain FAISS (26); uses single retriever rather than hybrid BM25+MMR (29). Introduces the structured extraction query format carried forward into 29.

**Outputs:** `framework_synthesis_responses_<timestamp>.json`, `outputs/clean_code1_doi_mapping_<timestamp>.json`.

### Notebook 29 — ZeoSyn Synthesis Generation (Hybrid BM25+MMR) ⭐ *Best ZeoSyn generation*

Generates synthesis descriptions using **BM25 + MMR hybrid retrieval** — the most complete retrieval strategy in the ZeoSyn generation track.

**Model:** Mistral-7B-Instruct-v0.1 (GPU 3)

**Retrieval:** Hybrid — BM25 (k=5) + MMR (k=5, fetch_k=40, λ=0.6) → 10 deduplicated docs

**Dataset:** `ZEOSYN.csv`

Same structured extraction query and incremental-save pipeline as notebook 27, but with hybrid retrieval combining keyword matching (BM25) and semantic diversity (MMR).

**Outputs:** `framework_synthesis_responses_<timestamp>.json`, `outputs/clean_code1_doi_mapping_<timestamp>.json`.

---

## Domain Model Exploration

### Notebook 24 — ZeoDapModelLR1e5: Exploratory Direct QA ⚠️ *Historical — precursor to 23g*

Early exploratory test of **ZeoDapModelLR1e5** (`aleynabeste/ZeoDapModelLR1e5`) with its CoT adapter (`ZeoDapModelLR1e5COT`) answering open-ended zeolite questions directly — no RAG, no MCQ benchmark.

**Model:** ZeoDapModelLR1e5 + COT adapter (GPU 1), max_new_tokens=250, temperature=0.01

Loads the fine-tuned model, wraps it in a `HuggingFacePipeline`, and calls `answer_llama("what are zeolites")` as a sanity check. Establishes that ZeoDap can be loaded and queried before integrating into the full RAG benchmarking pipeline in notebook 23g.

---

## Historical / Early Prototypes

> The notebook below is retained for lineage only — fully superseded by later work.

### Notebook 17 — Single-Query RAG with LLM-Judge (Prototype) ⚠️ *Superseded by 30*

Early prototype answering a **single hardcoded query** (`"ABW framework synthesis"`) using Mistral-7B-Instruct-v0.1 with k=2 FAISS retrieval. The same Mistral model acts as a 4-criterion judge (relevance, accuracy, completeness, citation, each 0–10). The `evaluate_rag_response()` function defined here is the direct prototype for the GPT-4.1 judge in notebook 30. No MCQ, no question bank, no looped benchmarking.
