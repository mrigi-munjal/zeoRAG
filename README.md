# zeoRAG

Retrieval-Augmented Generation benchmarking for zeolite synthesis literature.

**Kernels used in this repo:**
- **`mrigi_tor190_v8`** — default for everything (torch 1.12.1).
- **`mrigi_tor190_v9`** — used **only** by notebook **23k** (v8 clone + `peft<0.4` for loading LoRA adapters).

Every notebook that needs a specific kernel has a plain-text banner at the top telling you which one to select. See [zeoRAG/CLAUDE.md](CLAUDE.md) for env details.

**Key notebooks:**
- **21b** — retrieval-only benchmarking (title → DOI, FAISS vs BM25 vs Hybrid)
- **23e** — generation benchmarking (MCQ, 90 questions, Llama + GPT-4.1, multi-category)
- **23i / 23j / 23k** — multi-model 100-question benchmarks: **23i** = MCQ, **23j** = open-ended, **23k** = recovery run for the 7 `_COT` models that failed to load in 23j
- **30** — LLM-as-judge quality evaluation of 23e outputs
- **30a** — LLM-as-judge for 23i (MCQ)
- **30b** — **dual-judge** (GPT-4.1 + GPT-4o) for 23j+23k open-ended answers, with judge-agreement views
- **30c** — FT-vs-base analysis: per-FT-model win/loss lists vs Llama base + GPT-4.1 autoclustered topic breakdown
- **30d** — aggregate/visualize an existing single-judge results file without re-judging
- **30e** — ⭐ **paper figures** from the dual-judge results (Table 1 labels, read-only, no API calls)
- **30f** — GPT-5.2 no-context generation + judging (SI reference point only)

**Jump to:** [Methods](#methods-open-ended-evaluation) · [Results](#results-open-ended-dual-judge-evaluation)

---

# Methods: Open-Ended Evaluation

> Everything in this section describes the **open-ended** track: generation in **23j** (+ **23k** for the LoRA-adapter variants), judging in **30b**, figures in **30e**. The older MCQ track (23e/23i → 30/30a) is documented in the notebook index further down.

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
- **Q67 was subsequently swapped** — the original item (Pd/ZSM-22 hydroisomerization) produced a context block that triggered CUDA OOM on every model at k=15; it was replaced with the previously-unused 15th replacement item (Al organization in SSZ-13). Pre-swap file preserved as `zeolite_openended_100_pre_q67swap.xlsx`.
- **Conversion to open-ended:** only the question stem is presented to the model. The A–E answer options are discarded; the text of the correct option becomes the **reference (gold) answer** supplied to the judge.

Resulting set (`zeolite_openended_100.xlsx`): 100 questions, **99 unique DOIs**, question length mean 230 chars (127–367), gold answer length mean 174 chars.

## 3. Models evaluated

All models are Llama-3-8B-Instruct derivatives, evaluated in fp16 on a single GPU with sequential load/unload.

| Family | Base variant | +COT variant |
|---|---|---|
| Reference | `Llama-3-8B-Instruct` (stock) | `base_llama_COT` |
| DAPT | `DAPT_LR1e5` | `DAPT_LR1e5_COT` |
| synv2V2 (step 80) | `synv2V2_step80` | `synv2V2_step80_COT` |
| synv2V2 (final) | `synv2V2_final` | `synv2V2_final_COT` |
| synv2 base (step 80) | `synv2_base_step80` | `synv2_base_step80_COT` |
| synv2 base (final) | `synv2_base_final` | `synv2_base_final_COT` |
| Full-paper 120M | `fullpaper_120M_LR1e5` | `fullpaper_120M_COT` |

**Reported in the paper.** Of the 14 variants above, the 9 that appear in Table 1 (and
therefore in all Results figures) are: `Llama-3-8B-Instruct`, `base_llama_COT`,
`DAPT_LR1e5`, `DAPT_LR1e5_COT`, `synv2V2_step80`, `synv2V2_step80_COT`,
`synv2_base_step80_COT`, `fullpaper_120M_LR1e5`, `fullpaper_120M_COT`. The `synv2V2_final`,
`synv2_base_final` and `synv2_base_step80` checkpoints were judged but are not part of
Table 1. GPT-5.2 (Table 1 row 1) has no generation results and is absent throughout.

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

**Source:** `judge_results_23j_dual_final_20260802_184737.json` · **Figures:** notebook **30e** → `figures_30e/`
**Primary judge: `gpt-4o`.** `gpt-4.1` scored every response as well; where the two disagree
it is stated explicitly — see [Judge sensitivity](#judge-sensitivity--read-before-citing-findings-2-and-3).

**Scope.** 9 of the 10 Table 1 variants, 100 open-ended questions, 2 retrieval conditions.
GPT-5.2 (row 1) has no generation results (notebook **30f** is built and ready to fill this gap).

**Metrics reported.** Correctness and completeness only — see
[Why only two metrics](#why-only-two-metrics-are-reported).

---

## Main results

Mean ± SEM, judge `gpt-4o`. Indented rows are the `+ CoT FT` variant of the row above.

| Model (Table 1) | Correctness<br>no context | Correctness<br>RAG (MMR k=15) | Completeness<br>no context | Completeness<br>RAG |
|---|---|---|---|---|
| Llama-3-8B-Instruct | 7.15 ± 0.16 | 7.56 ± 0.15 | 8.68 ± 0.05 | 8.22 ± 0.09 |
| &nbsp;&nbsp;+ CoT FT | 7.58 ± 0.13 | 7.73 ± 0.15 | 8.97 ± 0.02 | 8.80 ± 0.04 |
| Broad zeolite abstracts DAPT | 6.96 ± 0.18 | 6.65 ± 0.18 | 8.54 ± 0.06 | 7.21 ± 0.15 |
| &nbsp;&nbsp;+ CoT FT | 7.30 ± 0.17 | 7.76 ± 0.15 | 8.83 ± 0.07 | 8.70 ± 0.06 |
| Synthesis abstracts DAPT | 7.05 ± 0.17 | 7.01 ± 0.16 | 8.56 ± 0.06 | 7.71 ± 0.13 |
| &nbsp;&nbsp;+ CoT FT | 7.44 ± 0.14 | **7.83 ± 0.13** | 8.95 ± 0.03 | 8.77 ± 0.05 |
| Synthesis abstracts DAPT (base LM) + CoT FT | 4.92 ± 0.33 | 7.51 ± 0.17 | 5.77 ± 0.39 | 8.30 ± 0.13 |
| Synthesis full-paper DAPT | 6.74 ± 0.18 | 6.74 ± 0.17 | 8.45 ± 0.07 | 7.27 ± 0.15 |
| &nbsp;&nbsp;+ CoT FT | 7.31 ± 0.16 | 7.79 ± 0.15 | 8.96 ± 0.03 | 8.70 ± 0.07 |

Best configuration under `gpt-4o`: **Synthesis abstracts DAPT + CoT FT under RAG, 7.83**,
followed by full-paper DAPT + CoT (7.79) and broad abstracts DAPT + CoT (7.76). All three
top slots are `DAPT + CoT` combinations, and all seven `+ CoT FT` rows outrank every
non-CoT row under retrieval.

---

## Finding 1 — CoT fine-tuning is the dominant effect, and retrieval amplifies it

| Pair | Correctness<br>no context | Correctness<br>RAG | Completeness<br>no context | Completeness<br>RAG |
|---|---|---|---|---|
| Llama-3-8B-Instruct (no DAPT) | +0.41 \* | +0.17 n.s. | +0.29 \*\*\* | +0.63 \*\*\* |
| Broad zeolite abstracts DAPT | +0.34 n.s. | **+1.14** \*\*\* | +0.29 \*\*\* | **+1.49** \*\*\* |
| Synthesis abstracts DAPT | +0.39 \* | **+0.85** \*\*\* | +0.39 \*\*\* | **+1.07** \*\*\* |
| Synthesis full-paper DAPT | +0.58 \*\* | **+1.01** \*\*\* | +0.52 \*\*\* | **+1.47** \*\*\* |

Pooled over the 4 pairs:

| Metric | Condition | n | Mean Δ | Cohen's d | p |
|---|---|---|---|---|---|
| Correctness | no context | 387 | +0.43 | 0.23 | 7.6 × 10⁻⁶ |
| Correctness | **RAG** | 368 | **+0.81** | **0.54** | 3.1 × 10⁻²⁰ |
| Completeness | no context | 396 | +0.37 | 0.49 | 1.4 × 10⁻²⁰ |
| Completeness | **RAG** | 376 | **+1.17** | **0.90** | 1.4 × 10⁻⁴² |

**The CoT advantage roughly doubles once retrieval is available** (d = 0.23 → 0.54
correctness; 0.49 → 0.90 completeness). This is the one finding that replicates cleanly
under both judges. The three **DAPT** pairs all gain strongly under RAG (+0.85 to +1.14);
the **no-DAPT** pair does not (+0.17, n.s.) — see Finding 3.

## Finding 2 — DAPT-only models fail to benefit from retrieval

Change in correctness when MMR context is added (`gpt-4o`):

| Model | no context → RAG | Δ |
|---|---|---|
| Llama-3-8B-Instruct (no DAPT) | 7.15 → 7.56 | **+0.41** |
| Broad zeolite abstracts DAPT | 6.96 → 6.65 | **−0.31** |
| Synthesis abstracts DAPT | 7.05 → 7.01 | **−0.04** |
| Synthesis full-paper DAPT | 6.74 → 6.74 | **−0.00** |
| *all four* **+ CoT FT** | | **+0.15 … +0.48** |

Every **DAPT-only** checkpoint gains essentially nothing from retrieval (−0.31 to 0.00),
while **every CoT variant** gains, and stock Llama-3-8B-Instruct — which had no DAPT —
gains the most of any non-CoT model (+0.41). Completeness is starker: the DAPT-only models
*drop* by 1.2–1.3 points when context is added.

Reading: DAPT on raw zeolite text appears to erode instruction-following, so a long
retrieved passage stops functioning as evidence. CoT fine-tuning restores it. **DAPT only
pays off when paired with CoT FT.**

> **Judge-dependent.** Under `gpt-4.1` this effect is stronger and uniformly negative
> (−0.44, −0.19, −0.22). Under `gpt-4o` two of the three are ≈ 0 rather than negative. The
> safe claim is "**DAPT-only models fail to benefit from retrieval**"; the stronger claim
> "retrieval actively *harms* them" holds only under `gpt-4.1`.

## Finding 3 — CoT and DAPT interact; neither dominates alone

`Llama-3-8B-Instruct + CoT FT` (CoT on the stock model, no zeolite DAPT) reaches 7.73 under
RAG — **4th** of nine, behind all three `DAPT + CoT` combinations (7.76–7.83). It is also
the only pair whose CoT gain under RAG is not significant (+0.17, n.s.), because its base
was already the strongest non-CoT model.

So on this benchmark the best recipe is **DAPT *and* CoT together**: DAPT alone
underperforms stock Llama, CoT alone lands mid-pack, and the combination takes the top
three slots.

> **Judge-dependent — this finding reverses.** Under `gpt-4.1`,
> `Llama-3-8B-Instruct + CoT FT` is the **best model overall** (8.45, ahead of every
> DAPT+CoT variant), supporting the opposite conclusion — that CoT alone suffices and the
> domain corpora add nothing. The two judges disagree on the ordering of the top four
> models, which are separated by only ~0.1 points. **Do not make a strong claim in either
> direction from this data.**

## Finding 4 — Instruction tuning is a prerequisite

The one variant built on the **base** (non-instruct) LM,
`Synthesis abstracts DAPT (base LM) + CoT FT`, is far the weakest without context
(**4.92** vs 7.30–7.58 for its instruct-based siblings) with 2× the standard error,
indicating erratic output. Retrieval recovers it substantially (4.92 → 7.51, **+2.59**, the
largest retrieval gain of any model) but it still trails. CoT fine-tuning does not
substitute for instruction tuning. **Replicates under both judges.**

---

## Judge sensitivity — read before citing Findings 2 and 3

Two independent judges scored every response with byte-identical prompts. Agreement is
high in aggregate but **not high enough to resolve the top of the leaderboard**.

| Metric | Pearson r (per question) | Agree within ±1 | n |
|---|---|---|---|
| Correctness | 0.880 | 92% | 1,734 |
| Completeness | 0.830 | 84% | 1,749 |

Model-level correlation on correctness is **r = 0.961**, and `gpt-4.1` scores systematically
~0.1–0.7 points higher than `gpt-4o` across the board. What survives and what does not:

| Finding | `gpt-4.1` | `gpt-4o` | Verdict |
|---|---|---|---|
| 1 — CoT helps, more so with RAG | d 0.36 → 0.73 | d 0.23 → 0.54 | **Robust** |
| 2 — DAPT-only + retrieval | −0.44, −0.19, −0.22 | −0.31, −0.04, −0.00 | **Weakened** — "no benefit" is safe, "actively harmful" is not |
| 3 — best configuration | CoT-only wins (8.45) | DAPT+CoT wins (7.83) | **Reverses** — do not claim either |
| 4 — base LM is weakest | 5.09 | 4.92 | **Robust** |

The top four models are separated by ~0.1 points, well inside the disagreement between
judges. Any claim about *which* configuration is best is not supported; claims about
*CoT vs no-CoT* and *instruct vs base LM* are.

## Supplementary — GPT-5.2 reference point (SI only, not a main-paper result)

> **Framing.** This comparison belongs in the **Supplementary Information, not the main
> paper.** The objective of this work is *not* to beat a frontier closed model. It is to
> characterise how domain-adaptive pretraining, chain-of-thought fine-tuning and retrieval
> interact in an **8B open-weight model that can be run locally on a single GPU** over a
> private literature corpus. GPT-5.2 is reported only to locate the benchmark on an
> absolute scale — i.e. to show these questions are hard but tractable — not as a
> competitive baseline. Cost, latency, data residency and reproducibility all differ by
> orders of magnitude between the two settings, so a head-to-head ranking would be a
> category error.

Notebook **30f** generated GPT-5.2 answers to the same 100 questions with the **verbatim
no-context prompt from 23j**, judged with the **same correctness prompt** as everything
else. GPT-5.2 answers from parametric knowledge only — it was given no retrieval.

| | judge `gpt-4o` | judge `gpt-4.1` |
|---|---|---|
| **GPT-5.2 (no context)** | **9.16** | **9.91** |
| Best open-weight, no context | 7.58 | 7.94 |
| Best open-weight, with RAG | 7.83 | 8.44 |

GPT-5.2 outscores **all 9 open-weight variants in both conditions**, every comparison at
p < 0.001. Under `gpt-4o` it does not lose a single question to any model (e.g. 77 wins /
0 losses / 20 ties against the best no-context CoT model; 71/0/29 against the best
RAG configuration). Margins are +1.33 to +2.52 against CoT variants and +2.1 to +2.5
against DAPT-only checkpoints.

**Two caveats that matter more than the ranking:**

1. **The benchmark is near-saturated for frontier models.** 9.16–9.91 out of 10 with
   essentially no losses means this question set no longer discriminates at that capability
   level. It discriminates well among 8B variants (4.92–7.83), which is what it was built
   for, but it cannot rank frontier models and should not be used to.
2. **Possible circularity.** The questions were LLM-generated from paper abstracts, the
   gold answers are the correct MCQ options from that same generation step, and the judges
   are OpenAI models. A frontier OpenAI model answering OpenAI-written questions graded by
   OpenAI judges may benefit from shared conventions in phrasing and emphasis that an 8B
   Llama derivative does not share. The open-weight *relative* comparisons are unaffected —
   every model faces the identical setup — but the absolute GPT-5.2 margin should be read
   with this in mind.

Artefacts: `results_gpt52_nocontext_*.json`, `judge_results_gpt52_*.json`,
`gpt52_vs_local_*.csv`, `figures_30e/fig6_gpt52_vs_local_*.svg`.

## Why only two metrics are reported

All four judge tasks were computed; two are excluded from the figures.

**Context relevance** scores the *query ↔ context* match. Every model receives identical
retrieved context from one shared MMR retriever, so the model never enters the computation.
Across all 9 models it varies by **0.098 points** — constant by construction. A useful
retriever sanity check (~7.8/10); useless for comparing models.

**Faithfulness** is the least trustworthy of the four: inter-judge agreement **r = 0.656**
vs 0.880 for correctness; under RAG it spans only 1.08 points across all 9 models; and at
**r = 0.62** with correctness it is largely redundant. Its one substantive result — CoT
variants scoring ~0.9 lower without context, suggesting ungrounded reasoning chains
confabulate — is interesting but rests on the weakest instrument and should be treated as a
hypothesis for follow-up.

## Limitations

- **GPT-5.2 is reported in the SI only**, by design — see
  [Supplementary](#supplementary--gpt-52-reference-point-si-only-not-a-main-paper-result).
  It is a scale reference, not a competitive baseline, and the benchmark is saturated at
  that capability level.
- **The primary-judge choice changes conclusions.** See
  [Judge sensitivity](#judge-sensitivity--read-before-citing-findings-2-and-3). Report both
  judges for any claim about model ordering.
- **Correctness and completeness correlate at r = 0.78** — not independent evidence.
  Completeness partly rewards thoroughness, which CoT models produce by construction, so
  part of the completeness gain reflects verbosity rather than better science.
- **Uneven judge-score coverage.** `Llama-3-8B-Instruct + CoT FT` and
  `Synthesis full-paper DAPT + CoT FT` returned usable correctness scores on 84–85% of RAG
  judge calls vs ~99% elsewhere (likely truncated JSON at `max_tokens=200`). Means use
  valid scores only and paired tests use matched questions, so comparisons hold — but this
  should be resolved before publication.
- **One retriever, one k** (MMR, k = 15) and **one question set** (100 items, no
  cross-dataset replication).

## Pending

| Item | Status |
|---|---|
| GPT-5.2 no-context generation + judging | **Done** (30f) — 9.16 / 9.91 correctness; SI only |
| Re-judge the 2 models with 84–85% coverage | Optional; would tighten two rows |
| Third judge to break the Finding-3 tie | Recommended if the "best configuration" claim matters |

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
