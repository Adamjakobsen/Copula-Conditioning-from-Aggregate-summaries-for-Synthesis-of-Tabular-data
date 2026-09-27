# CoCAST

**Copula-Conditioning from Aggregate summaries for Synthesis of Tabular data**

CoCAST generates psychiatric questionnaire responses from cohort summaries and
DSM-5 context. A Gaussian copula samples joint severity profiles. A pretrained
language model then generates responses to the seven questionnaires. The profile
sampler and language model require no participant-level records.

This repository contains the generation and numerical evaluation procedures for
the paper and supplement. The implementation covers 69 items across seven
specified domains. Adapting it to another questionnaire requires corresponding
schema, scoring and prompt resources.

## Installation

Use Python 3.12 or 3.13 in an isolated environment. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[evaluation,source,tokenizer]' -r requirements.txt
python -m unittest discover -s tests -q
cocast --help
```

`requirements.txt` pins the numerical evaluation stack. To train the
record-based comparators, install the baseline dependencies in this environment
or a separate one containing CoCAST:

```bash
python -m pip install -e '.[baselines]'
```

The paper uses SDV 1.34.1, CTGAN 0.12.1, RDT 1.20.0 and copulas 0.14.1, with
500 training epochs for each synthesizer. Install a CUDA-compatible PyTorch
build when using a GPU. Evaluation runs on CPU.

For generation, use a CUDA environment with **vLLM 0.29.0**, its compatible
PyTorch/Transformers dependencies, and CoCAST:

```bash
python -m pip install 'vllm==0.29.0'
python -m pip install -e .
python -c "import torch, vllm; print(vllm.__version__); print(torch.cuda.get_device_name())"
```

The paper's original BF16 27B runs used an A100 80 GB. The copula-ablation
runs used an H200 NVL with the same checkpoint and generation settings. The
smaller models need less memory. Managed generation downloads pinned
checkpoints to the Hugging Face cache, starts a localhost server and stops only
that server when its queue ends. 

## Inputs and data access

The supplied `src/cocast/resources/aggregate_example.yaml` contains summaries of
the paper's 564-participant baseline cohort, including two missing ages. It contains cohort-computed probit correlations,
severity-tier prevalences, sex proportions, an age histogram and missing-age
probability. 

Participant records are needed only to extract summaries, train CTGAN/TVAE, or
evaluate against the real cohort. Obtain the study data from the
[source study's OSF repository](https://osf.io/jz4ge/) subject to its access terms.
The project's public metadata did not declare a licence when checked on
27 September 2026. The real reference data are not redistributed here and
are not covered by the CoCAST licences. See
[reference-data provenance](dataset/REFERENCE_DATA.md).
Place the source file outside the package, for example in `private/`:

```bash
cocast summarize --real private/data_set_final_osf.sav \
  --output work/aggregates.yaml --export-real private/real_baseline.csv
```

The SPSS reader selects the first-wave columns. CSV inputs must contain
`W1_age_r`, `W1_sex_r`, nine `W1_depression_it1` through `W1_depression_it9`
columns, and ten `W1_<domain>_it1` through `W1_<domain>_it10` columns for each of:
`separation_anxiety`, `specific_phobia`, `social_anxiety`, `panic`,
`agoraphobia`, and `generalized_anxiety`. Sex is coded 1 or 2. Depression items
are integers 0–3, anxiety items 0–4. Missing age is retained. 

For the paper reproduction, use the included aggregate file and all 564 baseline
records for reference evaluation. For a different cohort, change the aggregate
path and counts in a copy of `configs/paper.yaml`. Relative configuration paths
resolve against that configuration file. `@aggregate_example` selects the
bundled aggregate resource. DSM-5 source attribution and page references are
retained in `knowledge_graph.json`.




## Reproduce the study

`configs/paper.yaml` is the complete experiment configuration. It includes
36 matched-size datasets and six expanded pools across seeds 42, 43 and 44,
including CTGAN and TVAE. Each matched-size dataset has 564 records. 

| Condition | Content | Default models |
|---|---|---|
| `natural` | Own questionnaire's score and DSM-5 context | 4B, 9B, 27B |
| `full` | All seven scores and relevant DSM-5 context | 4B, 9B, 27B |
| `balanced` | Own score, targeted tier probabilities and DSM-5 context | 4B, 9B, 27B |
| `independent` | Copula ablation: independently sampled severity scores and DSM-5 context | 27B |
| `expanded_natural` | Natural pool extended to 2,260 records | 27B |
| `expanded_balanced` | Balanced pool extended to 2,260 records | 27B |

Balanced generation requests 20% per tier for depression, specific phobia,
social anxiety, panic and generalized anxiety. Separation anxiety and agoraphobia
retain natural prevalences.

The copula ablation samples the seven severity targets independently, using
the same tier probabilities and uniform attainable scores within each tier.
It retains paired demographic draws, questionnaire-specific targets, DSM-5
context and decoding settings. It removes cross-disorder dependence from target
sampling. Realised tier counts may differ through sampling variation. The
paper includes this ablation for 27B-Qwen3.5.

Run the stages separately:

```bash
cocast study prepare --config configs/paper.yaml --out work/paper
cocast study generate --config configs/paper.yaml --out work/paper --port 8000
cocast study baselines --config configs/paper.yaml --out work/paper \
  --real private/real_baseline.csv
cocast study evaluate --config configs/paper.yaml --out work/paper \
  --real private/real_baseline.csv
```

Run generation in the CUDA environment, baselines in the SDV environment, and
evaluation in the pinned numerical environment. All stages use the same output
root. Baseline and study evaluation commands verify that the supplied real
cohort reproduces the configured summaries. Generation never trains the baselines or runs evaluation. The generation
runner keeps each model loaded across its selected conditions and seeds.
Expanded conditions first generate or validate their matched-size parent run,
then reuse its completed 564-record prefix.

To run one seed or model, use selectors consistently across stages:

```bash
cocast study generate --out work/seed42 --models 27b --seeds 42 \
  --conditions natural balanced expanded_natural expanded_balanced --port 8018
cocast study generate --out work/ablation --models 27b --seeds 42 43 44 \
  --conditions independent --port 8019
```

For all three ablation sizes, pass `--models 4b 9b 27b`. Independent terminals
must use different output roots. Use different ports if they share a node, and
ensure each server has its own allocated GPU. The runner does not change
`CUDA_VISIBLE_DEVICES`. Use `--tensor-parallel-size 2` only when both GPUs are
visible within the same allocation. `--existing-server` skips server management
and requires one selected model, the matching port, and the configured checkpoint
and generation settings on that server.

Rerunning the generation command validates completed runs and skips them.
Interrupted runs resume from their journals without regenerating accepted
responses. Changing inputs or scientific settings requires a different output
root. Invalid responses are retried up to the configured limit. An incomplete
run fails explicitly and is not eligible for evaluation. After inspecting the
invalid answers, `study generate --extra-attempts 2` permits two additional
attempts for unanswered requests in existing runs. The allowance is recorded
in the manifest. It never regenerates accepted answers. The equivalent stage
command is `generate --resume --extra-attempts 2`. Use ordinary resume without
this flag to continue an interrupted attempt allowance.

`study evaluate` evaluates the explicitly selected conditions and baselines.
Use `--no-baselines` for an LLM-only comparison, or `--baselines ctgan` to select
one comparator. `--fidelity-only` omits classifier fitting and should be used
with natural-prevalence conditions. Main evaluations refuse to overwrite an
existing output directory. Learning-curve evaluations resume at completed domain
boundaries after checking their fingerprints.

## Individual stages and numerical results

The stage commands also accept explicit inputs independently of the study runner:

```bash
cocast baseline --real private/real_baseline.csv --method ctgan \
  --seed 42 --epochs 500 --device cuda --out work/ctgan42
cocast baseline --real private/real_baseline.csv --method tvae \
  --seed 42 --epochs 500 --device cuda --out work/tvae42
cocast evaluate --real private/real_baseline.csv \
  --run work/paper/seed42/natural_27b --run work/ctgan42 --run work/tvae42 \
  --utility --out work/comparison
cocast learning-curves --real private/real_baseline.csv \
  --run work/paper/seed42/expanded_natural_27b \
  --run work/paper/seed42/expanded_balanced_27b \
  --ratios 0.5 1 2 5 --out work/volume_comparison
```

For a custom target mix, pass `profiles --prevalences configs/target_uniform5.yaml`.
For the full-profile context ablation, pass `prepare --context full` with the
same profiles. For the copula ablation, generate profiles with
`profiles --dependence independent`, then use `prepare --context own`. The
`study` commands select these settings automatically for `--conditions independent`.

After evaluating all seeds, aggregate the saved outputs:

```bash
cocast results \
  --evaluation work/paper/evaluation/seed42 \
  --evaluation work/paper/evaluation/seed43 \
  --evaluation work/paper/evaluation/seed44 \
  --evaluation work/paper/evaluation/volume_seed42 \
  --evaluation work/paper/evaluation/volume_seed43 \
  --evaluation work/paper/evaluation/volume_seed44 \
  --out work/paper/results
```


## Evaluation protocol

Fidelity uses marginal Jensen–Shannon divergence, bias-corrected Cramer's-V error
(overall, between domains and within questionnaires), and energy distance with
square-root normalized Hamming ground distances. Proximity uses ordinary
normalized Hamming distance for DCR and NNDR, plus exact-match percentages and a
real leave-one-out reference. Age is categorised using real-age quantiles, with
missing age as a separate category. Larger distances are not necessarily better,
and these measurements do not establish privacy protection.

Utility predicts each domain's achieved severity tier from age, sex and the other
six questionnaires. Target-domain items are excluded. LR denotes multinomial
logistic regression and HGB histogram gradient boosting. Real-only training
(TRTR), synthetic-only training (TSTR), and augmentation (TAUG) all test on real
held-out records. The synthetic training count is the ceiling of the selected
ratio times the real training-fold count. TAUG adds those records to the real
training fold.





## Output files

| Stage | Files |
|---|---|
| Summary extraction | One YAML, optional reference CSV |
| Profiles | One JSON per prevalence mix, size and seed |
| Prompts | `manifest.json`, `prompts.jsonl`, shared across models |
| LLM generation | `manifest.json`, `responses.jsonl`, `dataset.csv` |
| CTGAN/TVAE | `manifest.json`, `dataset.csv` |
| Evaluation | `manifest.json`, `metrics.csv`, optional `utility_folds.csv` |
| Result aggregation | `manifest.json`, `summary.csv`, `targeting_changes.csv` |



## Released synthetic datasets

`dataset/` contains all 42 synthetic CSVs used in the study, with a dataset
index, column dictionary and checksums. See `dataset/README.md` for scoring,
provenance and the distinction between matched-size datasets and expanded
pools. The three copula-ablation files are labelled `copula-ablation` and use
independent severity sampling. The index records `dependence=independent`
so they cannot be confused with the natural copula-conditioned datasets.
No real participant table is included.

Expanded pools retain their corresponding 564-row prefixes. Do not combine
a pool with its parent as though they were independent datasets. The release
contains the evaluated responses unchanged, including any duplicate response
patterns. The generation and evaluation commands above reproduce the complete
pipeline from the supplied aggregate inputs and an independently obtained real
reference table. Regeneration may vary across hardware or inference libraries.

## Licences and attribution

The original CoCAST Python code is provided under the [MIT License](LICENSE).
The 42 synthetic CSVs and CoCAST-authored dataset documentation in `dataset/`
are provided under [CC BY 4.0](dataset/LICENSE.txt). Research and commercial
reuse, modification and redistribution are permitted with attribution, a link
to the licence and an indication of changes. See `dataset/README.md` for
attribution details.

These grants apply only to rights held by the CoCAST contributors. The real
reference dataset and third-party DSM-5 descriptions, diagnostic criteria and
questionnaire wording are not relicensed by CoCAST. Their separate rights and
sources are described in [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
