# CoCAST synthetic psychiatric questionnaire datasets

This collection contains the synthetic datasets used for the CoCAST study's
main comparisons, copula and context ablations, and training-size evaluations. 



## Contents

| Dataset group | Models | Seeds | Rows per CSV | CSVs |
|---|---|---|---|---|
| Natural prevalences, own-questionnaire severity context | Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B | 42, 43, 44 | 564 | 9 |
| Natural prevalences, full severity-profile context | Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B | 42, 43, 44 | 564 | 9 |
| Balanced severity mix, own-questionnaire severity context | Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B | 42, 43, 44 | 564 | 9 |
| Record-trained baselines | CTGAN, TVAE | 42, 43, 44 | 564 | 6 |
| Copula ablation, independently sampled severity targets | Qwen3.5-27B | 42, 43, 44 | 564 | 3 |
| Expanded natural and balanced pools | Qwen3.5-27B | 42, 43, 44 | 2,260 | 6 |

The `data/` directory contains only these 42 datasets.

- `dataset_index.csv` identifies each file's method, model, seed, context,
  prevalence mix, dependence model, dimensions, missing-age count, generation settings and SHA-256
  fingerprint. `condition` identifies the matched-size, expanded or ablation
  group. `extends_file` and `prefix_records` identify nested pools.
- `data_dictionary.csv` describes every column, numerical range and missing-value
  convention. It contains metadata, not participant records.
- `SHA256SUMS.txt` provides file checksums for the datasets and documentation.

## Generation conditions

Each LLM record has generated age, recorded-sex category and answers to all 69
questionnaire items. The seven domains are depression, separation anxiety,
specific phobia, social anxiety, panic, agoraphobia and generalized anxiety.

`own-severity` prompts contain the assigned score for the questionnaire being
answered. `full-severity` prompts contain the assigned scores for all seven
domains. Both include demographics and the relevant DSM-5 context.
The copula ablation uses the same marginal tier probabilities and uniform
attainable scores within each tier, but samples scores independently across
disorders. It retains paired demographic draws, own-questionnaire severity
targets and DSM-5 context. The index identifies it as `method=copula_ablation`,
`dependence=independent` and `prevalence_mix=natural`. Realised marginal counts
may differ through sampling variation.

Natural generation uses the reference cohort's aggregate tier prevalences.
Balanced generation requests 20% per tier for depression, specific phobia,
social anxiety, panic and generalized anxiety. Separation anxiety and agoraphobia
retain natural prevalences. Achieved response distributions are not guaranteed
to match these requested probabilities.

The LLM conditions use the BF16 Qwen3.5 checkpoints identified in the index,
with temperature 1, top-p 1, disabled top-k filtering, min-p 0 and disabled
thinking. The copula ablation was generated on an H200 NVL using the same
checkpoint and decoding settings as the A100 runs. CTGAN and TVAE use 500 training epochs. The associated CoCAST code
release and manuscript describe sampling, validation and evaluation in detail.

## Expanded pools and training sizes

Each expanded file's first 564 rows are identical to its indexed matched-size
parent file. These are extensions, not independent replications. Do not concatenate
an expanded file with its parent as though they were disjoint datasets.

The archive stores 33,864 rows across all files, including these repeated prefixes.
This is not a count of independent participants or unique response patterns.
IDs may also repeat across other files because they are local synthetic row IDs.

The paper's 0.5x, 1x, 2x and 5x labels denote synthetic-to-real training-fold size
ratios, not fixed whole-cohort dataset sizes. Evaluation selects nested subsets
from the pools. Fold-dependent subsets and real/synthetic resampling controls
are constructed by the evaluation code, rather than supplied as additional CSVs.

## Columns and scoring

Files retain the questionnaire schema used in the study:

- `ID`: a synthetic row identifier in LLM datasets. It is absent from CTGAN/TVAE
  datasets and is not a model predictor or a real participant identifier.
- `W1_age_r`: generated age in years. Empty cells indicate missing age.
- `W1_sex_r`: generated recorded-sex category, 1 = male and 2 = female.
- `W1_depression_it1` through `W1_depression_it9`: integer responses from 0 to 3.
- `W1_<domain>_it1` through `W1_<domain>_it10`: integer responses from 0 to 4 for
  each of the six anxiety domains listed above.

`W1` denotes the first-wave schema. These files are cross-sectional and do not
contain longitudinal trajectories. All questionnaire items and sex are complete.
Missing-age counts are listed for each file. LLM files contain 72 columns,
including ID. Baseline files contain 71 columns.

The depression score is the sum of its nine items, ranging from 0 to 27. Its
five tiers have score ranges 0–4, 5–9, 10–14, 15–19 and 20–27. Each anxiety score
is the mean of its ten items, ranging from 0 to 4 in steps of 0.1. Its tier ranges
are 0.0–0.4, 0.5–1.4, 1.5–2.4, 2.5–3.4 and 3.5–4.0.



## Reading a dataset

Run this example from the repository root.

```python
import pandas as pd

synthetic = pd.read_csv(
    "dataset/data/qwen3.5-27b_natural_own-severity_seed42_n564.csv"
)
features = synthetic.drop(columns=["ID"], errors="ignore")
depression_score = features[
    [f"W1_depression_it{i}" for i in range(1, 10)]
].sum(axis=1)
```
## Real reference data

The real participant dataset is not included in this repository.

- Source project: [Longitudinal examination of DSM-5 anxiety and depression scales](https://osf.io/jz4ge/overview).
- Original file: `data_set_final_osf.sav`, [OSF file page](https://osf.io/c49g8/).
- Publication: Vidal-Arenas, V., Bravo, A. J., Ortet-Walker, J., Ortet, G., Ibáñez, M. I., and Mezquita, L. (2025). Longitudinal measurement invariance of the DSM-5 anxiety and depression severity measures. *European Journal of Psychological Assessment*, 41(3), 174–182. https://doi.org/10.1027/1015-5759/a000791



### Reproducing the analysis

Store the downloaded file outside the release, for example in `private/`.
From the repository root, extract the baseline cohort and summaries with:

```bash
cocast summarize --real private/data_set_final_osf.sav \
  --output work/aggregates.yaml --export-real private/real_baseline.csv
```





