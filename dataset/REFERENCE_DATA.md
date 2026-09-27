# Real reference data

The real participant dataset is not included in this repository.

- Source project: [Longitudinal examination of DSM-5 anxiety and depression scales](https://osf.io/jz4ge/overview).
- Original file: `data_set_final_osf.sav`, [OSF file page](https://osf.io/c49g8/).
- Publication: Vidal-Arenas, V., Bravo, A. J., Ortet-Walker, J., Ortet, G., Ibáñez, M. I., and Mezquita, L. (2025). Longitudinal measurement invariance of the DSM-5 anxiety and depression severity measures. *European Journal of Psychological Assessment*, 41(3), 174–182. https://doi.org/10.1027/1015-5759/a000791



## Reproducing the analysis

Store the downloaded file outside the release, for example in `private/`.
From the repository root, extract the baseline cohort and summaries with:

```bash
cocast summarize --real private/data_set_final_osf.sav \
  --output work/aggregates.yaml --export-real private/real_baseline.csv
```

