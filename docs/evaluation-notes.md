# Respiratory Model Evaluation Notes

This document summarizes the evaluation artifacts committed in results/reports. The numbers below are transcribed from those CSV reports; they should always be interpreted together with the evaluation protocol that produced them.

## Cross-validation summary

The grouped cross-validation summary reports the following mean metrics:

| Method | MAE | RMSE | R² |
|---|---:|---:|---:|
| Signal-processing baseline | 0.4404 | 1.0532 | 0.8411 |
| Extra Trees | 0.6316 | 1.1916 | 0.8324 |
| Gradient Boosting | 0.6539 | 1.1792 | 0.8320 |
| Random Forest | 0.6791 | 1.1886 | 0.8274 |

The cross-validation report also records fold-to-fold variation for the machine-learning models. For example, Extra Trees has an MAE standard deviation of approximately 0.2664 and an R² standard deviation of approximately 0.0939.

## Pooled out-of-fold summary

The pooled out-of-fold report gives a separate aggregate view:

| Method | MAE | RMSE | R² |
|---|---:|---:|---:|
| Signal-processing baseline | 0.4420 | 1.1059 | 0.8936 |
| Extra Trees | 0.6354 | 1.2714 | 0.8593 |
| Gradient Boosting | 0.6582 | 1.2664 | 0.8604 |
| Random Forest | 0.6805 | 1.2588 | 0.8621 |



## Interpretation

In these committed grouped evaluation artifacts, the conventional signal-processing baseline performs better than the tested machine-learning regressors on MAE and RMSE. This is a useful engineering result: a more complex model is not automatically a better respiratory-rate estimator.

The model-comparison CSV reports a different evaluation view and different metric values. Do not combine metrics from different splits or protocols into a single ranking. Always state the split strategy, aggregation method, target units and sample grouping when presenting results.

## Reproducibility checklist

Before using these metrics in a paper, résumé or presentation:

1. Re-run the pipeline from a clean environment and the documented BIDMC dataset version.
2. Confirm that subject/recording groups are kept separate between training and validation.
3. Record random seeds and library versions.
4. Confirm the target respiratory-rate units and the exact annotation-to-window matching procedure.
5. Report both the baseline and ML models, including uncertainty across folds.
6. Inspect outlier recordings and signal-quality limitations rather than hiding them.

## Scope

These are dataset-based engineering evaluation results. They do not establish clinical validity, diagnostic performance or generalization to unseen patient populations beyond the evaluated dataset.
