from collections import defaultdict
import numpy as np
from aif360.metrics import ClassificationMetric


def test(dataset, model, thresh_arr, unprivileged_groups, privileged_groups):
    try:
        # sklearn classifier
        y_val_pred_prob = model.predict_proba(dataset.features)
        pos_ind = 1
    except AttributeError:
        # aif360 inprocessing algorithm
        y_val_pred_prob = model.predict(dataset).scores
        pos_ind = 0

    metric_arrs = defaultdict(list)
    for thresh in thresh_arr:
        y_val_pred = (y_val_pred_prob[:, pos_ind] > thresh).astype(np.float64)

        dataset_pred = dataset.copy()
        dataset_pred.labels = y_val_pred
        metric = ClassificationMetric(
            dataset,
            dataset_pred,
            unprivileged_groups=unprivileged_groups,
            privileged_groups=privileged_groups,
        )

        metric_arrs["acc"].append(metric.accuracy())
        metric_arrs["bal_acc"].append(
            (metric.true_positive_rate() + metric.true_negative_rate()) / 2
        )
        metric_arrs["threshold_value"].append(thresh)
        metric_arrs["F1_score"].append(
            2
            * (
                (metric.precision() * metric.recall())
                / (metric.precision() + metric.recall())
            )
        )
        metric_arrs["false_omission_rate"].append(metric.false_omission_rate())
        metric_arrs["false_omission_rate_ratio"].append(
            metric.false_omission_rate_ratio()
        )
        metric_arrs["avg_odds_diff"].append(metric.average_odds_difference())
        metric_arrs["disp_imp"].append(metric.disparate_impact())
        metric_arrs["stat_par_diff"].append(metric.statistical_parity_difference())
        metric_arrs["eq_opp_diff"].append(metric.equal_opportunity_difference())
        metric_arrs["theil_ind"].append(metric.theil_index())

    return metric_arrs
