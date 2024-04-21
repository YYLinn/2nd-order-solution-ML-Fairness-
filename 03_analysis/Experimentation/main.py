from dataset import load_dataset
from base_model import train_base_model
from techniques import (
    apply_reweighting,
    apply_lfr,
    apply_disparate_impact,
    apply_calibrated_eq_odds,
    apply_reject_option_classification,
)
from generate_json import generate_json

# Load dataset
data_train, data_test = load_dataset()

# Define privileged and unprivileged groups
unprivileged_groups = [{"race": 0}]
privileged_groups = [{"race": 1}]

# Train base model
base_model_metrics = train_base_model(
    data_train.features,
    data_train.labels.ravel(),
    data_test.features,
    data_test.labels.ravel(),
    unprivileged_groups,
    privileged_groups,
)

# Apply reweighting
data_train_reweighed = apply_reweighting(
    data_train, unprivileged_groups, privileged_groups
)
reweight_metrics = train_base_model(
    data_train_reweighed.features,
    data_train_reweighed.labels.ravel(),
    data_test.features,
    data_test.labels.ravel(),
    unprivileged_groups,
    privileged_groups,
)

# Apply LFR
data_train_lfr = apply_lfr(data_train, unprivileged_groups, privileged_groups)
lfr_metrics = train_base_model(
    data_train_lfr.features,
    data_train_lfr.labels.ravel(),
    data_test.features,
    data_test.labels.ravel(),
    unprivileged_groups,
    privileged_groups,
)

# Apply Disparate Impact Remover
data_train_di = apply_disparate_impact(
    data_train, unprivileged_groups, privileged_groups
)
di_metrics = train_base_model(
    data_train_di.features,
    data_train_di.labels.ravel(),
    data_test.features,
    data_test.labels.ravel(),
    unprivileged_groups,
    privileged_groups,
)

# Apply Calibrated Equality of Odds
data_test_cpp_new = apply_calibrated_eq_odds(
    data_train, data_test, unprivileged_groups, privileged_groups
)
cp_metrics = train_base_model(
    data_train.features,
    data_train.labels.ravel(),
    data_test_cpp_new.features,
    data_test_cpp_new.labels.ravel(),
    unprivileged_groups,
    privileged_groups,
)

# Apply Reject Option Classification
data_test_roc = apply_reject_option_classification(
    data_train, data_test, unprivileged_groups, privileged_groups
)
roc_metrics = train_base_model(
    data_train.features,
    data_train.labels.ravel(),
    data_test_roc.features,
    data_test_roc.labels.ravel(),
    unprivileged_groups,
    privileged_groups,
)

# Generate JSON outputs
generate_json(
    [
        base_model_metrics,
        reweight_metrics,
        lfr_metrics,
        di_metrics,
        cp_metrics,
        roc_metrics,
    ],
    "fairness_metrics.json",
)
