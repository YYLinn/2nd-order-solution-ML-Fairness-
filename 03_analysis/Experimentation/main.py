import argparse
from dataset import load_dataset
from base_model import train_base_model
from fairness_technique import (
    apply_reweighting,
    apply_lfr,
    apply_disparate_impact,
    apply_calibrated_eq_odds,
    apply_reject_option_classification,
)
from generate_json import generate_json


def main():
    parser = argparse.ArgumentParser(
        description="Perform the required experimentation."
    )
    parser.add_argument(
        "--technique", type=str, help="Name of the fairness technique to use"
    )
    parser.add_argument(
        "--sensitive_attr",
        type=str,
        help="Name of the sensitive attribute",
    )
    args = parser.parse_args()

    if args.technique is None:
        print("Please provide a technique name using --technique")
        return
    if args.sensitive_attr is None:
        print(
            "Please provide a name for the sensitive attribute using --sensitive_attr"
        )
        return

    unprivileged_groups = [{args.sensitive_attr: 0}]
    privileged_groups = [{args.sensitive_attr: 1}]
    # Load dataset

    data_train, data_test = load_dataset(args.sensitive_attr)

    # Train base model
    base_model_metrics = train_base_model(
        data_train.features,
        data_train.labels.ravel(),
        data_test,
        unprivileged_groups,
        privileged_groups,
    )

    # Apply the specified fairness technique
    if args.technique == "reweighting":
        # Apply reweighting
        data_train_fair = apply_reweighting(
            data_train, unprivileged_groups, privileged_groups
        )
    elif args.technique == "lfr":
        # Apply LFR
        data_train_fair = apply_lfr(data_train, unprivileged_groups, privileged_groups)
    elif args.technique == "disparate_impact":
        # Apply Disparate Impact Remover
        data_train_fair = apply_disparate_impact(
            data_train, unprivileged_groups, privileged_groups
        )
    elif args.technique == "calibrated_eq_odds":
        # Apply Calibrated Equality of Odds
        data_train_fair = apply_calibrated_eq_odds(
            data_train, data_test, unprivileged_groups, privileged_groups
        )
    elif args.technique == "reject_option_classification":
        # Apply Reject Option Classification
        data_train_fair = apply_reject_option_classification(
            data_train, data_test, unprivileged_groups, privileged_groups
        )
    else:
        print("Unknown fairness technique:", args.technique)
        return

    # Train and evaluate model using the fair dataset
    fair_model_metrics = train_base_model(
        data_train_fair.features,
        data_train_fair.labels.ravel(),
        data_test,
        unprivileged_groups,
        privileged_groups,
    )

    # Generate JSON outputs
    generate_json(
        [base_model_metrics, fair_model_metrics],
        "fairness_metrics.json",
    )


if __name__ == "__main__":
    main()
