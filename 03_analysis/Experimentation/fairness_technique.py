from aif360.algorithms.preprocessing import Reweighing, LFR, DisparateImpactRemover
from aif360.algorithms.postprocessing import CalibratedEqOddsPostprocessing, RejectOptionClassification


def apply_reweighting(data_train, unprivileged_groups, privileged_groups):
    RW = Reweighing(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )
    RW.fit(data_train)
    return RW.transform(data_train)


def apply_lfr(data_train, unprivileged_groups, privileged_groups):
    lfr = LFR(
        unprivileged_groups=unprivileged_groups, privileged_groups=privileged_groups
    )
    lfr.fit(data_train)
    return lfr.transform(data_train)


def apply_disparate_impact(data_train, unprivileged_groups, privileged_groups):
    di = DisparateImpactRemover(repair_level=1, sensitive_attribute="race")
    data_train_di = di.fit_transform(data_train)
    return data_train_di


def apply_calibrated_eq_odds(
    data_train, data_test, unprivileged_groups, privileged_groups
):
    cpp = CalibratedEqOddsPostprocessing(
        privileged_groups=privileged_groups,
        unprivileged_groups=unprivileged_groups,
        cost_constraint="fnr",
        seed=2023,
    )
    cpp = cpp.fit(data_test, data_test)
    return cpp.predict(data_test)


def apply_reject_option_classification(
    data_train, data_test, unprivileged_groups, privileged_groups
):
    roc = RejectOptionClassification(unprivileged_groups, privileged_groups)
    return roc.fit_predict(data_test, data_test)
