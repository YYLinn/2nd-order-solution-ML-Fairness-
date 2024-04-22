import numpy as np
import xgboost as xgb
import scipy.stats as stats
from sklearn.model_selection import RandomizedSearchCV
from test_function import test


def train_base_model(
    X_train, y_train, data_test, unprivileged_groups, privileged_groups
):
    """Building a base model."""
    xgb_model = xgb.XGBClassifier(random_state=2023)
    param_dist = {
        "max_depth": stats.randint(3, 10),
        "learning_rate": stats.uniform(0.01, 0.1),
        "subsample": stats.uniform(0.5, 0.5),
        "n_estimators": stats.randint(50, 200),
    }

    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=10,
        cv=5,
        scoring="roc_auc",
        random_state=2023,
    )
    random_search.fit(X_train, y_train)
    best_xgb_original = random_search.best_estimator_
    thresh_arr = np.linspace(0.01, 0.5, 50)


    return test(
        dataset=data_test,
        model=best_xgb_original,
        thresh_arr=thresh_arr,
        unprivileged_groups=unprivileged_groups,
        privileged_groups=privileged_groups,
    )
