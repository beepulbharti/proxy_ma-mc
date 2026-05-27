# primary imports
import sys
sys.path.append("..")

# third party imports
import numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# local imports
import dataset
from boosters import MCBoost
from metrics import subgroup_metrics, mse, _find_worst_bound

def make_group_array(group_indices, n):
    group_array = []
    for group_idx in group_indices:
        group_vector = np.zeros(n, dtype=int)
        group_vector[group_idx] = 1
        group_array.append(group_vector)
    return np.stack(group_array, axis=1)

def make_label_classifier(classifier, seed=0):
    if classifier == "linear":
        return make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500)
        )

    elif classifier == "tree":
        return DecisionTreeClassifier(
            max_depth=25,
            random_state=seed
        )

    elif classifier == "rf":
        return RandomForestClassifier(
            max_depth=25,
            random_state=seed,
            n_jobs=-1
        )

    else:
        raise ValueError(f"Unknown classifier: {classifier}")
    
def load_seed_data(seed, exp_name, min_group_size):
    split = {"train": 0.6, "val": 0.3, "test": 0.1}

    data = dataset.Dataset(
        dataset_name=exp_name,
        groups="default",
        verbose=False,
        split=split,
        min_group_size=min_group_size,
        val_split_seed=seed
    )

    x_train, y_train = data.X_train, data.y_train
    x_val, y_val = data.X_val, data.y_val
    x_test, y_test = data.X_test, data.y_test

    group_train = make_group_array(data.groups_train, x_train.shape[0])
    group_val = make_group_array(data.groups_val, x_val.shape[0])
    group_test = make_group_array(data.groups_test, x_test.shape[0])

    return {
        "data": data,
        "x_train": x_train,
        "y_train": y_train,
        "group_train": group_train,
        "x_val": x_val,
        "y_val": y_val,
        "group_val": group_val,
        "x_test": x_test,
        "y_test": y_test,
        "group_test": group_test,
    }

def train_and_eval_proxies(
    x_train,
    group_train,
    x_val,
    x_test,
    group_test,
    group_names,
    seed=0,
):
    num_groups = group_train.shape[1]

    proxy_models = {}
    proxy_val = np.zeros((x_val.shape[0], num_groups), dtype=group_train.dtype)
    proxy_test = np.zeros((x_test.shape[0], num_groups), dtype=group_train.dtype)

    proxy_errors = np.zeros(num_groups)
    num_samples = np.zeros(num_groups)
    num_p_samples = np.zeros(num_groups)

    for i, group_name in enumerate(group_names):
        g_hat_model = RandomForestClassifier(
            max_depth=25,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1
        )

        g_train = group_train[:, i]
        g_hat_model.fit(x_train, g_train)

        g_hat_val = g_hat_model.predict(x_val)
        g_hat_test = g_hat_model.predict(x_test)

        proxy_models[group_name] = g_hat_model
        proxy_val[:, i] = g_hat_val
        proxy_test[:, i] = g_hat_test

        g_test = group_test[:, i]

        num_samples[i] = np.sum(g_test)
        num_p_samples[i] = np.sum(g_hat_test)
        proxy_errors[i] = mse(g_test, g_hat_test)

    return {
        "proxy_models": proxy_models,
        "proxy_val": proxy_val,
        "proxy_test": proxy_test,
        "proxy_mses": proxy_errors,
        "proxy_errors": proxy_errors,
        "num_samples": num_samples,
        "num_p_samples": num_p_samples,
    }

def compute_bound(proxy_stats, proxy_mses, num_groups):
    mse_fy = proxy_stats["agg"]["MSE"]

    bounds_1 = [
        np.sqrt(mse_fy * proxy_mses[i]) + proxy_stats[i]["ECE_1"]
        for i in range(num_groups)
    ]

    bounds_2 = [
        proxy_mses[i] + proxy_stats[i]["ECE_1"]
        for i in range(num_groups)
    ]

    all_bounds = [
        min(b1, b2)
        for b1, b2 in zip(bounds_1, bounds_2)
    ]

    return max(all_bounds), all_bounds

def run_one_seed(seed, exp_name, min_group_size, classifier):
    d = load_seed_data(seed, exp_name, min_group_size)

    data = d["data"]

    x_train = d["x_train"]
    y_train = d["y_train"]
    group_train = d["group_train"]

    x_val = d["x_val"]
    y_val = d["y_val"]

    x_test = d["x_test"]
    y_test = d["y_test"]
    group_test = d["group_test"]

    num_groups = group_train.shape[1]

    # Learn label classifier
    clf = make_label_classifier(classifier, seed=seed)
    clf.fit(x_train, y_train)

    f_val = clf.predict_proba(x_val)[:, 1]
    f_test = clf.predict_proba(x_test)[:, 1]

    # Learn proxies
    proxy_results = train_and_eval_proxies(
        x_train=x_train,
        group_train=group_train,
        x_val=x_val,
        x_test=x_test,
        group_test=group_test,
        group_names=data.group_names,
        seed=seed,
    )

    proxy_val = proxy_results["proxy_val"]
    proxy_test = proxy_results["proxy_test"]
    proxy_mses = proxy_results["proxy_mses"]

    # Initial validation statistics
    base_alpha = 0.01
    m = np.ceil(1 / base_alpha)

    f_0_val = np.round(f_val * m) / m

    proxy_list_val = [
        np.where(proxy_val[:, i] == 1)[0]
        for i in range(proxy_val.shape[1])
    ]

    proxy_stats_val = subgroup_metrics(proxy_list_val, y_val, f_0_val)

    _, g = _find_worst_bound(proxy_stats_val, proxy_mses)

    # Fit MCBoost if needed
    p_reduction = 0.75
    alpha = proxy_stats_val[g]["ECE_1"] * p_reduction

    print(f"seed={seed}, alpha={alpha}")

    group_list_test = data.groups_test
    proxy_list_test = [
        np.where(proxy_test[:, i] == 1)[0]
        for i in range(proxy_test.shape[1])
    ]

    if proxy_stats_val[g]["ECE_1"] > 0.05:
        mc = MCBoost()
        mc.fit(f_val, y_val, proxy_list_val, alpha ** 2, tol=1e-5)

        m = mc.m
        f_0_test = np.round(f_test * m) / m

        group_stats_test = subgroup_metrics(group_list_test, y_test, f_0_test)
        proxy_stats_test = subgroup_metrics(proxy_list_test, y_test, f_0_test)

        f_adj = mc.predict(f_test, proxy_list_test)

    else:
        mc = None

        f_0_test = np.round(f_test * m) / m

        group_stats_test = subgroup_metrics(group_list_test, y_test, f_0_test)
        proxy_stats_test = subgroup_metrics(proxy_list_test, y_test, f_0_test)

        f_adj = f_0_test

    # Adjusted statistics
    adj_proxy_stats_test = subgroup_metrics(proxy_list_test, y_test, f_adj)
    adj_group_stats_test = subgroup_metrics(group_list_test, y_test, f_adj)

    # Calculate bounds
    init_bound, all_init_bounds = compute_bound(
        proxy_stats=proxy_stats_test,
        proxy_mses=proxy_mses,
        num_groups=num_groups,
    )

    final_bound, all_final_bounds = compute_bound(
        proxy_stats=adj_proxy_stats_test,
        proxy_mses=proxy_mses,
        num_groups=num_groups,
    )

    # Extract relevant statistics
    initial_eces = [
        group_stats_test[i]["ECE_1"]
        for i in range(num_groups)
    ]

    final_eces = [
        adj_group_stats_test[i]["ECE_1"]
        for i in range(num_groups)
    ]

    initial_worst_ece = max(initial_eces)
    final_worst_ece = max(final_eces)

    return {
        "seed": seed,
        "classifier": classifier,

        "proxy errors": proxy_results["proxy_errors"],
        "proxy_mses": proxy_mses,

        "initial eces": initial_eces,
        "initial_worst_ece": initial_worst_ece,
        "init_bound": init_bound,
        "all_init_bounds": all_init_bounds,

        "final eces": final_eces,
        "final_worst_ece": final_worst_ece,
        "final_bound": final_bound,
        "all_final_bounds": all_final_bounds,

        "mcboost_used": mc is not None,
        "alpha": alpha,
        "worst_proxy_group": g,

        "group_stats_test": group_stats_test,
        "proxy_stats_test": proxy_stats_test,
        "adj_group_stats_test": adj_group_stats_test,
        "adj_proxy_stats_test": adj_proxy_stats_test,
    }