import numpy as np

def _find_worst_bound(stats, proxy_errors):
    num_groups = len(proxy_errors)
    mse_fy_test = stats['agg']['MSE']
    bounds_1 = [(mse_fy_test*proxy_errors[i])**0.5 + stats[i]['ECE_1'] for i in range(num_groups)]
    bounds_2 = [(proxy_errors[i] + stats[i]['ECE_1']) for i in range(num_groups)]
    all_bounds = np.array([min(b1, b2) for (b1, b2) in zip(bounds_1, bounds_2)])
    bound = np.max(all_bounds)
    g = np.argmax(all_bounds)
    return bound, g

def mse(f, y):
    # Mean squared error
    return np.mean((f - y) ** 2)

def AE(f, y):
    # Accuracy in expectation
    return (np.mean(f) - np.mean(y))**2

def discretized_ECE(predictions, labels, p=2): 
    # Discretized L1/L2 ECE
    ece = 0.0 
    unique_bins = np.unique(predictions)
    
    for bin_value in unique_bins:
        bin_mask = predictions == bin_value
        
        bin_size = np.sum(bin_mask)
        if bin_size > 0:
            bin_confidence = bin_value 
            bin_accuracy = np.mean(labels[bin_mask])
            
            if p == 1:
                ece += bin_size * np.abs(bin_confidence - bin_accuracy)
            elif p == 2:
                ece += bin_size * (bin_confidence - bin_accuracy)**2

    return ece / len(predictions)

def subgroup_metrics(subgroups, targets, positive_class_confs):
    # Overall and subgroup metrics
    subgroup_metrics = {}

    for i, group in enumerate(subgroups):
        p_g = len(group) / len(positive_class_confs) # P(g=1)

        # check for empty subgroups
        if len(group) <= 1:
            subgroup_metrics[i] = {
                "size": p_g, 
                "acc": 'NA',
                "AE": 'NA',
                "ECE": 'NA',
                "smECE": 'NA',
            }
            continue

        subgroup_confs = positive_class_confs[group]
        subgroup_targets = targets[group]

        # deterministic metrics
        ae = AE(subgroup_confs, subgroup_targets)
        ece_1 = discretized_ECE(subgroup_confs, subgroup_targets, p=1)
        ece_2 = discretized_ECE(subgroup_confs, subgroup_targets, p=2)

        subgroup_metrics[i] = {
            "p_g": p_g,
            "AE": p_g*ae,
            "ECE_1": p_g*ece_1,
            "ECE_2": p_g*ece_2,
        }

    # get aggregate metrics
    agg_metrics = {
        "p_g": 1.0,
        "MSE": np.mean((positive_class_confs - targets)**2),
        "ECE_1": round(discretized_ECE(positive_class_confs, targets, p=1), 4),
        "ECE_2": round(discretized_ECE(positive_class_confs, targets, p=2), 4),
    }
    max_ece2_group = max(
            (i for i in subgroup_metrics if subgroup_metrics[i]['ECE_2'] != 'NA'),
            key=lambda i: subgroup_metrics[i]['ECE_2']
        )
    min_ece2_group = min(
            (i for i in subgroup_metrics if subgroup_metrics[i]['ECE_2'] != 'NA'),
            key=lambda i: subgroup_metrics[i]['ECE_2']
        )

    # subgroup_metrics['mean'] = sg_mean
    subgroup_metrics['max'] = subgroup_metrics[max_ece2_group]
    subgroup_metrics['min'] = subgroup_metrics[min_ece2_group]
    subgroup_metrics['agg'] = agg_metrics

    return subgroup_metrics

def subgroup_metrics_ae(subgroups, targets, positive_class_confs):
    # Overall and subgroup metrics
    subgroup_metrics = {}

    for i, group in enumerate(subgroups):
        p_g = len(group) / len(positive_class_confs) # P(g=1)

        # check for empty subgroups
        if len(group) <= 1:
            subgroup_metrics[i] = {
                "size": p_g, 
                "acc": 'NA',
                "AE": 'NA',
            }
            continue

        subgroup_confs = positive_class_confs[group]
        subgroup_targets = targets[group]

        # deterministic metrics
        ae = AE(subgroup_confs, subgroup_targets)

        subgroup_metrics[i] = {
            "p_g": p_g,
            "AE": p_g*ae,
        }

    # get aggregate metrics
    agg_metrics = {
        "p_g": 1.0,
        "MSE": np.mean((positive_class_confs - targets)**2),
    }

    max_ae_group = max(
            (i for i in subgroup_metrics if subgroup_metrics[i]['AE'] != 'NA'),
            key=lambda i: subgroup_metrics[i]['AE']
        )
    min_ae_group = min(
            (i for i in subgroup_metrics if subgroup_metrics[i]['AE'] != 'NA'),
            key=lambda i: subgroup_metrics[i]['AE']
        )


    # subgroup_metrics['mean'] = sg_mean
    subgroup_metrics['max'] = subgroup_metrics[max_ae_group]
    subgroup_metrics['min'] = subgroup_metrics[min_ae_group]
    subgroup_metrics['agg'] = agg_metrics

    return subgroup_metrics