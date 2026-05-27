# third-party imports
import numpy as np
from tqdm import tqdm

# local imports
from metrics import subgroup_metrics

class MCBoost:
    """
    MCBoost: Multi-class Boosting for Group Fairness
    """

    def __init__(self):
        pass

    def _find_max_patch(self, f, y, groups):
        """
        Find the prediction value and group with maximum calibration error.

        Parameters:
        - f: ndarray of shape (n_samples,) - Predictions.
        - y: ndarray of shape (n_samples,) - True labels.
        - groups: ndarray of shape (n_samples, n_groups) - Group indicators.

        Returns:
        - v_t: The prediction value with maximum calibration error.
        - g_t: The group with maximum calibration error.
        - patch_vg: The patch to apply to the group with maximum calibration error.
        """

        # Initialize variables
        n = f.shape[0]
        unique_v = np.unique(f) 
        max_value = 0 
        v_t = None 
        g_t = None 

        for g_id, group_indices in enumerate(groups):
            group_predictions = f[group_indices]
            group_y = y[group_indices]

            # Precompute masks for unique prediction values
            masks = [np.where(group_predictions == v)[0] for v in unique_v]

            # Compute the probabilities and expectations
            probabilities = np.array([len(mask)/n for mask in masks])
            expectations = np.array([
                np.mean(group_y[mask]) - v if mask.any() else 0
                for mask, v in zip(masks, unique_v)
            ])
            patches = np.array([
                np.mean(group_y[mask]) if mask.any() else 0
                for mask in masks
            ])

            # Compute the product for the current group
            products = probabilities * (expectations**2)

            # Find the maximum for the current group
            max_idx = np.argmax(products)
            if products[max_idx] > max_value:
                worst_g = g_id
                worst_v = unique_v[max_idx]
                # delta_vg = expectations[max_idx]
                patch_vg = patches[max_idx]
                max_value = products[max_idx]
            
        return worst_v, worst_g, patch_vg
    
    def _update_predictions(self, f_t, g_t, v_t, groups, patch):
        # Copy the predictions to avoid modifying the original array
        f_t_new = f_t.copy()

        # Get indices of the selected group
        group_indices = groups[g_t]

        # Find the rows in the group that match the value of f
        matching_indices = group_indices[f_t_new[group_indices] == v_t]

        # Add the patch to the matching rows
        f_t_new[matching_indices] = patch

        return f_t_new

        
    def fit(self, f, y, groups, alpha, tol=1e-3):
        t = 0
        m = np.ceil(1/alpha)
        f_t = np.round(f * m) / m
        group_metrics = subgroup_metrics(groups, y, f_t)
        mses = []
        violations = []
        previous_mse = group_metrics['agg']['MSE']
        worst_violation = group_metrics['max']['ECE_2']
        mses.append(previous_mse)
        violations.append(worst_violation)

        adjustments = {}
        consecutive_small_diffs = 0
        while worst_violation >= alpha:
            print(f"Iteration: {t}", end="\r")
            v_t, g_t, patch_vg = self._find_max_patch(f_t, y, groups)
            patch_vg = np.round(patch_vg * m) / m
            adjustments[t] = {"v_t": v_t, "g_t": g_t, "patch_vg": patch_vg}
            f_t = self._update_predictions(f_t, g_t, v_t, groups, patch_vg)
            group_metrics = subgroup_metrics(groups, y, f_t)
            
            # Current MSE
            current_mse = group_metrics['agg']['MSE']
            mses.append(current_mse)

            # Check if the absolute difference in MSE is smaller than the tolerance
            if abs(current_mse - previous_mse) < tol:
                consecutive_small_diffs += 1
            else:
                consecutive_small_diffs = 0 
            
            if consecutive_small_diffs >= 3:
                print("\nFinished fitting because MSE is not decreasing for 3 consecutive iterations")
                break
            
            # Update previous MSE for the next iteration
            previous_mse = current_mse
            worst_violation = group_metrics['max']['ECE_2']
            violations.append(worst_violation)

            t += 1

        print(f"\nModel fitting complete after {t} round(s)")
        self.m = m
        self.alpha = alpha
        self.mses = mses
        self.violations = violations
        self.adjustments = adjustments

    def predict(self, f, groups):
        if self.adjustments is None:
            raise RuntimeError("The model must be fitted using the `fit` method before calling `predict`.")
        f_t = f.copy()
        f_t = np.round(f_t * self.m) / self.m
        for t, data in tqdm(self.adjustments.items()):
            v_t = data["v_t"]
            g_t = data["g_t"]
            patch_vg = data["patch_vg"]
            f_t = self._update_predictions(f_t, g_t, v_t, groups, patch_vg)
        
        return f_t


class MABoost:
    """
    Roth 2022: Algorithm 10.
    Learns a predictor that satisfies multiaccuracy
    """
    def __init__(self):
        pass

    def fit(self, f, y, groups):
        """
        Fit the model by solving the least-squares problem.

        Parameters:
        - f: ndarray of shape (n_samples,) - Original predictions.
        - y: ndarray of shape (n_samples,) - True labels.
        - groups: ndarray of shape (n_samples, n_groups) - Group indicators (A in ||Ax - b||^2).

        Sets:
        - self.lambdas: ndarray of shape (n_groups,) - The fitted coefficients.
        """

        # Solving the least squares problem ||Ax - b||^2
        b = y - f  # Residuals
        self.lambdas = np.linalg.pinv(groups) @ b  # Compute lambdas using pseudo-inverse

    def predict(self, f, groups):
        """
        Predict updated scores based on the fitted coefficients.

        Parameters:
        - f: ndarray of shape (n_samples,) - Original scores.
        - groups: ndarray of shape (n_samples, n_groups) - Group indicators (A in ||Ax - b||^2).

        Returns:
        - f_hat: ndarray of shape (n_samples,) - Adjusted scores.

        Raises:
        - RuntimeError: If `fit` has not been called before `predict`.
        """
        if self.lambdas is None:
            raise RuntimeError("The model must be fitted using the `fit` method before calling `predict`.")

        # Ensure inputs are NumPy arrays
        f = np.asarray(f)
        groups = np.asarray(groups)

        # Compute adjusted predictions
        f_adj = f + groups @ self.lambdas

        return f_adj
        