"""Regression Decision Tree algorithm implemented using PyTorch tensors."""

import torch
from tqdm import tqdm


class RegressionDecisionTree:
    """Regression Decision Tree with look-ahead capability and pruning support.

    Parameters
    ----------
    max_depth : int
        Maximum depth of the tree.
    min_impurity_decrease : float
        Minimum impurity decrease required to split (lower bound).
    look_ahead : int, optional
        Number of levels to look ahead when evaluating splits. Defaults to 0.
    criterion : str, optional
        The function to measure split quality. Supported criteria are "MSE" (mean
        squared error) and "absolute_error" (mean absolute deviation). Defaults to "MSE".
    leaf_value : str, optional
        The method to compute leaf node values. Supported methods are "mean" and
        "median". Defaults to "mean".
    """

    class Node:
        """Tree node representing either a split or a leaf."""

        def __init__(
            self,
            feature=None,
            threshold=None,
            left=None,
            right=None,
            value=None,
            decrease=None,
            n_samples=None,
            sum_y=None,
        ):
            """Initialize a tree node.

            Args:
                feature (int, optional): Index of the feature to split on.
                threshold (float, optional): Threshold value for the split.
                left (Node, optional): Left child node.
                right (Node, optional): Right child node.
                value (float, optional): Predicted value if leaf node.
                decrease (float, optional): Impurity decrease from the split.
                n_samples (int, optional): Number of samples in the subtree.
                sum_y (torch.Tensor, optional): Sum of target values in the subtree.
            """
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value
            self.decrease = decrease
            self.n_samples = n_samples
            self.sum_y = sum_y

    def __init__(
        self,
        max_depth,
        min_impurity_decrease,
        look_ahead=0,
        criterion="MSE",
        leaf_value="mean",
    ):
        """Initialize the RegressionDecisionTree with hyperparameters.

        Args:
            max_depth (int): Maximum depth of the tree.
            min_impurity_decrease (float): Minimum impurity decrease required to split
                (lower bound).
            look_ahead (int, optional): Number of levels to look ahead when evaluating
                splits. Defaults to 0.
            criterion (str, optional): The function to measure split quality. Supported
                criteria are "MSE" (mean squared error) and "absolute_error" (mean absolute
                deviation). Defaults to "MSE".
            leaf_value (str, optional): The method to compute leaf node values. Supported
                methods are "mean" and "median". Defaults to "mean".
        """
        if criterion not in ["MSE", "absolute_error"]:
            raise ValueError(
                f"criterion must be 'MSE' or 'absolute_error', got '{criterion}'"
            )
        if leaf_value not in ["mean", "median"]:
            raise ValueError(
                f"leaf_value must be 'mean' or 'median', got '{leaf_value}'"
            )

        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.look_ahead = look_ahead
        self.criterion = criterion
        self.leaf_value = leaf_value
        self.root = None
        self.impurity_decreases = []

    def _compute_impurity(self, y, mask=None):
        """Compute the impurity for a set of targets based on the selected criterion.

        Args:
            y (torch.Tensor): Target tensor.
            mask (torch.Tensor, optional): Boolean mask indicating which samples to include.
                If None, all samples are included.

        Returns:
            float: Impurity value.
        """
        if mask is None:
            n = len(y)
            if n == 0:
                return 0.0
            if self.criterion == "MSE":
                return torch.sum(y**2) - (torch.sum(y) ** 2) / n
            # absolute_error
            if self.leaf_value == "mean":
                center = torch.sum(y) / n
            else:  # median
                center = torch.median(y)
            return torch.sum(torch.abs(y - center))

        n = torch.sum(mask).item()
        if n == 0:
            return 0.0
        y_masked = torch.where(
            mask, y, torch.tensor(0.0, dtype=y.dtype, device=y.device)
        )

        if self.criterion == "MSE":
            sum_y = torch.sum(y_masked)
            sum_y2 = torch.sum(y_masked**2)
            return sum_y2 - (sum_y**2) / n
        # absolute_error
        if self.leaf_value == "mean":
            center = torch.sum(y_masked) / n
        else:  # median
            y_values = y[mask]
            center = torch.median(y_values)
        return torch.sum(torch.abs(y_masked - center * mask.float()))

    def _find_best_split(
        self, X, y, mask=None, look_ahead=None
    ):  # pylint: disable=invalid-name,too-many-branches
        """Find the best feature and split point that maximizes impurity decrease.

        Args:
            X (torch.Tensor): Feature tensor of shape (n_samples, n_features).
            y (torch.Tensor): Target tensor of shape (n_samples,).
            mask (torch.Tensor, optional): Boolean mask indicating which samples to consider.
                If None, all samples are considered.
            look_ahead (int, optional): Look ahead levels for this call.
                Defaults to self.look_ahead.

        Returns:
            tuple: (best_feature, best_threshold, best_left_mask, best_right_mask,
                best_augmented_decrease, best_immediate_decrease)
                   or (None, None, None, None, -inf, -inf) if no valid split is found.
        """
        # pylint: disable=too-many-statements
        if look_ahead is None:
            look_ahead = self.look_ahead

        n_samples, n_features = X.shape

        # Create or validate mask
        if mask is None:
            mask = torch.ones(n_samples, dtype=torch.bool, device=X.device)

        n_masked = torch.sum(mask).item()
        if n_masked < 2:
            return None, None, None, None, float("-inf"), float("-inf")

        current_impurity = self._compute_impurity(y, mask)

        if look_ahead == 0:
            # Vectorized implementation for look_ahead == 0
            # Sort indices for each feature, considering only masked samples
            feature_sorted_indices = torch.argsort(X, dim=0)
            y_sorted = y[feature_sorted_indices]
            x_sorted = torch.gather(X, 0, feature_sorted_indices)

            # Propagate mask through sorting
            mask_sorted = mask[feature_sorted_indices]

            consecutive_same = x_sorted[1:, :] == x_sorted[:-1, :]

            # Compute cumulative sums, but only for masked samples
            y_sorted_masked = torch.where(
                mask_sorted, y_sorted, torch.tensor(0.0, dtype=y.dtype, device=y.device)
            )
            mask_sorted_float = mask_sorted.float()

            cumsum_y = torch.cumsum(y_sorted_masked, dim=0)
            cumsum_y2 = torch.cumsum(y_sorted_masked**2, dim=0)
            cumsum_mask = torch.cumsum(mask_sorted_float, dim=0)

            left_sum_y = cumsum_y[:-1, :]
            left_sum_y2 = cumsum_y2[:-1, :]
            left_n = cumsum_mask[:-1, :]

            total_sum_y = cumsum_y[-1, :]
            total_sum_y2 = cumsum_y2[-1, :]
            total_n = cumsum_mask[-1, :]

            right_sum_y = total_sum_y - left_sum_y
            right_sum_y2 = total_sum_y2 - left_sum_y2
            right_n = total_n - left_n

            # Compute impurities, avoiding division by zero
            left_impurity = torch.where(
                left_n > 0,
                left_sum_y2 - (left_sum_y**2) / left_n,
                torch.tensor(0.0, dtype=y.dtype, device=y.device),
            )
            right_impurity = torch.where(
                right_n > 0,
                right_sum_y2 - (right_sum_y**2) / right_n,
                torch.tensor(0.0, dtype=y.dtype, device=y.device),
            )
            impurity_after_split = left_impurity + right_impurity

            decreases = current_impurity - impurity_after_split
            # Invalidate splits where consecutive values are same or split doesn't respect mask
            # Also invalidate if left or right side would have no masked samples
            invalid_splits = consecutive_same | (left_n == 0) | (right_n == 0)
            decreases = torch.where(
                invalid_splits,
                torch.tensor(float("-inf"), dtype=decreases.dtype, device=X.device),
                decreases,
            )

            max_decrease_per_feature, best_split_per_feature = torch.max(
                decreases, dim=0
            )
            best_feature = torch.argmax(max_decrease_per_feature)
            best_decrease = max_decrease_per_feature[best_feature]
            best_split = best_split_per_feature[best_feature]

            if best_decrease == float("-inf"):
                return None, None, None, None, float("-inf"), float("-inf")

            sorted_indices = feature_sorted_indices[:, best_feature]
            threshold_left = X[sorted_indices[best_split], best_feature]
            threshold_right = X[sorted_indices[best_split + 1], best_feature]
            best_threshold = (threshold_left + threshold_right) / 2

            # Create masks instead of indices
            feature_values = X[:, best_feature]
            best_left_mask = mask & (feature_values <= best_threshold)
            best_right_mask = mask & (feature_values > best_threshold)

            return (
                best_feature,
                best_threshold,
                best_left_mask,
                best_right_mask,
                best_decrease,
                best_decrease,
            )

        # Looped implementation for look_ahead > 0
        best_feature = None
        best_threshold = None
        best_left_mask = None
        best_right_mask = None
        best_augmented = float("-inf")
        best_immediate = float("-inf")

        for feature in range(n_features):
            feature_values = X[:, feature]
            # Get unique values among masked samples
            masked_feature_values = torch.where(
                mask,
                feature_values,
                torch.tensor(float("inf"), dtype=X.dtype, device=X.device),
            )
            sorted_indices = torch.argsort(masked_feature_values)
            sorted_feature_values = masked_feature_values[sorted_indices]
            sorted_mask = mask[sorted_indices]

            # Find the range of valid (masked) samples
            n_valid = torch.sum(sorted_mask).item()
            if n_valid < 2:
                continue

            for split in range(1, int(n_valid)):
                # Get the actual indices in the original tensor
                split_idx = split
                if (
                    sorted_feature_values[split_idx - 1]
                    == sorted_feature_values[split_idx]
                ):
                    continue

                # Create masks for left and right splits
                threshold = (
                    sorted_feature_values[split_idx - 1]
                    + sorted_feature_values[split_idx]
                ) / 2
                left_mask = mask & (feature_values <= threshold)
                right_mask = mask & (feature_values > threshold)

                # Check if both sides have samples
                if (
                    torch.sum(left_mask).item() == 0
                    or torch.sum(right_mask).item() == 0
                ):
                    continue

                left_imp = self._compute_impurity(y, left_mask)
                right_imp = self._compute_impurity(y, right_mask)
                immediate_after = left_imp + right_imp
                immediate_decrease = current_impurity - immediate_after

                left_best_aug = self._find_best_split(
                    X, y, mask=left_mask, look_ahead=look_ahead - 1
                )[4]
                if left_best_aug == float("-inf"):
                    left_best_aug = 0.0
                right_best_aug = self._find_best_split(
                    X, y, mask=right_mask, look_ahead=look_ahead - 1
                )[4]
                if right_best_aug == float("-inf"):
                    right_best_aug = 0.0

                augmented = immediate_decrease + left_best_aug + right_best_aug

                if augmented > best_augmented:
                    best_augmented = augmented
                    best_immediate = immediate_decrease
                    best_feature = feature
                    best_threshold = threshold
                    best_left_mask = left_mask.clone()
                    best_right_mask = right_mask.clone()

        if best_feature is None:
            return None, None, None, None, float("-inf"), float("-inf")
        return (
            best_feature,
            best_threshold,
            best_left_mask,
            best_right_mask,
            best_augmented,
            best_immediate,
        )

    def _compute_leaf_value(self, y, mask):
        """Compute the leaf node value based on the selected leaf_value method.

        Args:
            y (torch.Tensor): Target tensor of shape (n_samples,).
            mask (torch.Tensor): Boolean mask indicating which samples to consider.

        Returns:
            torch.Tensor: The computed leaf value.
        """
        n_samples = torch.sum(mask).item()
        if n_samples == 0:
            return torch.tensor(0.0, device=y.device)

        if self.leaf_value == "mean":
            y_masked = torch.where(
                mask, y, torch.tensor(0.0, dtype=y.dtype, device=y.device)
            )
            sum_y = torch.sum(y_masked)
            return sum_y / n_samples
        # median
        y_values = y[mask]
        return torch.median(y_values)

    def _build_tree(self, X, y, mask, depth):  # pylint: disable=invalid-name
        """Recursively build the regression decision tree.

        Args:
            X (torch.Tensor): Feature tensor of shape (n_samples, n_features).
            y (torch.Tensor): Target tensor of shape (n_samples,).
            mask (torch.Tensor): Boolean mask indicating which samples to consider.
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the constructed tree or subtree.
        """
        n_samples = torch.sum(mask).item()
        y_masked = torch.where(
            mask, y, torch.tensor(0.0, dtype=y.dtype, device=y.device)
        )
        sum_y = torch.sum(y_masked)

        if depth == self.max_depth or n_samples < 2:
            value = self._compute_leaf_value(y, mask)
            return self.Node(value=value, n_samples=n_samples, sum_y=sum_y)

        (
            feature,
            threshold,
            left_mask,
            right_mask,
            augmented_decrease,
            immediate_decrease,
        ) = self._find_best_split(X, y, mask)

        if augmented_decrease > self.min_impurity_decrease and feature is not None:
            left_node = self._build_tree(X, y, left_mask, depth + 1)
            right_node = self._build_tree(X, y, right_mask, depth + 1)

            self.impurity_decreases.append(immediate_decrease)

            return self.Node(
                feature=feature,
                threshold=threshold,
                left=left_node,
                right=right_node,
                decrease=immediate_decrease,
                n_samples=left_node.n_samples + right_node.n_samples,
                sum_y=left_node.sum_y + right_node.sum_y,
            )

        value = self._compute_leaf_value(y, mask)
        return self.Node(value=value, n_samples=n_samples, sum_y=sum_y)

    def fit(self, X, y):  # pylint: disable=invalid-name
        """Build the regression decision tree using the provided data.

        Args:
            X (torch.Tensor): 2D tensor of shape (n_samples, n_features) with features.
            y (torch.Tensor): 1D tensor of shape (n_samples,) with targets.
        """
        self.impurity_decreases = []  # Reset for each fit
        n_samples = X.shape[0]
        mask = torch.ones(n_samples, dtype=torch.bool, device=X.device)
        self.root = self._build_tree(X, y, mask, 0)

    def _predict_one(self, node, x):
        """Make a prediction for a single sample.

        Args:
            node (Node): Current node in the tree.
            x (torch.Tensor): Feature vector of shape (n_features,).

        Returns:
            float: Predicted value.
        """
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return self._predict_one(node.left, x)
        return self._predict_one(node.right, x)

    def predict(self, X: torch.Tensor) -> torch.Tensor:  # pylint: disable=invalid-name
        """Make predictions for a batch of samples.

        Args:
            X (torch.Tensor): 2D tensor of shape (n_samples, n_features) with features.

        Returns:
            torch.Tensor: 1D tensor of shape (n_samples,) with predictions.
        """
        if self.root is None:
            raise ValueError("The tree has not been fitted yet.")

        n_samples = X.shape[0]
        if n_samples == 0:
            root_sum_y = self.root.sum_y
            if isinstance(root_sum_y, torch.Tensor):
                return torch.empty(0, dtype=root_sum_y.dtype, device=X.device)
            return torch.empty(0, dtype=X.dtype, device=X.device)

        root_sum_y = self.root.sum_y
        pred_dtype = (
            root_sum_y.dtype
            if isinstance(root_sum_y, torch.Tensor)
            else X.dtype if torch.is_floating_point(X) else torch.get_default_dtype()
        )
        predictions = torch.empty(n_samples, dtype=pred_dtype, device=X.device)

        stack = [
            (
                self.root,
                torch.ones(n_samples, dtype=torch.bool, device=X.device),
            )
        ]

        while stack:
            node, mask = stack.pop()
            if not torch.any(mask):
                continue
            if node.value is not None:
                value = node.value.to(dtype=pred_dtype, device=X.device)
                predictions = torch.where(mask, value, predictions)
                continue

            # At this point, node is a split node and must have a threshold
            assert (
                node.threshold is not None
            ), "Split node must have a threshold"  # nosec
            threshold = node.threshold.to(dtype=X.dtype, device=X.device)

            feature_values = X[:, node.feature]
            go_left = feature_values <= threshold
            left_mask = mask & go_left
            right_mask = mask & (~go_left)

            if node.left is not None:
                stack.append((node.left, left_mask))
            if node.right is not None:
                stack.append((node.right, right_mask))

        return predictions

    def copy_tree(self, node):
        """Deep copy a tree node and its subtrees.

        Args:
            node (Node): The node to copy.

        Returns:
            Node: A deep copy of the node and its subtrees.
        """
        if node is None:
            return None
        new_node = self.Node(
            feature=node.feature,
            threshold=node.threshold,
            value=node.value,
            decrease=node.decrease,
            n_samples=node.n_samples,
            sum_y=node.sum_y,
        )
        new_node.left = self.copy_tree(node.left)
        new_node.right = self.copy_tree(node.right)
        return new_node

    def prune(self, min_imp, node=None):
        """Prune the tree bottom-up for a given min_impurity_decrease.

        Args:
            min_imp (float): The min_impurity_decrease threshold for pruning.
            node (Node, optional): The current node to prune (defaults to root).
        """
        if node is None:
            node = self.root
        if node is None or node.value is not None:
            return
        if node.left is not None:
            self.prune(min_imp, node.left)
        if node.right is not None:
            self.prune(min_imp, node.right)
        if node.decrease is not None and node.decrease <= min_imp:
            if node.sum_y is not None and node.n_samples is not None:
                node.value = node.sum_y / node.n_samples
            node.left = None
            node.right = None
            node.feature = None
            node.threshold = None
            node.decrease = None

    def find_optimal_min_impurity_decrease(  # pylint: disable=invalid-name
        self, X, y, n_folds=5, k_repeats=3
    ):
        """Find the optimal min_impurity_decrease using repeated n-fold cross-validation.

        Args:
            X (torch.Tensor): Features.
            y (torch.Tensor): Targets.
            n_folds (int): Number of folds.
            k_repeats (int): Number of repeats.
        """
        # Step 1: Build tree on full dataset and collect decreases
        self.fit(X, y)
        ds = sorted([d.item() for d in self.impurity_decreases])
        mids = [(ds[i] + ds[i + 1]) / 2 for i in range(len(ds) - 1)] if ds else []
        candidates = sorted([self.min_impurity_decrease] + mids + [float("inf")])

        # Initialize MSE tracker
        mse_per_cand = {cand: [] for cand in candidates}

        # Step 3: Repeated n-fold CV
        device = X.device
        n_samples = X.shape[0]
        fold_size = n_samples // n_folds

        # Calculate total iterations for progress bar
        total_iterations = k_repeats * n_folds * len(candidates)

        with tqdm(total=total_iterations, desc="Cross-validation") as pbar:
            for _ in range(k_repeats):  # pylint: disable=too-many-nested-blocks
                indices = torch.randperm(n_samples, device=device)
                for f in range(n_folds):
                    test_start = f * fold_size
                    test_end = (f + 1) * fold_size if f < n_folds - 1 else n_samples
                    test_idx = indices[test_start:test_end]
                    train_idx = torch.cat((indices[:test_start], indices[test_end:]))

                    X_train, y_train = X[train_idx], y[train_idx]
                    X_test, y_test = X[test_idx], y[test_idx]

                    # Build full tree on train once per fold
                    fold_tree = RegressionDecisionTree(
                        self.max_depth,
                        self.min_impurity_decrease,
                        self.look_ahead,
                        self.criterion,
                        self.leaf_value,
                    )
                    fold_tree.fit(X_train, y_train)

                    # Iterate through candidates in ascending order
                    # Since candidates are sorted, we can progressively prune the same tree
                    for cand in candidates:
                        if cand == float("inf") and fold_tree.root is not None:
                            # For infinite threshold, convert to a single leaf
                            # Only do this if the root is not already a leaf
                            if fold_tree.root.value is None:
                                if (
                                    fold_tree.root.sum_y is not None
                                    and fold_tree.root.n_samples is not None
                                ):
                                    fold_tree.root = fold_tree.Node(
                                        value=fold_tree.root.sum_y
                                        / fold_tree.root.n_samples,
                                        n_samples=fold_tree.root.n_samples,
                                        sum_y=fold_tree.root.sum_y,
                                    )
                        else:
                            fold_tree.prune(cand)

                        pred = fold_tree.predict(X_test)
                        mse = torch.mean((pred - y_test) ** 2).item()
                        mse_per_cand[cand].append(mse)

                        pbar.update(1)

        # Compute average MSE and select best
        avg_mse = {cand: sum(mses) / len(mses) for cand, mses in mse_per_cand.items()}
        best_cand = min(avg_mse.items(), key=lambda x: x[1])[0]

        # Step 4: Prune the full tree to the best candidate
        if best_cand == float("inf") and self.root is not None:
            if self.root.sum_y is not None and self.root.n_samples is not None:
                self.root = self.Node(
                    value=self.root.sum_y / self.root.n_samples,
                    n_samples=self.root.n_samples,
                    sum_y=self.root.sum_y,
                )
        else:
            self.prune(best_cand)

        # Clear collected decreases as they were for candidate generation
        self.impurity_decreases = []

    def export_text(self, feature_names=None, max_depth=10, spacing=3, decimals=2):
        """Build a text report showing the rules of the regression decision tree.

        Parameters
        ----------
        feature_names : list of str, optional
            Names of each of the features. If None, generic names will be used
            ("feature_0", "feature_1", ...).
        max_depth : int, default=10
            Only the first max_depth levels of the tree are exported.
            Truncated branches will be marked with "...".
        spacing : int, default=3
            Number of spaces between edges. The higher it is, the wider the result.
        decimals : int, default=2
            Number of decimal digits to display.

        Returns
        -------
        report : str
            Text summary of all the rules in the decision tree.
        """
        if self.root is None:
            return "This tree has not been fitted yet."

        # Generate default feature names if not provided
        if feature_names is None:
            # Count features from the tree structure
            n_features = self._count_features(self.root)
            feature_names = [f"feature_{i}" for i in range(n_features)]

        # Build the report recursively
        lines = []
        self._export_text_recursive(
            self.root, feature_names, max_depth, spacing, decimals, lines, depth=0
        )
        return "\n".join(lines)

    def _count_features(self, node):
        """Count the number of features used in the tree.

        Args:
            node (Node): Current node.

        Returns:
            int: Maximum feature index + 1.
        """
        if node is None or node.value is not None:
            return 0
        max_feature = node.feature
        left_features = self._count_features(node.left)
        right_features = self._count_features(node.right)
        return max(max_feature + 1, left_features, right_features)

    def _export_text_recursive(
        self, node, feature_names, max_depth, spacing, decimals, lines, depth=0
    ):
        """Recursively build the text representation of the tree.

        Args:
            node (Node): Current node.
            feature_names (list): Names of features.
            max_depth (int): Maximum depth to export.
            spacing (int): Number of spaces between edges.
            decimals (int): Number of decimal digits.
            lines (list): List to accumulate output lines.
            depth (int): Current depth in the tree.
        """
        # Build indentation: each level adds "|" + spacing spaces
        indent = ("|" + " " * spacing) * depth

        if depth >= max_depth:
            lines.append(f"{indent}|--- ...")
            return

        if node.value is not None:
            # Leaf node - display the predicted value
            value = node.value
            if isinstance(value, torch.Tensor):
                value = value.item()
            lines.append(f"{indent}|--- value: [{value:.{decimals}f}]")
            return

        # Split node - display the feature and threshold
        feature_name = feature_names[node.feature]
        threshold = node.threshold
        if isinstance(threshold, torch.Tensor):
            threshold = threshold.item()

        # Left branch (<=)
        lines.append(f"{indent}|--- {feature_name} <= {threshold:.{decimals}f}")
        self._export_text_recursive(
            node.left, feature_names, max_depth, spacing, decimals, lines, depth + 1
        )

        # Right branch (>)
        lines.append(f"{indent}|--- {feature_name} >  {threshold:.{decimals}f}")
        self._export_text_recursive(
            node.right, feature_names, max_depth, spacing, decimals, lines, depth + 1
        )
