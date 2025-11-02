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

    def __init__(self, max_depth, min_impurity_decrease, look_ahead=0):
        """Initialize the RegressionDecisionTree with hyperparameters.

        Args:
            max_depth (int): Maximum depth of the tree.
            min_impurity_decrease (float): Minimum impurity decrease required to split
                (lower bound).
            look_ahead (int, optional): Number of levels to look ahead when evaluating
                splits. Defaults to 0.
        """
        self.max_depth = max_depth
        self.min_impurity_decrease = min_impurity_decrease
        self.look_ahead = look_ahead
        self.root = None
        self.impurity_decreases = []

    def _compute_impurity(self, y):
        """Compute the impurity (total sum of squares) for a set of targets.

        Args:
            y (torch.Tensor): Target tensor.

        Returns:
            float: Impurity value.
        """
        n = len(y)
        if n == 0:
            return 0.0
        return torch.sum(y**2) - (torch.sum(y) ** 2) / n

    def _find_best_split(self, X, y, look_ahead=None):  # pylint: disable=invalid-name
        """Find the best feature and split point that maximizes impurity decrease.

        Args:
            X (torch.Tensor): Feature tensor of shape (n_samples, n_features).
            y (torch.Tensor): Target tensor of shape (n_samples,).
            look_ahead (int, optional): Look ahead levels for this call.
                Defaults to self.look_ahead.

        Returns:
            tuple: (best_feature, best_threshold, best_left_indices, best_right_indices,
                best_augmented_decrease, best_immediate_decrease)
                   or (None, None, None, None, -inf, -inf) if no valid split is found.
        """
        # pylint: disable=too-many-statements
        if look_ahead is None:
            look_ahead = self.look_ahead

        n_samples, n_features = X.shape
        if n_samples < 2:
            return None, None, None, None, float("-inf"), float("-inf")

        current_impurity = self._compute_impurity(y)

        if look_ahead == 0:
            # Vectorized implementation for look_ahead == 0
            feature_sorted_indices = torch.argsort(X, dim=0)
            y_sorted = y[feature_sorted_indices]
            X_sorted = torch.gather(X, 0, feature_sorted_indices)

            consecutive_same = X_sorted[1:, :] == X_sorted[:-1, :]

            cumsum_y = torch.cumsum(y_sorted, dim=0)
            cumsum_y2 = torch.cumsum(y_sorted**2, dim=0)

            left_sum_y = cumsum_y[:-1, :]
            left_sum_y2 = cumsum_y2[:-1, :]
            left_n = (
                torch.arange(1, n_samples, device=X.device)
                .unsqueeze(1)
                .expand(-1, n_features)
            )

            total_sum_y = cumsum_y[-1, :]
            total_sum_y2 = cumsum_y2[-1, :]
            right_sum_y = total_sum_y - left_sum_y
            right_sum_y2 = total_sum_y2 - left_sum_y2
            right_n = n_samples - left_n

            left_impurity = left_sum_y2 - (left_sum_y**2) / left_n
            right_impurity = right_sum_y2 - (right_sum_y**2) / right_n
            impurity_after_split = left_impurity + right_impurity

            decreases = current_impurity - impurity_after_split
            decreases[consecutive_same] = float("-inf")

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

            best_left_indices = sorted_indices[: best_split + 1]
            best_right_indices = sorted_indices[best_split + 1 :]

            return (
                best_feature,
                best_threshold,
                best_left_indices,
                best_right_indices,
                best_decrease,
                best_decrease,
            )

        # Looped implementation for look_ahead > 0
        best_feature = None
        best_threshold = None
        best_left_indices = None
        best_right_indices = None
        best_augmented = float("-inf")
        best_immediate = float("-inf")

        for feature in range(n_features):
            feature_values = X[:, feature]
            sorted_indices = torch.argsort(feature_values)
            sorted_feature_values = feature_values[sorted_indices]
            for split in range(1, n_samples):
                if sorted_feature_values[split - 1] == sorted_feature_values[split]:
                    continue

                left_indices = sorted_indices[:split]
                right_indices = sorted_indices[split:]

                X_left = X[left_indices]
                y_left = y[left_indices]
                X_right = X[right_indices]
                y_right = y[right_indices]

                left_imp = self._compute_impurity(y_left)
                right_imp = self._compute_impurity(y_right)
                immediate_after = left_imp + right_imp
                immediate_decrease = current_impurity - immediate_after

                left_best_aug = self._find_best_split(
                    X_left, y_left, look_ahead=look_ahead - 1
                )[4]
                if left_best_aug == float("-inf"):
                    left_best_aug = 0.0
                right_best_aug = self._find_best_split(
                    X_right, y_right, look_ahead=look_ahead - 1
                )[4]
                if right_best_aug == float("-inf"):
                    right_best_aug = 0.0

                augmented = immediate_decrease + left_best_aug + right_best_aug

                if augmented > best_augmented:
                    best_augmented = augmented
                    best_immediate = immediate_decrease
                    best_feature = feature
                    best_threshold = (
                        sorted_feature_values[split - 1] + sorted_feature_values[split]
                    ) / 2
                    best_left_indices = left_indices.clone()
                    best_right_indices = right_indices.clone()

        if best_feature is None:
            return None, None, None, None, float("-inf"), float("-inf")
        return (
            best_feature,
            best_threshold,
            best_left_indices,
            best_right_indices,
            best_augmented,
            best_immediate,
        )

    def _build_tree(self, X, y, depth):  # pylint: disable=invalid-name
        """Recursively build the regression decision tree.

        Args:
            X (torch.Tensor): Feature tensor of shape (n_samples, n_features).
            y (torch.Tensor): Target tensor of shape (n_samples,).
            depth (int): Current depth of the tree.

        Returns:
            Node: Root node of the constructed tree or subtree.
        """
        n_samples = X.shape[0]
        sum_y = torch.sum(y)

        if depth == self.max_depth or n_samples < 2:
            value = (
                sum_y / n_samples
                if n_samples > 0
                else torch.tensor(0.0, device=X.device)
            )
            return self.Node(value=value, n_samples=n_samples, sum_y=sum_y)

        (
            feature,
            threshold,
            left_indices,
            right_indices,
            augmented_decrease,
            immediate_decrease,
        ) = self._find_best_split(X, y)

        if augmented_decrease > self.min_impurity_decrease and feature is not None:
            left_X = X[left_indices]
            left_y = y[left_indices]
            right_X = X[right_indices]
            right_y = y[right_indices]

            left_node = self._build_tree(left_X, left_y, depth + 1)
            right_node = self._build_tree(right_X, right_y, depth + 1)

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

        value = (
            sum_y / n_samples if n_samples > 0 else torch.tensor(0.0, device=X.device)
        )
        return self.Node(value=value, n_samples=n_samples, sum_y=sum_y)

    def fit(self, X, y):  # pylint: disable=invalid-name
        """Build the regression decision tree using the provided data.

        Args:
            X (torch.Tensor): 2D tensor of shape (n_samples, n_features) with features.
            y (torch.Tensor): 1D tensor of shape (n_samples,) with targets.
        """
        self.impurity_decreases = []  # Reset for each fit
        self.root = self._build_tree(X, y, 0)

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

    def predict(self, X):  # pylint: disable=invalid-name
        """Make predictions for a batch of samples.

        Args:
            X (torch.Tensor): 2D tensor of shape (n_samples, n_features) with features.

        Returns:
            torch.Tensor: 1D tensor of shape (n_samples,) with predictions.
        """
        return torch.tensor(
            [self._predict_one(self.root, x) for x in X], device=X.device
        )

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
            for _ in range(k_repeats):
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
                        self.max_depth, self.min_impurity_decrease, self.look_ahead
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
