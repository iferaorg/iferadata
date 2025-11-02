"""Tests for the Regression Decision Tree algorithm."""

import torch
import pytest

from algorithms.regression_decision_tree import RegressionDecisionTree


def test_basic_tree_construction():
    """Test basic tree construction and prediction."""
    # Create simple dataset
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    predictions = tree.predict(X)
    # Check that predictions are close to actual values
    assert predictions.shape == y.shape
    assert torch.allclose(predictions, y, atol=0.5)


def test_leaf_node_prediction():
    """Test prediction when tree is just a leaf (no splits)."""
    X = torch.tensor([[1.0], [1.0], [1.0]])
    y = torch.tensor([2.0, 2.0, 2.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    predictions = tree.predict(X)
    # All predictions should be the mean of y
    assert torch.allclose(predictions, torch.tensor([2.0, 2.0, 2.0]))


def test_max_depth_limit():
    """Test that max_depth is respected."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0], [7.0], [8.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])

    tree = RegressionDecisionTree(max_depth=1, min_impurity_decrease=0.0)
    tree.fit(X, y)

    # Count depth of tree
    def count_depth(node, current_depth=0):
        if node.value is not None:
            return current_depth
        return max(
            count_depth(node.left, current_depth + 1),
            count_depth(node.right, current_depth + 1),
        )

    depth = count_depth(tree.root)
    assert depth <= 1


def test_min_impurity_decrease():
    """Test that min_impurity_decrease prevents small splits."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 1.1, 3.0, 3.1])

    # Very high threshold should prevent any splits
    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=100.0)
    tree.fit(X, y)

    # Tree should be just a leaf node
    assert tree.root.value is not None


def test_compute_impurity():
    """Test impurity computation."""
    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)

    # Test with simple values
    y = torch.tensor([1.0, 2.0, 3.0])
    impurity = tree._compute_impurity(y)

    # Manual calculation: sum(y^2) - (sum(y))^2 / n
    # = (1 + 4 + 9) - (6)^2 / 3 = 14 - 12 = 2
    assert torch.isclose(torch.tensor(impurity), torch.tensor(2.0))

    # Test with empty tensor
    y_empty = torch.tensor([])
    impurity_empty = tree._compute_impurity(y_empty)
    assert impurity_empty == 0.0


def test_look_ahead_zero():
    """Test tree building with look_ahead=0."""
    X = torch.tensor([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0], [4.0, 4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0, look_ahead=0)
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape


def test_look_ahead_positive():
    """Test tree building with look_ahead > 0."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0, look_ahead=1)
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape


def test_copy_tree():
    """Test tree copying functionality."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    # Copy the tree
    copied_root = tree.copy_tree(tree.root)

    # Verify predictions are the same
    def predict_with_node(node, x):
        if node.value is not None:
            return node.value
        if x[node.feature] <= node.threshold:
            return predict_with_node(node.left, x)
        else:
            return predict_with_node(node.right, x)

    for x in X:
        orig_pred = predict_with_node(tree.root, x)
        copy_pred = predict_with_node(copied_root, x)
        assert torch.isclose(orig_pred, copy_pred)


def test_prune():
    """Test tree pruning functionality."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    # Store original structure info
    original_has_children = tree.root.left is not None

    # Prune with very high threshold - should convert all to leaves
    tree.prune(min_imp=1000.0)

    # Root should now be a leaf
    if original_has_children:
        # If there were children before, pruning should have converted to leaf
        assert tree.root.value is not None


def test_find_optimal_min_impurity_decrease():
    """Test optimal min_impurity_decrease finder."""
    # Create a dataset with clear structure
    torch.manual_seed(42)
    X = torch.randn(50, 2)
    y = X[:, 0] * 2 + X[:, 1] * 3 + torch.randn(50) * 0.1

    tree = RegressionDecisionTree(max_depth=5, min_impurity_decrease=0.0)
    tree.find_optimal_min_impurity_decrease(X, y, n_folds=3, k_repeats=2)

    # Tree should still make predictions after optimization
    predictions = tree.predict(X)
    assert predictions.shape == y.shape


def test_multiple_features():
    """Test tree with multiple features."""
    X = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    y = torch.tensor([1.0, 2.0, 2.0, 3.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape
    # Predictions should be somewhat close to actual values
    mse = torch.mean((predictions - y) ** 2)
    assert mse < 1.0


def test_impurity_decreases_tracking():
    """Test that impurity decreases are tracked correctly."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    # Should have tracked some decreases
    assert len(tree.impurity_decreases) > 0
    # All decreases should be non-negative
    for dec in tree.impurity_decreases:
        assert dec >= 0


def test_device_consistency():
    """Test that predictions are on the same device as inputs."""
    if torch.cuda.is_available():
        X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="cuda")
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    else:
        X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.device == X.device


def test_empty_splits():
    """Test handling of datasets that cannot be split."""
    # Single sample
    X = torch.tensor([[1.0]])
    y = torch.tensor([2.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert torch.isclose(predictions[0], y[0])


def test_identical_features():
    """Test handling of identical feature values."""
    X = torch.tensor([[1.0], [1.0], [1.0], [1.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    # Should create a leaf with mean value
    predictions = tree.predict(X)
    mean_y = torch.mean(y)
    assert torch.allclose(predictions, mean_y.expand(4))


def test_node_sample_counts():
    """Test that node sample counts are correct."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    # Root should have all samples
    assert tree.root.n_samples == 4


def test_sum_y_tracking():
    """Test that sum_y is tracked correctly."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    tree.fit(X, y)

    # Root sum_y should equal total sum
    assert torch.isclose(tree.root.sum_y, torch.sum(y))


def test_find_optimal_with_progress_bar():
    """Test that find_optimal works with tqdm progress bar."""
    # Create a simple dataset
    torch.manual_seed(123)
    X = torch.randn(30, 2)
    y = X[:, 0] + X[:, 1] + torch.randn(30) * 0.1

    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    # This should show progress bar during execution
    tree.find_optimal_min_impurity_decrease(X, y, n_folds=2, k_repeats=2)

    # Verify tree can still predict
    predictions = tree.predict(X)
    assert predictions.shape == y.shape
    # Check that predictions are reasonable (low MSE)
    mse = torch.mean((predictions - y) ** 2)
    assert mse < 10.0  # Should be reasonably accurate
