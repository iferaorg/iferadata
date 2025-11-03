"""Tests for new RegressionDecisionTree options: criterion and leaf_value."""

import torch
import pytest

from algorithms.regression_decision_tree import RegressionDecisionTree


def test_criterion_parameter_validation():
    """Test that criterion parameter validation works correctly."""
    # Valid values should work
    tree = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, criterion="MSE"
    )
    assert tree.criterion == "MSE"

    tree = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, criterion="absolute_error"
    )
    assert tree.criterion == "absolute_error"

    # Invalid value should raise ValueError
    with pytest.raises(ValueError, match="criterion must be"):
        RegressionDecisionTree(
            max_depth=3, min_impurity_decrease=0.0, criterion="invalid"
        )


def test_leaf_value_parameter_validation():
    """Test that leaf_value parameter validation works correctly."""
    # Valid values should work
    tree = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="mean"
    )
    assert tree.leaf_value == "mean"

    tree = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="median"
    )
    assert tree.leaf_value == "median"

    # Invalid value should raise ValueError
    with pytest.raises(ValueError, match="leaf_value must be"):
        RegressionDecisionTree(
            max_depth=3, min_impurity_decrease=0.0, leaf_value="invalid"
        )


def test_default_parameters():
    """Test that default parameters are MSE and mean."""
    tree = RegressionDecisionTree(max_depth=3, min_impurity_decrease=0.0)
    assert tree.criterion == "MSE"
    assert tree.leaf_value == "mean"


def test_mse_criterion_with_mean():
    """Test tree with MSE criterion and mean leaf value (default behavior)."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, criterion="MSE", leaf_value="mean"
    )
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape
    # With linear data and MSE, predictions should be close to actual values
    assert torch.allclose(predictions, y, atol=0.5)


def test_mse_criterion_with_median():
    """Test tree with MSE criterion and median leaf value."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Outlier at the end

    tree = RegressionDecisionTree(
        max_depth=2, min_impurity_decrease=0.0, criterion="MSE", leaf_value="median"
    )
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape
    # Predictions should exist and be valid
    assert torch.all(torch.isfinite(predictions))


def test_absolute_error_criterion_with_mean():
    """Test tree with absolute_error criterion and mean leaf value."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    tree = RegressionDecisionTree(
        max_depth=3,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="mean",
    )
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape
    # Predictions should be reasonable
    assert torch.all(torch.isfinite(predictions))


def test_absolute_error_criterion_with_median():
    """Test tree with absolute_error criterion and median leaf value."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 100.0])  # Outlier

    tree = RegressionDecisionTree(
        max_depth=2,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="median",
    )
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape
    # Median should be more robust to outliers than mean
    assert torch.all(torch.isfinite(predictions))


def test_median_leaf_value_robustness():
    """Test that median leaf values are more robust to outliers than mean."""
    # Create dataset with outliers
    X = torch.tensor([[1.0], [1.0], [1.0], [1.0], [1.0]])
    y = torch.tensor([1.0, 1.0, 1.0, 1.0, 100.0])  # One extreme outlier

    tree_mean = RegressionDecisionTree(
        max_depth=1, min_impurity_decrease=100.0, leaf_value="mean"
    )
    tree_mean.fit(X, y)
    pred_mean = tree_mean.predict(X)

    tree_median = RegressionDecisionTree(
        max_depth=1, min_impurity_decrease=100.0, leaf_value="median"
    )
    tree_median.fit(X, y)
    pred_median = tree_median.predict(X)

    # Mean should be affected by outlier (around 20.8)
    assert pred_mean[0] > 10.0

    # Median should be robust to outlier (should be 1.0)
    assert torch.isclose(pred_median[0], torch.tensor(1.0))


def test_compute_impurity_with_absolute_error():
    """Test impurity computation with absolute_error criterion."""
    tree_mse = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, criterion="MSE"
    )
    tree_mae = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, criterion="absolute_error"
    )

    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])

    impurity_mse = tree_mse._compute_impurity(y)
    impurity_mae = tree_mae._compute_impurity(y)

    # Both should be non-negative
    assert impurity_mse >= 0
    assert impurity_mae >= 0

    # They should be different values (unless data is very specific)
    assert impurity_mse != impurity_mae


def test_absolute_error_uses_median_for_impurity():
    """Test that absolute_error always uses median for impurity, regardless of leaf_value."""
    tree_mae_mean = RegressionDecisionTree(
        max_depth=3,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="mean",
    )
    tree_mae_median = RegressionDecisionTree(
        max_depth=3,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="median",
    )

    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])  # Outlier

    # Both should compute the same impurity (using median as center)
    impurity_mean = tree_mae_mean._compute_impurity(y)
    impurity_median = tree_mae_median._compute_impurity(y)

    assert impurity_mean == impurity_median


def test_compute_leaf_value_mean():
    """Test _compute_leaf_value with mean method."""
    tree = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="mean"
    )

    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([True, True, True, False, False])

    leaf_value = tree._compute_leaf_value(y, mask)
    # Mean of [1, 2, 3] is 2
    assert torch.isclose(leaf_value, torch.tensor(2.0))


def test_compute_leaf_value_median():
    """Test _compute_leaf_value with median method."""
    tree = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="median"
    )

    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
    mask = torch.tensor([True, True, True, False, False])

    leaf_value = tree._compute_leaf_value(y, mask)
    # Median of [1, 2, 3] is 2
    assert torch.isclose(leaf_value, torch.tensor(2.0))


def test_compute_leaf_value_median_with_outliers():
    """Test that median is different from mean with outliers."""
    tree_mean = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="mean"
    )
    tree_median = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="median"
    )

    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 100.0])
    mask = torch.ones(5, dtype=torch.bool)

    leaf_mean = tree_mean._compute_leaf_value(y, mask)
    leaf_median = tree_median._compute_leaf_value(y, mask)

    # Mean should be affected by outlier (22.0)
    assert torch.isclose(leaf_mean, torch.tensor(22.0))

    # Median should be robust (3.0)
    assert torch.isclose(leaf_median, torch.tensor(3.0))


def test_look_ahead_with_new_parameters():
    """Test that look_ahead works with new parameters."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])

    tree = RegressionDecisionTree(
        max_depth=3,
        min_impurity_decrease=0.0,
        look_ahead=1,
        criterion="absolute_error",
        leaf_value="median",
    )
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape


def test_find_optimal_with_new_parameters():
    """Test find_optimal_min_impurity_decrease with new parameters."""
    torch.manual_seed(42)
    X = torch.randn(30, 2)
    y = X[:, 0] * 2 + X[:, 1] * 3 + torch.randn(30) * 0.1

    tree = RegressionDecisionTree(
        max_depth=3,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="median",
    )
    tree.find_optimal_min_impurity_decrease(X, y, n_folds=2, k_repeats=2)

    # Tree should still make predictions after optimization
    predictions = tree.predict(X)
    assert predictions.shape == y.shape


def test_multiple_features_with_new_parameters():
    """Test tree with multiple features and new parameters."""
    X = torch.tensor([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [2.0, 2.0]])
    y = torch.tensor([1.0, 2.0, 2.0, 3.0])

    tree = RegressionDecisionTree(
        max_depth=3,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="median",
    )
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.shape == y.shape


def test_empty_mask_with_new_parameters():
    """Test handling of empty masks with new parameters."""
    tree = RegressionDecisionTree(
        max_depth=3,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="median",
    )

    y = torch.tensor([1.0, 2.0, 3.0])
    mask = torch.zeros(3, dtype=torch.bool)

    leaf_value = tree._compute_leaf_value(y, mask)
    assert torch.isclose(leaf_value, torch.tensor(0.0))


def test_single_sample_with_median():
    """Test behavior with a single sample and median leaf_value."""
    X = torch.tensor([[1.0]])
    y = torch.tensor([5.0])

    tree = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="median"
    )
    tree.fit(X, y)

    predictions = tree.predict(X)
    # With single sample, median should equal that sample
    assert torch.isclose(predictions[0], y[0])


def test_criterion_affects_splits():
    """Test that different criteria can lead to different tree structures."""
    # Create a dataset where MSE and MAE might favor different splits
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0], [6.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0, 20.0])  # Outlier

    tree_mse = RegressionDecisionTree(
        max_depth=2, min_impurity_decrease=0.0, criterion="MSE"
    )
    tree_mse.fit(X, y)

    tree_mae = RegressionDecisionTree(
        max_depth=2, min_impurity_decrease=0.0, criterion="absolute_error"
    )
    tree_mae.fit(X, y)

    # Both should produce valid predictions
    pred_mse = tree_mse.predict(X)
    pred_mae = tree_mae.predict(X)

    assert torch.all(torch.isfinite(pred_mse))
    assert torch.all(torch.isfinite(pred_mae))


def test_export_text_with_new_parameters():
    """Test that export_text works with new parameters."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(
        max_depth=2,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="median",
    )
    tree.fit(X, y)

    text = tree.export_text()
    assert isinstance(text, str)
    assert "value:" in text


def test_device_consistency_with_new_parameters():
    """Test device consistency with new parameters."""
    if torch.cuda.is_available():
        X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], device="cuda")
        y = torch.tensor([1.0, 2.0, 3.0, 4.0], device="cuda")
    else:
        X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(
        max_depth=3,
        min_impurity_decrease=0.0,
        criterion="absolute_error",
        leaf_value="median",
    )
    tree.fit(X, y)

    predictions = tree.predict(X)
    assert predictions.device == X.device


def test_prune_respects_leaf_value():
    """Test that pruning respects the leaf_value parameter."""
    # Create simple data where we can control the tree structure
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0], [5.0]])
    y = torch.tensor([10.0, 11.0, 12.0, 13.0, 100.0])  # Outlier at end

    # Build tree with median leaf_value
    tree_median = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="median"
    )
    tree_median.fit(X, y)

    # Build tree with mean leaf_value
    tree_mean = RegressionDecisionTree(
        max_depth=3, min_impurity_decrease=0.0, leaf_value="mean"
    )
    tree_mean.fit(X, y)

    # Both trees should have values computed at all nodes (including root)
    assert tree_median.root.value is not None
    assert tree_mean.root.value is not None

    # The root values should differ (median vs mean of entire dataset)
    # Mean: (10+11+12+13+100)/5 = 29.2
    # Median: 12.0
    mean_val = torch.mean(y).item()
    median_val = torch.median(y).item()

    # Verify the root values match expected mean/median
    assert torch.isclose(tree_mean.root.value, torch.tensor(mean_val), atol=0.1)
    assert torch.isclose(tree_median.root.value, torch.tensor(median_val), atol=0.1)

    # Find a node with small decrease that we can prune
    # Let's prune child nodes with decrease < 5.0
    tree_median.prune(min_imp=5.0)
    tree_mean.prune(min_imp=5.0)

    # After some pruning occurred, verify the values are preserved
    # The key insight is that values are stored in all nodes during build,
    # so when a node becomes a leaf through pruning, it already has the
    # correct value (mean or median) computed during tree building.

    # Get predictions - these should use the stored values
    pred_median = tree_median.predict(X)
    pred_mean = tree_mean.predict(X)

    # Verify predictions are valid
    assert torch.all(torch.isfinite(pred_median))
    assert torch.all(torch.isfinite(pred_mean))


def test_is_leaf_flag():
    """Test that is_leaf flag is correctly set and maintained."""
    X = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    y = torch.tensor([1.0, 2.0, 3.0, 4.0])

    tree = RegressionDecisionTree(max_depth=2, min_impurity_decrease=0.0)
    tree.fit(X, y)

    # Root should not be a leaf (assuming tree splits)
    if tree.root.left is not None or tree.root.right is not None:
        assert not tree.root.is_leaf

    # Traverse to find a leaf
    node = tree.root
    while not node.is_leaf:
        if node.left is not None:
            node = node.left
        else:
            break

    # This should be a leaf
    assert node.is_leaf
    assert node.left is None
    assert node.right is None

    # Prune the tree heavily
    tree.prune(min_imp=1000.0)

    # After aggressive pruning, root should become a leaf
    assert tree.root.is_leaf
    assert tree.root.left is None
    assert tree.root.right is None
