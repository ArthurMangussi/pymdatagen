"""Regression tests for publicly selected imputation metrics."""

import pandas as pd
import pytest

from mdatagen.metrics.metric import EvaluateImputation


@pytest.mark.parametrize(
    'original,imputed,expected',
    [
        ([[1], [2]], [[1], [2]], 100.0),
        ([[1], [2]], [[3], [4]], 0.0),
        ([[1], [2]], [[1], [4]], 50.0),
        ([[1, 2], [3, 4]], [[1, 9], [3, 9]], 50.0),
        ([[1, 2], [3, 4]], [[1, 2], [3, 4]], 100.0),
        ([[1, 2, 3]], [[1, 9, 9]], 33.33),
        ([['cat', 'dog']], [['cat', 'bird']], 50.0),
    ],
)
def test_percentage_of_correct_prediction(original, imputed, expected):
    """PCP measures correctly reconstructed entries across all columns."""
    evaluator = EvaluateImputation(
        data_original=pd.DataFrame(original),
        data_imputed=pd.DataFrame(imputed),
        metric='percentage_of_correct_prediction',
    )
    assert evaluator.show() == expected


@pytest.mark.parametrize('shape', [(0, 0), (0, 2), (2, 0)])
def test_percentage_of_correct_prediction_empty(shape):
    """An empty set of entries has no defined correctness percentage."""
    original = pd.DataFrame(index=range(shape[0]), columns=range(shape[1]))
    evaluator = EvaluateImputation(original.copy(), original,
                                   metric='percentage_of_correct_prediction')
    with pytest.raises(ValueError, match='empty'):
        evaluator.show()
