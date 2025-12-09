"""
Unit tests for null model validation (Issue #70).

Tests NullModelValidator and NullModelResult classes.
"""

import pytest
import numpy as np
import pandas as pd
from types import SimpleNamespace

from kosmos.validation.null_model import NullModelValidator, NullModelResult


class TestNullModelResult:
    """Test NullModelResult dataclass."""

    def test_create_null_model_result(self):
        """Test creating NullModelResult with all required fields."""
        result = NullModelResult(
            observed_statistic=2.5,
            null_distribution=[0.5, 1.0, 1.5, 2.0, 2.5],
            permutation_p_value=0.03,
            null_percentile=95.0,
            passes_null_test=True,
            persists_in_noise=False,
            n_permutations=1000,
            shuffle_method='column',
            alpha=0.05
        )

        assert result.observed_statistic == 2.5
        assert result.permutation_p_value == 0.03
        assert result.passes_null_test == True
        assert result.persists_in_noise == False
        assert result.n_permutations == 1000

    def test_to_dict_serialization(self):
        """Test serialization to dictionary."""
        result = NullModelResult(
            observed_statistic=2.5,
            null_distribution=[0.5, 1.0, 1.5, 2.0, 2.5],
            permutation_p_value=0.03,
            null_percentile=95.0,
            passes_null_test=True,
            persists_in_noise=False,
            n_permutations=1000,
            shuffle_method='column',
            alpha=0.05,
            observed_effect_size=0.8
        )

        d = result.to_dict()

        assert isinstance(d, dict)
        assert d['observed_statistic'] == 2.5
        assert d['permutation_p_value'] == 0.03
        assert d['passes_null_test'] == True
        assert d['observed_effect_size'] == 0.8

    def test_from_dict_deserialization(self):
        """Test deserialization from dictionary."""
        data = {
            'observed_statistic': 2.5,
            'null_distribution': [0.5, 1.0, 1.5, 2.0, 2.5],
            'permutation_p_value': 0.03,
            'null_percentile': 95.0,
            'passes_null_test': True,
            'persists_in_noise': False,
            'n_permutations': 1000,
            'shuffle_method': 'column',
            'alpha': 0.05
        }

        result = NullModelResult.from_dict(data)

        assert result.observed_statistic == 2.5
        assert result.passes_null_test == True

    def test_is_valid_property(self):
        """Test is_valid property logic."""
        # Valid: passes null test, doesn't persist in noise
        result_valid = NullModelResult(
            observed_statistic=2.5,
            null_distribution=[1.0],
            permutation_p_value=0.01,
            null_percentile=99.0,
            passes_null_test=True,
            persists_in_noise=False,
            n_permutations=1000,
            shuffle_method='column',
            alpha=0.05
        )
        assert result_valid.is_valid == True

        # Invalid: fails null test
        result_fail_null = NullModelResult(
            observed_statistic=1.0,
            null_distribution=[1.0],
            permutation_p_value=0.2,
            null_percentile=50.0,
            passes_null_test=False,
            persists_in_noise=False,
            n_permutations=1000,
            shuffle_method='column',
            alpha=0.05
        )
        assert result_fail_null.is_valid == False

        # Invalid: persists in noise
        result_persists = NullModelResult(
            observed_statistic=1.5,
            null_distribution=[1.0],
            permutation_p_value=0.03,
            null_percentile=60.0,
            passes_null_test=True,
            persists_in_noise=True,
            n_permutations=1000,
            shuffle_method='column',
            alpha=0.05
        )
        assert result_persists.is_valid == False

    def test_get_summary_valid(self):
        """Test summary message for valid finding."""
        result = NullModelResult(
            observed_statistic=2.5,
            null_distribution=[1.0],
            permutation_p_value=0.03,
            null_percentile=97.0,
            passes_null_test=True,
            persists_in_noise=False,
            n_permutations=1000,
            shuffle_method='column',
            alpha=0.05
        )

        summary = result.get_summary()
        assert 'VALID' in summary
        assert '0.0300' in summary

    def test_get_summary_persists_in_noise(self):
        """Test summary message when finding persists in noise."""
        result = NullModelResult(
            observed_statistic=1.0,
            null_distribution=[1.0],
            permutation_p_value=0.5,
            null_percentile=50.0,
            passes_null_test=False,
            persists_in_noise=True,
            n_permutations=1000,
            shuffle_method='column',
            alpha=0.05
        )

        summary = result.get_summary()
        assert 'WARNING' in summary
        assert 'persists in noise' in summary


class TestNullModelValidatorInit:
    """Test NullModelValidator initialization."""

    def test_default_init(self):
        """Test default initialization."""
        validator = NullModelValidator()

        assert validator.n_permutations == 1000
        assert validator.alpha == 0.05
        assert validator.persistence_threshold == 0.5

    def test_custom_init(self):
        """Test custom initialization."""
        validator = NullModelValidator(
            n_permutations=500,
            alpha=0.01,
            random_seed=42,
            persistence_threshold=0.6
        )

        assert validator.n_permutations == 500
        assert validator.alpha == 0.01
        assert validator.persistence_threshold == 0.6

    def test_reproducibility_with_seed(self):
        """Test that results are reproducible with same seed."""
        finding = {
            'statistics': {
                'test_type': 't_test',
                'statistic': 2.0,
                'degrees_of_freedom': 30
            }
        }

        validator1 = NullModelValidator(n_permutations=100, random_seed=42)
        validator2 = NullModelValidator(n_permutations=100, random_seed=42)

        result1 = validator1.validate_finding(finding)
        result2 = validator2.validate_finding(finding)

        assert result1.permutation_p_value == result2.permutation_p_value


class TestShuffleMethods:
    """Test shuffle methods."""

    def test_shuffle_columns(self):
        """Test column shuffling breaks correlations."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50],
            'c': [100, 200, 300, 400, 500]
        })

        validator = NullModelValidator(random_seed=42)
        shuffled = validator.shuffle_columns(df, ['a', 'b'])

        # Values should be the same but order should be different
        assert set(shuffled['a'].values) == set(df['a'].values)
        assert set(shuffled['b'].values) == set(df['b'].values)
        # Column 'c' should be unchanged
        assert list(shuffled['c'].values) == list(df['c'].values)
        # At least one column should have different order
        assert not (list(shuffled['a'].values) == list(df['a'].values) and
                    list(shuffled['b'].values) == list(df['b'].values))

    def test_shuffle_rows(self):
        """Test row shuffling."""
        df = pd.DataFrame({
            'a': [1, 2, 3, 4, 5],
            'b': [10, 20, 30, 40, 50]
        })

        validator = NullModelValidator(random_seed=42)
        shuffled = validator.shuffle_rows(df)

        # Same values, same shape
        assert set(shuffled['a'].values) == set(df['a'].values)
        assert len(shuffled) == len(df)
        # Order should be different
        assert list(shuffled['a'].values) != list(df['a'].values)
        # Within each row, a and b should still correspond
        for _, row in shuffled.iterrows():
            assert row['b'] == row['a'] * 10

    def test_shuffle_labels(self):
        """Test label shuffling for group comparisons."""
        df = pd.DataFrame({
            'value': [1, 2, 3, 4, 5, 6],
            'group': ['A', 'A', 'A', 'B', 'B', 'B']
        })

        validator = NullModelValidator(random_seed=42)
        shuffled = validator.shuffle_labels(df, 'group')

        # Labels should have same values
        assert set(shuffled['group'].values) == set(df['group'].values)
        # Order should be different
        assert list(shuffled['group'].values) != list(df['group'].values)
        # Values should be unchanged
        assert list(shuffled['value'].values) == list(df['value'].values)

    def test_shuffle_residuals(self):
        """Test residual shuffling for regression."""
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 1.9, 3.2, 3.8, 5.1])

        validator = NullModelValidator(random_seed=42)
        y_new = validator.shuffle_residuals(y, y_pred)

        # New y should have similar variance
        assert len(y_new) == len(y)
        # Should be different from original
        assert not np.allclose(y_new, y)


class TestStatisticExtraction:
    """Test test statistic extraction from findings."""

    def test_extract_t_statistic(self):
        """Test extracting t-statistic."""
        finding = {'statistics': {'t_statistic': 2.5}}

        validator = NullModelValidator()
        stat = validator._extract_test_statistic(finding)

        assert stat == 2.5

    def test_extract_f_statistic(self):
        """Test extracting F-statistic."""
        finding = {'statistics': {'f_statistic': 4.2}}

        validator = NullModelValidator()
        stat = validator._extract_test_statistic(finding)

        assert stat == 4.2

    def test_extract_correlation(self):
        """Test extracting correlation coefficient."""
        finding = {'statistics': {'correlation': -0.85}}

        validator = NullModelValidator()
        stat = validator._extract_test_statistic(finding)

        assert stat == 0.85  # Absolute value

    def test_extract_generic_statistic(self):
        """Test extracting generic 'statistic' key."""
        finding = {'statistics': {'statistic': 3.0}}

        validator = NullModelValidator()
        stat = validator._extract_test_statistic(finding)

        assert stat == 3.0

    def test_extract_from_p_value(self):
        """Test extracting from p-value when no statistic available."""
        finding = {'statistics': {'p_value': 0.01}}

        validator = NullModelValidator()
        stat = validator._extract_test_statistic(finding)

        # Should convert p-value to z-score (approximately 2.58 for p=0.01)
        assert 2.5 < stat < 2.7

    def test_no_statistic_raises_error(self):
        """Test that missing statistic raises ValueError."""
        finding = {'statistics': {}}

        validator = NullModelValidator()

        with pytest.raises(ValueError, match="No test statistic found"):
            validator._extract_test_statistic(finding)


class TestPermutationPValue:
    """Test permutation p-value calculation."""

    def test_extreme_statistic_low_pvalue(self):
        """Test that extreme statistic gives low p-value."""
        validator = NullModelValidator()
        null_dist = np.array([1.0, 1.1, 1.2, 1.3, 1.4] * 200)  # 1000 values around 1.0
        observed = 5.0  # Much larger than null

        p_value = validator._calculate_permutation_pvalue(observed, null_dist)

        # Should be very small
        assert p_value < 0.01

    def test_typical_statistic_moderate_pvalue(self):
        """Test that typical statistic gives moderate p-value."""
        validator = NullModelValidator()
        null_dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)
        observed = 3.0  # Middle of distribution

        p_value = validator._calculate_permutation_pvalue(observed, null_dist)

        # Should be around 0.5
        assert 0.3 < p_value < 0.7

    def test_empty_null_distribution_returns_1(self):
        """Test empty null distribution returns p=1."""
        validator = NullModelValidator()
        null_dist = np.array([])
        observed = 2.0

        p_value = validator._calculate_permutation_pvalue(observed, null_dist)

        assert p_value == 1.0


class TestPersistenceDetection:
    """Test detection of findings that persist in noise."""

    def test_extreme_statistic_not_persistent(self):
        """Test that extreme statistics don't persist in noise."""
        validator = NullModelValidator()
        null_dist = np.array([1.0, 1.5, 2.0, 2.5, 3.0] * 200)  # IQR: ~1.5 to ~2.5
        observed = 10.0  # Way outside IQR

        persists = validator._check_persistence_in_noise(observed, null_dist)

        assert persists == False

    def test_typical_statistic_persists(self):
        """Test that typical statistics persist in noise."""
        validator = NullModelValidator()
        null_dist = np.array([1.0, 1.5, 2.0, 2.5, 3.0] * 200)
        observed = 2.0  # Right in the middle of null distribution

        persists = validator._check_persistence_in_noise(observed, null_dist)

        assert persists == True

    def test_boundary_case(self):
        """Test boundary case at IQR edge."""
        validator = NullModelValidator()
        null_dist = np.arange(0, 100)  # Uniform 0-99
        q25, q75 = np.percentile(null_dist, [25, 75])

        # Just inside IQR
        persists_inside = validator._check_persistence_in_noise(q25 + 1, null_dist)
        assert persists_inside == True

        # Just outside IQR
        persists_outside = validator._check_persistence_in_noise(q75 + 10, null_dist)
        assert persists_outside == False


class TestParametricNull:
    """Test parametric null distribution generation."""

    def test_t_distribution_for_ttest(self):
        """Test t-distribution is used for t-tests."""
        finding = {'statistics': {'test_type': 't_test', 'degrees_of_freedom': 30}}

        validator = NullModelValidator(n_permutations=100, random_seed=42)
        null_dist = validator._parametric_null(finding)

        # T-distribution with 30 df should have values mostly under 3
        assert len(null_dist) == 100
        assert np.mean(null_dist) < 3

    def test_f_distribution_for_anova(self):
        """Test F-distribution is used for ANOVA."""
        finding = {'statistics': {'test_type': 'anova', 'degrees_of_freedom': 50}}

        validator = NullModelValidator(n_permutations=100, random_seed=42)
        null_dist = validator._parametric_null(finding)

        # F-distribution values should be non-negative
        assert len(null_dist) == 100
        assert np.all(null_dist >= 0)

    def test_normal_distribution_default(self):
        """Test normal distribution is used as default."""
        finding = {'statistics': {'test_type': 'unknown'}}

        validator = NullModelValidator(n_permutations=100, random_seed=42)
        null_dist = validator._parametric_null(finding)

        # Absolute values of normal distribution
        assert len(null_dist) == 100
        assert np.all(null_dist >= 0)


class TestShuffleMethodDetermination:
    """Test automatic shuffle method selection."""

    def test_ttest_uses_label_shuffle(self):
        """Test t-test selects label shuffle."""
        finding = {'statistics': {'test_type': 't_test'}}

        validator = NullModelValidator()
        method = validator._determine_shuffle_method(finding)

        assert method == 'label'

    def test_anova_uses_label_shuffle(self):
        """Test ANOVA selects label shuffle."""
        finding = {'statistics': {'test_type': 'anova'}}

        validator = NullModelValidator()
        method = validator._determine_shuffle_method(finding)

        assert method == 'label'

    def test_correlation_uses_column_shuffle(self):
        """Test correlation selects column shuffle."""
        finding = {'statistics': {'test_type': 'correlation'}}

        validator = NullModelValidator()
        method = validator._determine_shuffle_method(finding)

        assert method == 'column'

    def test_regression_uses_residual_shuffle(self):
        """Test regression selects residual shuffle."""
        finding = {'statistics': {'test_type': 'linear_regression'}}

        validator = NullModelValidator()
        method = validator._determine_shuffle_method(finding)

        assert method == 'residual'

    def test_unknown_uses_column_default(self):
        """Test unknown test type uses column shuffle as default."""
        finding = {'statistics': {'test_type': 'exotic_test'}}

        validator = NullModelValidator()
        method = validator._determine_shuffle_method(finding)

        assert method == 'column'


class TestValidateFinding:
    """Test the main validate_finding method."""

    def test_valid_finding_passes(self):
        """Test that a valid finding with strong effect passes."""
        finding = {
            'statistics': {
                'test_type': 't_test',
                'statistic': 4.0,
                'p_value': 0.001,
                'effect_size': 0.9,
                'degrees_of_freedom': 50
            }
        }

        validator = NullModelValidator(n_permutations=100, random_seed=42)
        result = validator.validate_finding(finding)

        assert result.passes_null_test == True
        assert result.persists_in_noise == False
        assert result.is_valid == True

    def test_weak_finding_fails(self):
        """Test that a weak finding fails null test."""
        finding = {
            'statistics': {
                'test_type': 't_test',
                'statistic': 0.5,  # Very weak
                'p_value': 0.6,
                'degrees_of_freedom': 50
            }
        }

        validator = NullModelValidator(n_permutations=100, random_seed=42)
        result = validator.validate_finding(finding)

        # Weak statistic should fail
        assert result.permutation_p_value > 0.05 or result.persists_in_noise == True

    def test_missing_statistics_handled(self):
        """Test handling of finding without statistics."""
        finding = {'summary': 'No statistics here'}

        validator = NullModelValidator()
        result = validator.validate_finding(finding)

        # Should return invalid result with warning
        assert result.is_valid == False
        assert len(result.warnings) > 0

    def test_computation_time_tracked(self):
        """Test that computation time is tracked."""
        finding = {
            'statistics': {
                'test_type': 't_test',
                'statistic': 2.5,
                'degrees_of_freedom': 30
            }
        }

        validator = NullModelValidator(n_permutations=100)
        result = validator.validate_finding(finding)

        assert result.computation_time_seconds > 0


class TestBatchValidation:
    """Test batch validation methods."""

    def test_batch_validate(self):
        """Test validating multiple findings at once."""
        findings = [
            {'statistics': {'test_type': 't_test', 'statistic': 3.0, 'degrees_of_freedom': 30}},
            {'statistics': {'test_type': 't_test', 'statistic': 2.0, 'degrees_of_freedom': 30}},
            {'statistics': {'test_type': 't_test', 'statistic': 1.0, 'degrees_of_freedom': 30}}
        ]

        validator = NullModelValidator(n_permutations=50, random_seed=42)
        results = validator.batch_validate(findings)

        assert len(results) == 3
        assert all(isinstance(r, NullModelResult) for r in results)

    def test_validation_statistics(self):
        """Test aggregate validation statistics."""
        findings = [
            {'statistics': {'test_type': 't_test', 'statistic': 4.0, 'degrees_of_freedom': 30}},
            {'statistics': {'test_type': 't_test', 'statistic': 3.5, 'degrees_of_freedom': 30}},
            {'statistics': {'test_type': 't_test', 'statistic': 0.5, 'degrees_of_freedom': 30}}
        ]

        validator = NullModelValidator(n_permutations=50, random_seed=42)
        results = validator.batch_validate(findings)
        stats = validator.get_validation_statistics(results)

        assert stats['count'] == 3
        assert 'valid_count' in stats
        assert 'valid_rate' in stats
        assert 'mean_p_value' in stats
        assert 0 <= stats['valid_rate'] <= 1


class TestPercentileCalculation:
    """Test percentile calculation."""

    def test_high_percentile_for_extreme(self):
        """Test extreme value gives high percentile."""
        validator = NullModelValidator()
        null_dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)
        observed = 10.0  # Higher than all null values

        percentile = validator._calculate_percentile(observed, null_dist)

        assert percentile == 100.0

    def test_low_percentile_for_small(self):
        """Test small value gives low percentile."""
        validator = NullModelValidator()
        null_dist = np.array([1.0, 2.0, 3.0, 4.0, 5.0] * 200)
        observed = 0.1  # Smaller than all null values

        percentile = validator._calculate_percentile(observed, null_dist)

        assert percentile == 0.0

    def test_middle_percentile(self):
        """Test middle value gives ~50 percentile."""
        validator = NullModelValidator()
        null_dist = np.arange(1, 101)  # 1 to 100
        observed = 50

        percentile = validator._calculate_percentile(observed, null_dist)

        # Should be around 50%
        assert 45 < percentile < 55


class TestDistributionSummary:
    """Test distribution summarization."""

    def test_summarize_distribution_returns_percentiles(self):
        """Test that summary returns 5 percentiles."""
        validator = NullModelValidator()
        dist = np.arange(100)

        summary = validator._summarize_distribution(dist)

        assert len(summary) == 5
        # Should be approximately [5, 25, 50, 75, 95] for uniform 0-99
        assert summary[0] < summary[1] < summary[2] < summary[3] < summary[4]

    def test_summarize_empty_distribution(self):
        """Test summarizing empty distribution."""
        validator = NullModelValidator()
        dist = np.array([])

        summary = validator._summarize_distribution(dist)

        assert summary == [0.0, 0.0, 0.0, 0.0, 0.0]

    def test_summarize_none_distribution(self):
        """Test summarizing None distribution."""
        validator = NullModelValidator()

        summary = validator._summarize_distribution(None)

        assert summary == [0.0, 0.0, 0.0, 0.0, 0.0]
