import pytest


class TestExamples:
    """Using a simple class to group tests."""

    def test_assert_true_equals_true(self):
        assert True is True

    def test_assert_false_equals_false(self):
        assert False is False

    def test_assert_true_not_equals_false(self):
        assert True is not False

    def test_assert_false_not_equals_true(self):
        assert False is not True

    def test_assert_none_equals_none(self):
        assert None is None

    def test_assert_none_not_equals_value(self):
        assert None != 1

    def test_value_error_raised(self):
        with pytest.raises(ValueError):
            raise ValueError("This is a ValueError")
