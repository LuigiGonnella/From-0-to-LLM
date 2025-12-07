import pytest
from function_02 import computeFee

# ---------------------
# OFFER APPLIES
# ---------------------
def test_min_group_offer():
    result = computeFee(10, 2, 1, 1)
    assert result == pytest.approx(10.0, rel=1e-2)

def test_max_under15_discounted():
    result = computeFee(20, 5, 1, 4)
    assert result == pytest.approx(20.0, rel=1e-2)  # Only adult pays

def test_mixed_ages():
    # 3 passengers: 1 adult, 1 teen, 1 child → 2 pay
    result = computeFee(15, 3, 1, 1)
    assert result == pytest.approx(30.0, rel=1e-2)

def test_all_over18():
    result = computeFee(12, 3, 3, 0)
    assert result == pytest.approx(36.0, rel=1e-2)

# ---------------------
# OFFER DOES NOT APPLY
# ---------------------
def test_no_adult():
    result = computeFee(10, 3, 0, 3)
    assert result == pytest.approx(30.0, rel=1e-2)  # All pay

def test_group_too_small_for_offer():
    result = computeFee(10, 1, 1, 0)
    assert result == pytest.approx(10.0, rel=1e-2)  # Full price

# ---------------------
# ERROR CASES
# ---------------------
def test_group_too_large():
    with pytest.raises(ValueError):
        computeFee(10, 6, 2, 2)

def test_inconsistent_numbers():
    with pytest.raises(ValueError):
        computeFee(10, 4, 1, 5)  # over18 + under15 > passengers

def test_negative_values():
    with pytest.raises(ValueError):
        computeFee(10, 3, -1, 1)

def test_negative_base_price():
    with pytest.raises(ValueError):
        computeFee(-5, 3, 1, 1)

# ---------------------
# BOUNDARY CASES
# ---------------------
def test_min_group_size():
    result = computeFee(10, 2, 1, 0)
    assert result == pytest.approx(20.0, rel=1e-2)

def test_max_group_size():
    result = computeFee(10, 5, 1, 0)
    assert result == pytest.approx(50.0, rel=1e-2)

def test_one_under15():
    result = computeFee(10, 3, 1, 1)
    assert result == pytest.approx(20.0, rel=1e-2)