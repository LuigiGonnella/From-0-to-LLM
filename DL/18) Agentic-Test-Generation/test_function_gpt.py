 
import pytest
from function_01 import racer_disqualified

# ------------------------------------------------------------
# INPUT VALIDATION ERRORS
# ------------------------------------------------------------

def test_invalid_times_not_list():
    with pytest.raises(ValueError):
        racer_disqualified("abc", [1,2,3], 1, [10])

def test_invalid_times_wrong_length():
    with pytest.raises(ValueError):
        racer_disqualified([1,2], [1,2,3], 1, [10])

def test_invalid_times_non_int():
    with pytest.raises(ValueError):
        racer_disqualified([1, "a", 3], [1,2,3], 1, [10])

def test_invalid_winner_times_not_list():
    with pytest.raises(ValueError):
        racer_disqualified([1,2,3], "abc", 1, [10])

def test_invalid_winner_times_wrong_length():
    with pytest.raises(ValueError):
        racer_disqualified([1,2,3], [1,2], 1, [10])

def test_invalid_winner_times_non_int():
    with pytest.raises(ValueError):
        racer_disqualified([1,2,3], [1, "x", 3], 1, [10])

def test_invalid_n_penalties_not_int():
    with pytest.raises(ValueError):
        racer_disqualified([1,2,3], [1,2,3], "a", [10])

def test_invalid_penalties_not_list():
    with pytest.raises(ValueError):
        racer_disqualified([1,2,3], [1,2,3], 1, "abc")

def test_invalid_penalties_non_int():
    with pytest.raises(ValueError):
        racer_disqualified([1,2,3], [1,2,3], 1, [10, "x"])

def test_invalid_n_penalties_mismatch():
    with pytest.raises(ValueError):
        racer_disqualified([1,2,3], [1,2,3], 3, [10,20])


# ------------------------------------------------------------
# NO DISQUALIFICATION CASES
# ------------------------------------------------------------

def test_no_disqualification_clean_run():
    assert not racer_disqualified([10, 20, 30], [10, 20, 30], 0, [])

def test_no_disqualification_small_penalties():
    assert not racer_disqualified([12, 22, 32], [10, 20, 30], 2, [5, 10])

def test_no_disqualification_exact_time_limit():
    assert not racer_disqualified([20, 40, 60], [10, 20, 30], 1, [0])


# ------------------------------------------------------------
# PENALTIES — TOTAL > 100
# ------------------------------------------------------------

def test_disqualified_total_penalties_over_100():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 3, [50, 40, 20])

def test_disqualified_total_penalties_exactly_101():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 2, [60, 41])


# ------------------------------------------------------------
# PENALTIES — SINGLE PENALTY > 100
# ------------------------------------------------------------

def test_disqualified_single_penalty_over_100():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 1, [101])

def test_disqualified_single_penalty_exactly_101():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 1, [101])


# ------------------------------------------------------------
# PENALTIES — TOO MANY PENALTIES
# ------------------------------------------------------------

def test_disqualified_too_many_penalties():
    assert racer_disqualified([1,2,3], [1,2,3], 6, [1,1,1,1,1,1])

def test_disqualified_too_many_penalties_with_low_values():
    assert racer_disqualified([1,2,3], [1,2,3], 7, [0,0,0,0,0,0,0])


# ------------------------------------------------------------
# TIME EXCEEDS LIMIT (> 2 × winner_time)
# ------------------------------------------------------------

def test_disqualified_time_exceeds_limit_first_event():
    assert racer_disqualified([21, 20, 30], [10, 20, 30], 0, [])

def test_disqualified_time_exceeds_limit_second_event():
    assert racer_disqualified([10, 41, 30], [10, 20, 30], 0, [])

def test_disqualified_time_exceeds_limit_third_event():
    assert racer_disqualified([10, 20, 61], [10, 20, 30], 0, [])


# ------------------------------------------------------------
# MULTIPLE DISQUALIFYING CONDITIONS
# ------------------------------------------------------------

def test_disqualified_multiple_reasons_penalty_and_time():
    assert racer_disqualified([50, 20, 30], [10, 20, 30], 2, [150, 10])

def test_disqualified_all_rules_triggered():
    assert racer_disqualified([100, 200, 300], [10, 20, 30], 10, [200]*10)

