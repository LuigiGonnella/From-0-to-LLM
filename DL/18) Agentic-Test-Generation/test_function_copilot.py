 
import pytest
from function_01 import racer_disqualified

# Test cases for disqualification due to total penalties > 100
def test_disqualified_total_penalties_over_100():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 3, [50, 40, 20])

def test_disqualified_total_penalties_exactly_101():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 2, [60, 41])

# Test cases for disqualification due to single penalty > 100
def test_disqualified_single_penalty_over_100():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 1, [101])

def test_disqualified_single_penalty_exactly_101():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 1, [101])

# Test cases for disqualification due to n_penalties > 5
def test_disqualified_more_than_5_penalties():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 6, [10, 10, 10, 10, 10, 10])

def test_disqualified_exactly_6_penalties():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 6, [5, 5, 5, 5, 5, 5])

# Test cases for disqualification due to excessive time (> 2x winner time)
def test_disqualified_excessive_time_first_event():
    assert racer_disqualified([5, 1, 2], [2, 1, 2], 1, [10])

def test_disqualified_excessive_time_second_event():
    assert racer_disqualified([0, 7, 2], [0, 3, 2], 1, [10])

def test_disqualified_excessive_time_third_event():
    assert racer_disqualified([0, 1, 9], [0, 1, 4], 1, [10])

def test_disqualified_time_exactly_2x_winner_time_plus_1():
    assert racer_disqualified([0, 1, 7], [0, 1, 3], 1, [10])

# Test cases for NOT disqualified (boundary cases)
def test_not_disqualified_total_penalties_exactly_100():
    assert not racer_disqualified([0, 1, 2], [0, 1, 2], 2, [50, 50])

def test_not_disqualified_single_penalty_exactly_100():
    assert not racer_disqualified([0, 1, 2], [0, 1, 2], 1, [100])

def test_not_disqualified_exactly_5_penalties():
    assert not racer_disqualified([0, 1, 2], [0, 1, 2], 5, [10, 10, 10, 10, 10])

def test_not_disqualified_time_exactly_2x_winner_time():
    assert not racer_disqualified([0, 2, 4], [0, 1, 2], 1, [10])

def test_not_disqualified_all_valid():
    assert not racer_disqualified([0, 1, 2], [0, 1, 2], 3, [20, 30, 40])

# Test cases for input validation errors
def test_invalid_times_not_list():
    with pytest.raises(ValueError, match="times must be a list of three integers"):
        racer_disqualified((0, 1, 2), [0, 1, 2], 1, [10])

def test_invalid_times_wrong_length():
    with pytest.raises(ValueError, match="times must be a list of three integers"):
        racer_disqualified([0, 1], [0, 1, 2], 1, [10])

def test_invalid_times_not_integers():
    with pytest.raises(ValueError, match="times must be a list of three integers"):
        racer_disqualified([0.5, 1, 2], [0, 1, 2], 1, [10])

def test_invalid_winner_times_not_list():
    with pytest.raises(ValueError, match="winner_times must be a list of three integers"):
        racer_disqualified([0, 1, 2], (0, 1, 2), 1, [10])

def test_invalid_winner_times_wrong_length():
    with pytest.raises(ValueError, match="winner_times must be a list of three integers"):
        racer_disqualified([0, 1, 2], [0, 1], 1, [10])

def test_invalid_n_penalties_not_int():
    with pytest.raises(ValueError, match="n_penalties must be an integer"):
        racer_disqualified([0, 1, 2], [0, 1, 2], 1.5, [10])

def test_invalid_penalties_not_list():
    with pytest.raises(ValueError, match="penalties must be a list of integers"):
        racer_disqualified([0, 1, 2], [0, 1, 2], 1, (10,))

def test_invalid_penalties_not_integers():
    with pytest.raises(ValueError, match="penalties must be a list of integers"):
        racer_disqualified([0, 1, 2], [0, 1, 2], 1, [10.5])

def test_invalid_n_penalties_mismatch():
    with pytest.raises(ValueError, match="n_penalties must match the length of the penalties list"):
        racer_disqualified([0, 1, 2], [0, 1, 2], 2, [10])

# Edge cases
def test_zero_penalties():
    assert not racer_disqualified([0, 1, 2], [0, 1, 2], 0, [])

def test_zero_winner_times():
    assert not racer_disqualified([0, 0, 0], [0, 0, 0], 1, [10])

def test_negative_penalties():
    assert not racer_disqualified([0, 1, 2], [0, 1, 2], 2, [-10, -20])

