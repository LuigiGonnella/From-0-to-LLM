 
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

# Test cases for disqualification due to number of penalties > 5
def test_disqualified_number_of_penalties_over_5():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 6, [50, 40, 20])

def test_disqualified_number_of_penalties_exactly_6():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 6, [60, 41])
