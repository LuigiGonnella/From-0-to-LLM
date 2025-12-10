 
import pytest
from function_01 import racer_disqualified

# Test cases for disqualification due to total penalties > 100
def test_disqualified_total_penalties_over_100():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 3, [50, 40, 20]) == True

def test_disqualified_total_penalties_exactly_101():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 2, [60, 41]) == True

# Test cases for disqualification due to single penalty > 100
def test_disqualified_single_penalty_over_100():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 1, [101]) == True

def test_disqualified_single_penalty_exactly_101():
    assert racer_disqualified([0, 1, 2], [0, 1, 2], 1, [101]) == True
