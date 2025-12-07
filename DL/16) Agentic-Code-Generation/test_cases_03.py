import pytest
from function_03 import computeMaxTime

# ---------------------
# CATEGORY A - EASY TRACKS
# ---------------------
def test_category_a_low_speed():
    # avg_speed <= 30, increase by 5%
    result = computeMaxTime(100, 25, 'A')
    assert result == pytest.approx(105.0, rel=1e-2)

def test_category_a_boundary_30():
    # avg_speed = 30, increase by 5%
    result = computeMaxTime(100, 30, 'A')
    assert result == pytest.approx(105.0, rel=1e-2)

def test_category_a_mid_speed():
    # 30 < avg_speed <= 35, increase by 10%
    result = computeMaxTime(100, 32, 'A')
    assert result == pytest.approx(110.0, rel=1e-2)

def test_category_a_boundary_35():
    # avg_speed = 35, increase by 10%
    result = computeMaxTime(100, 35, 'A')
    assert result == pytest.approx(110.0, rel=1e-2)

def test_category_a_high_speed():
    # avg_speed > 35, increase by 15%
    result = computeMaxTime(100, 40, 'A')
    assert result == pytest.approx(115.0, rel=1e-2)

# ---------------------
# CATEGORY B - NORMAL TRACKS
# ---------------------
def test_category_b_low_speed():
    # avg_speed <= 30, increase by 20%
    result = computeMaxTime(100, 28, 'B')
    assert result == pytest.approx(120.0, rel=1e-2)

def test_category_b_boundary_30():
    # avg_speed = 30, increase by 20%
    result = computeMaxTime(100, 30, 'B')
    assert result == pytest.approx(120.0, rel=1e-2)

def test_category_b_mid_speed():
    # 30 < avg_speed <= 35, increase by 25%
    result = computeMaxTime(100, 33, 'B')
    assert result == pytest.approx(125.0, rel=1e-2)

def test_category_b_boundary_35():
    # avg_speed = 35, increase by 25%
    result = computeMaxTime(100, 35, 'B')
    assert result == pytest.approx(125.0, rel=1e-2)

def test_category_b_high_speed():
    # avg_speed > 35, increase by 30%
    result = computeMaxTime(100, 45, 'B')
    assert result == pytest.approx(130.0, rel=1e-2)

# ---------------------
# CATEGORY C - HARD TRACKS
# ---------------------
def test_category_c_low_speed():
    # Always increase by 50%, regardless of speed
    result = computeMaxTime(100, 20, 'C')
    assert result == pytest.approx(150.0, rel=1e-2)

def test_category_c_mid_speed():
    result = computeMaxTime(100, 32, 'C')
    assert result == pytest.approx(150.0, rel=1e-2)

def test_category_c_high_speed():
    result = computeMaxTime(100, 50, 'C')
    assert result == pytest.approx(150.0, rel=1e-2)

# ---------------------
# ERROR CASES
# ---------------------
def test_invalid_track_type():
    # Invalid track_type should return 0
    result = computeMaxTime(100, 30, 'D')
    assert result == pytest.approx(0.0, rel=1e-2)

def test_invalid_track_type_lowercase():
    result = computeMaxTime(100, 30, 'a')
    assert result == pytest.approx(0.0, rel=1e-2)

def test_negative_winner_time():
    result = computeMaxTime(-100, 30, 'A')
    assert result == pytest.approx(0.0, rel=1e-2)

def test_negative_avg_speed():
    result = computeMaxTime(100, -30, 'A')
    assert result == pytest.approx(0.0, rel=1e-2)

def test_zero_winner_time():
    result = computeMaxTime(0, 30, 'A')
    assert result == pytest.approx(0.0, rel=1e-2)

def test_zero_avg_speed():
    result = computeMaxTime(100, 0, 'A')
    assert result == pytest.approx(0.0, rel=1e-2)

# ---------------------
# BOUNDARY CASES
# ---------------------
def test_min_winner_time():
    result = computeMaxTime(1, 30, 'A')
    assert result == pytest.approx(1.05, rel=1e-2)

def test_large_winner_time():
    result = computeMaxTime(1000, 40, 'B')
    assert result == pytest.approx(1300.0, rel=1e-2)

def test_decimal_winner_time():
    result = computeMaxTime(123.45, 30, 'C')
    assert result == pytest.approx(185.175, rel=1e-2)

def test_decimal_avg_speed():
    result = computeMaxTime(100, 30.5, 'A')
    assert result == pytest.approx(110.0, rel=1e-2)

# ---------------------
# EDGE CASES
# ---------------------
def test_speed_just_above_30():
    # Should use 10% increase for category A
    result = computeMaxTime(100, 30.01, 'A')
    assert result == pytest.approx(110.0, rel=1e-2)

def test_speed_just_above_35():
    # Should use 15% increase for category A
    result = computeMaxTime(100, 35.01, 'A')
    assert result == pytest.approx(115.0, rel=1e-2)

def test_very_high_speed():
    result = computeMaxTime(100, 100, 'A')
    assert result == pytest.approx(115.0, rel=1e-2)