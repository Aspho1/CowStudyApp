# tests/test_utils.py
import pytest
from cowstudyapp.utils import round_to_interval

def test_round_to_interval():
    # Test exact intervals
    assert round_to_interval(300) == 300
    assert round_to_interval(600) == 600
    
    # Test rounding down
    assert round_to_interval(301) == 300
    assert round_to_interval(449) == 300
    
    # Test rounding up
    assert round_to_interval(450) == 600
    assert round_to_interval(599) == 600
    
    # Test with different interval
    assert round_to_interval(61, interval=60) == 60
    assert round_to_interval(89, interval=60) == 60
    assert round_to_interval(90, interval=60) == 120

test_round_to_interval()