"""Tests for the overlap-aware adaptive-strength schedule.

Checks:
* ``alpha_t`` is non-increasing in ``bar_s_t`` (monotonicity).
* ``alpha_t = alpha_max`` when ``s_t = 0`` at steady state.
* ``alpha_t = 0`` when ``s_t = 1`` at steady state.
* Fast-fail on invalid alpha_max or beta_s.
"""
from __future__ import annotations

import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from adaptive_ogp.schedule import OverlapAwareSchedule


def test_monotone_in_s():
    sched_low = OverlapAwareSchedule(alpha_max=0.5, beta_s=0.0)
    sched_high = OverlapAwareSchedule(alpha_max=0.5, beta_s=0.0)
    a_low = sched_low.update(0.0)
    a_high = sched_high.update(0.9)
    assert a_low >= a_high, "alpha_t must not increase with s_t"


def test_saturation_low():
    sched = OverlapAwareSchedule(alpha_max=0.5, beta_s=0.0)
    for _ in range(100):
        alpha = sched.update(0.0)
    assert abs(alpha - 0.5) < 1e-6


def test_saturation_high():
    sched = OverlapAwareSchedule(alpha_max=0.5, beta_s=0.0)
    for _ in range(100):
        alpha = sched.update(1.0)
    assert abs(alpha - 0.0) < 1e-6


def test_bad_alpha_max():
    with pytest.raises(AssertionError):
        OverlapAwareSchedule(alpha_max=-0.1)
    with pytest.raises(AssertionError):
        OverlapAwareSchedule(alpha_max=1.5)


def test_bad_beta_s():
    with pytest.raises(AssertionError):
        OverlapAwareSchedule(beta_s=1.0)
    with pytest.raises(AssertionError):
        OverlapAwareSchedule(beta_s=-0.1)


def test_out_of_range_s_rejected():
    sched = OverlapAwareSchedule(alpha_max=0.5)
    with pytest.raises(AssertionError):
        sched.update(-0.1)
    with pytest.raises(AssertionError):
        sched.update(1.2)
