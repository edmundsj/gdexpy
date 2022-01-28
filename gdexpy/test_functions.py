import pytest
import numpy as np
from numpy.testing import assert_allclose
from gdexpy.cellFunctions import clip_vector

def test_clip_vector_same_Q():
    target_vector = np.array([5, -2])
    direction_vector = 1/np.sqrt(2) * np.array([1, -1])

    desired_vector = 2 * np.array([1, -1])
    actual_vector = clip_vector(direction_vector, target_vector)

    assert_allclose(actual_vector, desired_vector)

def test_clip_vector_different_Q():
    target_vector = np.array([5, -2])
    direction_vector = np.array([-1, -1])

    desired_vector = np.array([-2, -2])
    actual_vector = clip_vector(direction_vector, target_vector)
    assert_allclose(actual_vector, desired_vector)

def test_clip_vector_left():
    target_vector = np.array([-1640, 0])
    direction_vector = np.array([-1, 1.2e-16])
    desired_vector = np.array([-1640, 0])
    actual_vector = clip_vector(direction_vector, target_vector)
    assert_allclose(actual_vector, desired_vector, atol=1e-7)

