import numpy as np
from SPI2py.analysis.distance import minimum_distance_linesegment_linesegment

# TODO Add tests for all negative coordinates and mixed coordinates

# AB is a Point and CD is a Point


def test_point_point_overlap():

    a = np.array([0., 0., 0.])
    b = np.array([0., 0., 0.])
    c = np.array([0., 0., 0.])
    d = np.array([0., 0., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 0.0)


def test_point_point_overlap_positive_coords():

    a = np.array([1., 1., 1.])
    b = np.array([1., 1., 1.])
    c = np.array([1., 1., 1.])
    d = np.array([1., 1., 1.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 0.0)


def test_point_point_overlap_negative_coords():

    a = np.array([-1., -1., -1.])
    b = np.array([-1., -1., -1.])
    c = np.array([-1., -1., -1.])
    d = np.array([-1., -1., -1.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 0.0)


def test_point_point_overlap_mixed_coords():

    a = np.array([-1., 1., 0.])
    b = np.array([-1., 1., 0.])
    c = np.array([-1., 1., 0.])
    d = np.array([-1., 1., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 0.0)



def test_point_point_non_overlap():

    a = np.array([0., 0., 0.])
    b = np.array([0., 0., 0.])
    c = np.array([1., 0., 0.])
    d = np.array([1., 0., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 1.0)


# AB is a Point and CD is a Line Segment


def test_point_linesegment_overlap_1():

    a = np.array([0., 0., 0.])
    b = np.array([0., 0., 0.])
    c = np.array([0., 0., 0.])
    d = np.array([1., 0., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 0.0)


def test_point_linesegment_non_overlap_1():

    a = np.array([0., 0., 0.])
    b = np.array([0., 0., 0.])
    c = np.array([0., 0., 1.])
    d = np.array([1., 0., 1.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 1.0)


# AB is a Line Segment and CD is a Point

def test_linesegment_point_overlap_2():

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([0., 0., 0.])
    d = np.array([0., 0., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 0.0)


def test_linesegment_point_non_overlap_2():

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([1., 0., 1.])
    d = np.array([1., 0., 1.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 1.0)


# AB is a Line Segment and CD is a Line Segment


def test_fully_overlapping():

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([0., 0., 0.])
    d = np.array([1., 0., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 0.0)


def test_partially_overlapping():

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([0.5, 0., 0.])
    d = np.array([1.5, 0., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 0.0)


def test_parallel_horizontal_within_range():

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([0., 0., 1.])
    d = np.array([1., 0., 1.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 1.0)


def test_parallel_horizontal_out_of_range():

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([2., 0., 1.])
    d = np.array([3., 0., 1.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    expected_dist = np.linalg.norm(c-b)

    assert np.isclose(dist, expected_dist)


def test_parallel_horizontal_along_same_axis():

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([2., 0., 0.])
    d = np.array([3., 0., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 1.0)


def test_parallel_vertical_within_range():

        a = np.array([0., 0., 0.])
        b = np.array([0., 1., 0.])
        c = np.array([0., 0., 1.])
        d = np.array([0., 1., 1.])

        dist = minimum_distance_linesegment_linesegment(a, b, c, d)

        assert np.isclose(dist, 1.0)


def test_parallel_vertical_out_of_range():

    a = np.array([0., 0., 0.])
    b = np.array([0., 1., 0.])
    c = np.array([1., 2., 0.])
    d = np.array([1., 3., 0.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    expected_dist = np.linalg.norm(c - b)

    assert np.isclose(dist, expected_dist)


def test_skew():

    a = np.array([0., 0., 0.])
    b = np.array([1., 0., 0.])
    c = np.array([0., 0., 2.])
    d = np.array([1., 0., 1.])

    dist = minimum_distance_linesegment_linesegment(a, b, c, d)

    assert np.isclose(dist, 1.0)


# Autograd tests


