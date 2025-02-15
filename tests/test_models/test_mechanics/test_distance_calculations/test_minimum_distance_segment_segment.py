import jax.numpy as jnp
from SPI2py.models.mechanics.distance import minimum_distances_segments_segments

# AB is a Point and CD is a Point


def test_point_point_overlap():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([0., 0., 0.])
    c = jnp.array([0., 0., 0.])
    d = jnp.array([0., 0., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 0.0)


def test_point_point_overlap_positive_coordinates():

    a = jnp.array([1., 1., 1.])
    b = jnp.array([1., 1., 1.])
    c = jnp.array([1., 1., 1.])
    d = jnp.array([1., 1., 1.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 0.0)


def test_point_point_overlap_negative_coordinates():

    a = jnp.array([-1., -1., -1.])
    b = jnp.array([-1., -1., -1.])
    c = jnp.array([-1., -1., -1.])
    d = jnp.array([-1., -1., -1.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 0.0)


def test_point_point_overlap_mixed_coordinates():

    a = jnp.array([-1., 1., 0.])
    b = jnp.array([-1., 1., 0.])
    c = jnp.array([-1., 1., 0.])
    d = jnp.array([-1., 1., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 0.0)



def test_point_point_non_overlap():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([0., 0., 0.])
    c = jnp.array([1., 0., 0.])
    d = jnp.array([1., 0., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 1.0)


# AB is a Point and CD is a Line Segment


def test_point_linesegment_overlap_1():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([0., 0., 0.])
    c = jnp.array([0., 0., 0.])
    d = jnp.array([1., 0., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 0.0)


def test_point_linesegment_non_overlap_1():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([0., 0., 0.])
    c = jnp.array([0., 0., 1.])
    d = jnp.array([1., 0., 1.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 1.0)


# AB is a Line Segment and CD is a Point

def test_linesegment_point_overlap_2():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([1., 0., 0.])
    c = jnp.array([0., 0., 0.])
    d = jnp.array([0., 0., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 0.0)


def test_linesegment_point_non_overlap_2():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([1., 0., 0.])
    c = jnp.array([1., 0., 1.])
    d = jnp.array([1., 0., 1.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 1.0)


# AB is a Line Segment and CD is a Line Segment


def test_fully_overlapping():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([1., 0., 0.])
    c = jnp.array([0., 0., 0.])
    d = jnp.array([1., 0., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 0.0)


def test_partially_overlapping():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([1., 0., 0.])
    c = jnp.array([0.5, 0., 0.])
    d = jnp.array([1.5, 0., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 0.0)


def test_parallel_horizontal_within_range():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([1., 0., 0.])
    c = jnp.array([0., 0., 1.])
    d = jnp.array([1., 0., 1.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 1.0)


def test_parallel_horizontal_out_of_range():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([1., 0., 0.])
    c = jnp.array([2., 0., 1.])
    d = jnp.array([3., 0., 1.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    expected_dist = jnp.linalg.norm(c-b)

    assert jnp.isclose(dist, expected_dist)


def test_parallel_horizontal_along_same_axis():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([1., 0., 0.])
    c = jnp.array([2., 0., 0.])
    d = jnp.array([3., 0., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 1.0)


def test_parallel_vertical_within_range():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([0., 1., 0.])
    c = jnp.array([0., 0., 1.])
    d = jnp.array([0., 1., 1.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 1.0)


def test_parallel_vertical_out_of_range():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([0., 1., 0.])
    c = jnp.array([1., 2., 0.])
    d = jnp.array([1., 3., 0.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    expected_dist = jnp.linalg.norm(c - b)

    assert jnp.isclose(dist, expected_dist)


def test_skew():

    a = jnp.array([0., 0., 0.])
    b = jnp.array([1., 0., 0.])
    c = jnp.array([0., 0., 2.])
    d = jnp.array([1., 0., 1.])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.isclose(dist, 1.0)


# VECTORIZED


# AB is a Point and CD is a Point


def test_point_point_overlap_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    c = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    d = jnp.array([[0., 0., 0.], [0., 0., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 0.0))


def test_point_point_overlap_positive_coordinates_vectorized():

    a = jnp.array([[1., 1., 1.], [1., 1., 1.]])
    b = jnp.array([[1., 1., 1.], [1., 1., 1.]])
    c = jnp.array([[1., 1., 1.], [1., 1., 1.]])
    d = jnp.array([[1., 1., 1.], [1., 1., 1.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 0.0))


def test_point_point_overlap_negative_coordinates_vectorized():

    a = jnp.array([[-1., -1., -1.], [-1., -1., -1.]])
    b = jnp.array([[-1., -1., -1.], [-1., -1., -1.]])
    c = jnp.array([[-1., -1., -1.], [-1., -1., -1.]])
    d = jnp.array([[-1., -1., -1.], [-1., -1., -1.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 0.0))


def test_point_point_overlap_mixed_coordinates_vectorized():

    a = jnp.array([[-1., 1., 0.], [-1., 1., 0.]])
    b = jnp.array([[-1., 1., 0.], [-1., 1., 0.]])
    c = jnp.array([[-1., 1., 0.], [-1., 1., 0.]])
    d = jnp.array([[-1., 1., 0.], [-1., 1., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 0.0))


def test_point_point_non_overlap_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    c = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    d = jnp.array([[1., 0., 0.], [1., 0., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 1.0))


# AB is a Point and CD is a Line Segment


def test_point_linesegment_overlap_1_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    c = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    d = jnp.array([[1., 0., 0.], [1., 0., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 0.0))


def test_point_linesegment_non_overlap_1_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    c = jnp.array([[0., 0., 1.], [0., 0., 1.]])
    d = jnp.array([[1., 0., 1.], [1., 0., 1.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 1.0))


# AB is a Line Segment and CD is a Point

# FIXME
def test_linesegment_point_overlap_2_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    c = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    d = jnp.array([[0., 0., 0.], [0., 0., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 0.0))


def test_linesegment_point_non_overlap_2_vectorized():

    a = jnp.array([[0., 0., 0.],[0., 0., 0.]])
    b = jnp.array([[1., 0., 0.],[1., 0., 0.]])
    c = jnp.array([[1., 0., 1.],[1., 0., 1.]])
    d = jnp.array([[1., 0., 1.],[1., 0., 1.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 1.0))


# AB is a Line Segment and CD is a Line Segment


def test_fully_overlapping_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    c = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    d = jnp.array([[1., 0., 0.], [1., 0., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 0.0))


def test_partially_overlapping_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    c = jnp.array([[0.5, 0., 0.], [0.5, 0., 0.]])
    d = jnp.array([[1.5, 0., 0.], [1.5, 0., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 0.0))


def test_parallel_horizontal_within_range_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    c = jnp.array([[0., 0., 1.], [0., 0., 1.]])
    d = jnp.array([[1., 0., 1.], [1., 0., 1.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 1.0))


def test_parallel_horizontal_out_of_range_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    c = jnp.array([[2., 0., 1.], [2., 0., 1.]])
    d = jnp.array([[3., 0., 1.], [3., 0., 1.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    expected_dist = jnp.linalg.norm(c-b, axis=1)

    assert jnp.all(jnp.isclose(dist, expected_dist))


def test_parallel_horizontal_along_same_axis_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    c = jnp.array([[2., 0., 0.], [2., 0., 0.]])
    d = jnp.array([[3., 0., 0.], [3., 0., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 1.0))


def test_parallel_vertical_within_range_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[0., 1., 0.], [0., 1., 0.]])
    c = jnp.array([[0., 0., 1.], [0., 0., 1.]])
    d = jnp.array([[0., 1., 1.], [0., 1., 1.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 1.0))



def test_parallel_vertical_out_of_range_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[0., 1., 0.], [0., 1., 0.]])
    c = jnp.array([[1., 2., 0.], [1., 2., 0.]])
    d = jnp.array([[1., 3., 0.], [1., 3., 0.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    expected_dist = jnp.linalg.norm(c - b, axis=1)

    assert jnp.all(jnp.isclose(dist, expected_dist))


def test_skew_vectorized():

    a = jnp.array([[0., 0., 0.], [0., 0., 0.]])
    b = jnp.array([[1., 0., 0.], [1., 0., 0.]])
    c = jnp.array([[0., 0., 2.], [0., 0., 2.]])
    d = jnp.array([[1., 0., 1.], [1., 0., 1.]])

    dist = minimum_distances_segments_segments(a, b, c, d)

    assert jnp.all(jnp.isclose(dist, 1.0))

