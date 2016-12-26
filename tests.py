from unittest import TestCase

import numpy
from numpy.testing import assert_array_equal

from lib import binary_to_one_hot


class TestBinaryToOneHot(TestCase):
    def test_call(self):
        binary = numpy.array([0, 1, 1])
        hot = binary_to_one_hot(binary)
        assert_array_equal(hot, numpy.array([[1, 0], [0, 1], [0, 1]]))
