import unittest

import pyopencl as cl
import os
import numpy as np
from PIL import Image

import group_pixels_opencl  # local


SAMPLE_IMAGE_PATH = os.path.join(os.path.dirname(__file__),
                                 "photos", "rice.jpg")


class KernelTestCase(unittest.TestCase):

    def test_pixel_coordinate_conversion_round_trip(self):
        size = (3, 5)
        total_size = size[0] * size[1]
        context = cl.create_some_context()
        dimensions_buf = cl.Buffer(context, group_pixels_opencl.READ_ONLY,
                                   hostbuf=np.array(size, dtype=cl.cltypes.int))
        indices_buf = cl.Buffer(
            context, group_pixels_opencl.WRITE_ONLY,
            hostbuf=np.empty(total_size, dtype=cl.cltypes.int))
        prog = group_pixels_opencl.load_program(
            context, group_pixels_opencl.GROUPING_KERNEL_PATH)
        with cl.CommandQueue(context) as queue:
            prog.convertIndicesToRowColumnAndBack(
                queue, (total_size,), (1,),
                dimensions_buf, indices_buf).wait()
            result = np.empty(total_size, dtype=cl.cltypes.int)
            cl.enqueue_copy(queue, result, indices_buf)

        np.testing.assert_array_equal(
            np.array(range(total_size), dtype=cl.cltypes.int),
            result)


def select_pixels(img, nums):
    # Converting the whole image to a list takes a long time.
    # ImageCore obojects don't support slicing.
    data = img.getdata()
    return [data[i] for i in nums]

def _assert_image_equal(test_case, expected, actual):
    test_case.assertEqual(expected.size, actual.size)
    test_case.assertEqual(select_pixels(expected, range(0, 10000, 2000)),
                          select_pixels(actual, range(0, 10000, 2000)))



class PythonTestCase(unittest.TestCase):

    def test_photo_to_and_from_bands_is_noop(self):
        img = Image.open(SAMPLE_IMAGE_PATH)
        _assert_image_equal(
            self,
            img,
            group_pixels_opencl._image_from_bands(
                # Notably this fails if we use dtype=cl.cltypes.int
                [np.array(img.getdata(band=ix), dtype=cl.cltypes.uchar)
                 for ix in range(3)],
                img.size))


    def test_wackify_with_zero_iterations_is_noop(self):
        img = Image.open(SAMPLE_IMAGE_PATH)
        result = group_pixels_opencl.opencl_wackify_image(
            img, iterations=0, threshold=100)
        _assert_image_equal(self, img, result)


if __name__ == "__main__":
    unittest.main()
