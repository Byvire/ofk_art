from absl import app  # type: ignore
from absl import flags  # type: ignore

import numpy as np
import os
import pyopencl as cl
from PIL import Image
import sys
import time
from typing import Optional


_INFILE_FLAG = flags.DEFINE_string(
    "infile", None,
    "Image file to apply a weird effect to. (Not modified in place.)",
    required=True,
)

_OUTFILE_FLAG = flags.DEFINE_string(
    "outfile", None,
    "Output PNG file path. If not provided, image will be opened in browser. "
    "Will be overwritten if provided.",
)

_ITERATIONS_FLAG = flags.DEFINE_integer(
    "iterations", None,
    "How many iterations of the parallel BFS GPU algorithm to do. "
    "Each iteration, colors propagate by a distance of 1 pixel (or "
    "sqrt(2) pixels diagonally I guess?), so more iterations means larger "
    "patches of color in the final image. "
    "My machine (intel built-in GPU on laptop) can do about 22 iterations per "
    "second. "
    "If unspecified, we use max(image width, image depth), which may take a "
    "long time (e.g. 4600 iterations, 3-4 minutes).",
)

_THRESHOLD_FLAG = flags.DEFINE_integer(
    "threshold", 40,
    "Distance between two colors that is considered 'close'. Pixels that are "
    "close in color may be grouped together. "
    "I like values in the 20-40 range but this parameter is fun to play with.",
)



GROUPING_KERNEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "pixel_group_kernel.cl")
SUBSTITUTION_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "pixel_substitution_kernel.cl")


class Timer:
    def __init__(self, stfu=False):
        self._stfu = stfu  # means "be quiet"
        self._start = time.time()

    def msg(self, *args, **kwargs):
        if self._stfu:
            return
        if 'file' not in kwargs:
            kwargs['file'] = sys.stderr
        print("Time %.4f" % (time.time() - self._start),
              *args, **kwargs)


def load_program(context: cl.Context, kernel_path: str) -> cl.Program:
    with open(kernel_path, 'r') as f:
        return cl.Program(context, f.read()).build()


# Flag combinations for opencl Buffers.
# (It seems like using USE_HOST_PTR for large read-only buffers like
# the pixel data (lab_band_bufs) would be faster, but it's not.)
READ_WRITE = (cl.mem_flags.READ_WRITE
              | cl.mem_flags.COPY_HOST_PTR)
READ_ONLY = (cl.mem_flags.READ_ONLY
             | cl.mem_flags.COPY_HOST_PTR)
WRITE_ONLY = (cl.mem_flags.WRITE_ONLY
              | cl.mem_flags.COPY_HOST_PTR)


def _random_priorities(count: int, timer: Timer) -> np.ndarray:
    # This takes about 1.5 seconds, not as bad as I expected.
    # Random priorities look waaaay better than my original approach (which you
    # can try if you comment out the "shuffle" line.)
    priorities = np.array(range(count), dtype=cl.cltypes.int)
    timer.msg("shuffle priorities start")
    np.random.shuffle(priorities)
    timer.msg("shuffle priorities end")
    return priorities



def compute_pixel_groups(context: cl.Context,
                         queue: cl.CommandQueue,
                         lab_img: Image.Image,
                         iterations: int,
                         threshold: int,
                         timer: Timer) -> cl.Buffer:
    # Returns a read-write Buffer that maps each pixel to the coordinate of its
    # group's seed pixel.

    total_size = lab_img.size[0] * lab_img.size[1]

    # OpenCL "buffer" objects represent memory in the GPU's memory space. Making
    # buffers is sort of like "throwing data over the wall" to the CPU.

    dimensions_buf = cl.Buffer(context, READ_ONLY,
                               hostbuf=np.array(lab_img.size, dtype=cl.cltypes.int))
    timer.msg("Making lab band bufs")
    lab_band_bufs = [
        cl.Buffer(context, READ_ONLY, hostbuf=np.array(lab_img.getdata(band=band),
                                                       dtype=cl.cltypes.uchar))
        for band in range(3)]
    timer.msg("Done making lab band bufs")
    threshold_buf = cl.Buffer(context, READ_ONLY,
                              hostbuf=np.array([threshold], dtype=cl.cltypes.int))
    # Priorities would ideally be randomized but I think that will be slow as heck.
    priorities_buf = cl.Buffer(
        context, READ_WRITE,
        hostbuf=_random_priorities(total_size, timer))
    group_membership = cl.Buffer(
        context, READ_WRITE,
        hostbuf=np.array(range(total_size), dtype=cl.cltypes.int))
    # output param.
    group_membership_out = cl.Buffer(
        context, READ_WRITE,
        hostbuf=np.empty(total_size, dtype=cl.cltypes.int))

    grouping_program = load_program(context, GROUPING_KERNEL_PATH)
    # Although this uses __getattr__ syntax, accessing .groupPixels (the kernel
    # name) multiple times does cause a new Kernel instance to be created each
    # time. (They were trying to be fancy and made their API bad.)
    grouping_kernel = grouping_program.groupPixels

    timer.msg('about to start GPUing')

    for ix in range(iterations):
        if (ix - 1) % 50 == 0:
            timer.msg('on iteration', ix)
        grouping_kernel(queue,
                        (total_size + (total_size % 128),),  # global size
                        (128,),  # local size  (must divide global size)
                        dimensions_buf,
                        *lab_band_bufs,
                        threshold_buf,
                        priorities_buf,
                        group_membership,
                        group_membership_out,
                        ).wait()
        # Swap input and output buffers between iterations.
        group_membership, group_membership_out = (
            group_membership_out, group_membership)
    timer.msg('finished group computation')
    return group_membership


def _image_from_bands(bands: list[np.ndarray], shape: tuple[int, int]) -> Image.Image:
    return Image.merge(
        "RGB",
        [Image.fromarray(np.reshape(band, shape[::-1], order="C"), "L")
         for band in bands])


def substitute_pixels(context: cl.Context,
                      queue: cl.CommandQueue,
                      rgb_img: Image.Image,
                      substitutions: cl.Buffer,
                      timer: Optional[Timer] = None) -> Image.Image:
    """Takes an input RGB image and returns a modified copy.

    Args:
      context: OpenCL context (used to execute code on GPU).
      queue: OpenCL execution queue. Should probably not be, like, still running
        a thing that writes to the `substitutions` buffer?
      rgb_img: An image in the Red-Green-Blue color space, as a template for the
        output image.
      substitutions: Holds an array of length [number-of-pixels-in-rgb_img].
        Each value is an index into the rgb_img pixel data. The value of pixel i
        in the output image will be rgb_image.getpixel(substitutions[i]).
      timer: For debugging, optional.

    Returns:
      A PIL Image object.

    Raises:
      Ha. Ha. If indices are out of bounds that's probably undefined behavior.
    """
    if timer is None:
        timer = Timer(stfu=True)
    timer.msg("Setting up for pixel substitution")
    substitution_program = load_program(context, SUBSTITUTION_KERNEL_PATH)
    substitution_kernel = substitution_program.substitutePixels

    total_size = rgb_img.size[0] * rgb_img.size[1]
    size_buf = cl.Buffer(
        context, READ_ONLY,
        hostbuf=np.array([total_size], dtype=cl.cltypes.int))

    band_out_bufs = []
    timer.msg("Starting pixel substitution GPU stuff")
    for band_index in range(3):
        band_in_buf = cl.Buffer(
            context, READ_ONLY,
            hostbuf=np.array(rgb_img.getdata(band=band_index),
                             dtype=cl.cltypes.uchar))
        band_out_buf = cl.Buffer(
            context, WRITE_ONLY,
            hostbuf=np.empty(total_size, dtype=cl.cltypes.uchar))
        substitution_kernel(queue,
                            (total_size + total_size % 128,),  # global size
                            (128,),  # local size
                            size_buf,
                            substitutions,
                            band_in_buf,
                            band_out_buf,
                            )
        band_out_bufs.append(band_out_buf)
    queue.finish()  # blocking

    bands = []
    for out_buf in band_out_bufs:
        result = np.empty(total_size, dtype=cl.cltypes.uchar)
        cl.enqueue_copy(queue, result, out_buf)
        bands.append(result)
    queue.finish()  # still blocking
    timer.msg("Finished pixel substitution GPU stuff")
    return _image_from_bands(bands, rgb_img.size)


def opencl_wackify_image(img: Image.Image,
                         iterations: int,
                         threshold: int = 40) -> Image.Image:
    timer = Timer()
    # RGB is the usual color space where colors are represented by red, green,
    # and blue components (3 bands). This is the color space for image display.
    # (The image is probably already in RGB format but we convert it anyway.)
    rgb_img = img.convert("RGB")
    # LAB is intended as a "perceptually uniform" color space, meaning that
    # Euclidean distance between LAB coordinates indicates how similar a human
    # would say the colors are. (It's imperfect but better than RGB.)
    lab_img = img.convert("LAB")
    del img
    timer.msg("converted to lab / rgb")

    lab_bands = [lab_img.getdata(band=i) for i in range(3)]

    context = cl.create_some_context()
    with cl.CommandQueue(context) as queue:
        memberships_buf = compute_pixel_groups(
            context, queue, lab_img, iterations=iterations,
            threshold=threshold,
            timer=timer)
        return substitute_pixels(context, queue, rgb_img, memberships_buf, timer)


def main(unused_argv):
    in_image = Image.open(_INFILE_FLAG.value)
    iterations = _ITERATIONS_FLAG.value
    if iterations is None:
        iterations = max(in_image.size[0], in_image.size[1])

    out_image = opencl_wackify_image(in_image, iterations,
                                     threshold=_THRESHOLD_FLAG.value)

    if _OUTFILE_FLAG.value is None:
        out_image.show()
    else:
        try:
            out_image.save(_OUTFILE_FLAG.value, format='png')
        except ValueError:
            out_image.show()
            raise


if __name__ == "__main__":
    app.run(main)
