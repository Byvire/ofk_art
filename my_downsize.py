"""Script to downsize an image.

I did this just as an exercise, since this functionality is built into PIL
anyway. My version is slightly worse because I round the output image size down
instead of up. Also because it's slower.
"""


from absl import app  # type: ignore
from absl import flags  # type: ignore
import numpy as np
import os
import pyopencl as cl
from PIL import Image


_INFILE_FLAG = flags.DEFINE_string(
    "infile", None,
    "Input image, of which this script will produce a downscaled copy.",
    required=True,
)

_OUTFILE_FLAG = flags.DEFINE_string(
    "outfile", None,
    "Output PNG file path. If not provided, image will be opened in browser. "
    "Will be overwritten if provided.",
)

_FACTOR_FLAG = flags.DEFINE_integer(
    "factor", 2,
    "Integer factor by which to downscale the image. E.g. if you start with a "
    "300x200 image and downscale by a factor of 3, you end up with a 100x66 "
    "image (with the bottom of the input image being truncated by 2 pixels).",
)


DOWNSIZE_KERNEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "downsize_kernel.cl")


def _image_from_bands(bands: list[np.ndarray], shape: tuple[int, int]) -> Image.Image:
    # Shamelessly copy-pasted from group_pixels_opencl.py because this project
    # isn't set up as a real package yet.
    return Image.merge(
        "LAB",
        [Image.fromarray(np.reshape(band, shape[::-1], order="C"), "L")
         for band in bands])


READ_ONLY = (cl.mem_flags.READ_ONLY
             | cl.mem_flags.COPY_HOST_PTR)
WRITE_ONLY = (cl.mem_flags.WRITE_ONLY
              | cl.mem_flags.COPY_HOST_PTR)


def downscale_image(img: Image.Image,
                    factor: int) -> Image.Image:
    img = img.convert("LAB")

    job_size = (img.size[0] // factor) * (img.size[1] // factor)

    context = cl.create_some_context()
    with cl.CommandQueue(context) as queue:
        dimensions_buf = cl.Buffer(context, READ_ONLY,
                                   hostbuf=np.array(img.size, dtype=cl.cltypes.int))
        lab_band_bufs = [
            cl.Buffer(context, READ_ONLY, hostbuf=np.array(img.getdata(band=band),
                                                           dtype=cl.cltypes.uchar))
            for band in range(3)]
        output_band_bufs = [
            cl.Buffer(context, WRITE_ONLY,
                      hostbuf=np.zeros(job_size, dtype=cl.cltypes.uchar))
            for _ in range(3)]
        factor_buf = cl.Buffer(context, READ_ONLY,
                               hostbuf=np.array([factor], dtype=cl.cltypes.int))
        with open(DOWNSIZE_KERNEL_PATH, "r") as f:
            program = cl.Program(context, f.read()).build()
        program.downsizeImage(
            queue,
            (job_size + 128 - (job_size % 128),),  # global size
            (128,),  # local size (must divide global size)
            dimensions_buf,
            *lab_band_bufs,
            *output_band_bufs,
            factor_buf,
        ).wait()

        # Move output band data back into CPU space (from GPU memory land).
        result_bands = []
        for out_buf in output_band_bufs:
            result = np.empty(job_size, dtype=cl.cltypes.uchar)
            cl.enqueue_copy(queue, result, out_buf)
            result_bands.append(result)
        queue.finish()  # waits for the enqueued copy operations
        return _image_from_bands(result_bands,
                                 (img.size[0] // factor, img.size[1] // factor))



def main(_):
    in_img = Image.open(_INFILE_FLAG.value)
    out_img = downscale_image(in_img, _FACTOR_FLAG.value).convert("RGB")
    if _OUTFILE_FLAG.value is not None:
        out_img.save(_OUTFILE_FLAG.value, format="png")
    else:
        out_img.show()

if __name__ == "__main__":
    app.run(main)
