from absl import app  # type: ignore
from absl import flags  # type: ignore
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
    "300x200 image and downscale by a factor of 3, you end up with a 100x67 "
    "image.",
)


def main(_):
    img = Image.open(_INFILE_FLAG.value)
    result = img.reduce(_FACTOR_FLAG.value)
    if _OUTFILE_FLAG.value is not None:
        result.save(_OUTFILE_FLAG.value, format="png")
    else:
        result.show()


if __name__ == "__main__":
    app.run(main)
