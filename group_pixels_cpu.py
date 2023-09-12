"""Script to group the pixels in an image, with no parallelism.

Generally takes 200-300 seconds for an image from my phone (4624x3672 pixels).
"""

from absl import app  # type: ignore
from absl import flags  # type: ignore
import random
import time

from PIL import Image
from PIL import ImageDraw

from typing import Callable, Iterable, Optional, TypeVar


INFILE_FLAG = flags.DEFINE_string(
    "infile", None,
    "Image file to apply a weird effect to. (Not modified in place.)",
    required=True,
)

THRESHOLD_FLAG = flags.DEFINE_float(
    "threshold", 20.0,
    "Closeness threshold (Euclidean distance in LAB color space) for two "
    "pixels to be considered 'similar' colors.")

THRESHOLD_INCREMENT_FLAG = flags.DEFINE_float(
    "threshold_increment", 0.0,
    "Amount to increase the Threshold (--threshold) each time we search from a "
    "new pixel. The idea is, once most of the image is already colored in, you "
    "can lower your standards a little so we end up with fewer distinct colors "
    "in the end result. Try a value like 0.1, 0.5, or 2. Default 0.")

OPEN_IMAGES_FLAG = flags.DEFINE_bool(
    "open_images", True,
    "Show results in your default image viewing program (e.g. a browser). "
    "Default True.")

TIME_LIMIT_FLAG = flags.DEFINE_float(
    "time_limit_seconds", None,
    "Soft time limit. To fill every pixel in the image takes 200-300 seconds. "
    "If you are in a hurry, set a value lower than that. "
    "Default no time limit")

OUTLINE_OUTFILE_FLAG = flags.DEFINE_string(
    "outline_outfile", None,
    "Output file for the generated image showing black-and-white outlines of "
    "the computed pixel groups (in PNG format). Optional.")

COLOR_OUTFILE_FLAG = flags.DEFINE_string(
    "color_outfile", None,
    "Output file for the generated image showing the pixel groups in solid "
    "blocks of color (in PNG format). Optional.")


# TODO: confident person on internet recommends LabDE2000. https://stackoverflow.com/questions/7530627
# So try that. Also try using the scipy image tools, which can convert formats and do other stuff.


def euclid_distance(pt_a: tuple[int | float, ...], pt_b: tuple[int | float, ...]) -> float:
    return sum((xa - xb)**2 for xa, xb in zip(pt_a, pt_b))**0.5


NodeT = TypeVar("NodeT")


def get_connected_set_with_predicate(
        start: NodeT,
        get_neighbors: Callable[[NodeT], Iterable[NodeT]],
        include: Callable[[NodeT], bool],
        spoken_for: Optional[set[NodeT]] = None,
        ) -> set[NodeT]:
    """Generic graph search to build a connected set of nodes based on an inclusion predicate.

    Args:
      start: The node at which the search begins. Will be included in the
        output set.
      get_neighbors: A function that, given a node, returns all nodes connected
        to it in the (implicit) graph.
      include: A predicate saying whether a node meets criteria to be included
        in the output set. (Returns True if it should be included.)
      spoken_for: (Optional) A set of nodes that shouldn't be included even if
        they satisfy the 'include' predicate, because they're already included
        in other sets that we want to be disjoint. This is also an output
        parameter, because even node added to the output set will also be added
        to spoken_for.

    Returns:
      A set of nodes reachable from `start`, all satisfying `include`, and
      disjoint from `spoken_for`. (Also, `spoken_for` will be modified.)
    """
    if spoken_for is None:
        spoken_for = set()  # in this case spoken_for could just be ignored
    scheduled = {start}
    result = set()
    stack = [start]
    while stack:
        current = stack.pop()
        if include(current):
            result.add(current)
            spoken_for.add(current)
            for neigh in get_neighbors(current):
                if neigh not in scheduled and neigh not in spoken_for:
                    scheduled.add(neigh)
                    stack.append(neigh)
    return result


def get_neighbor_coords(coord: tuple[int, int],
                        img: Image.Image) -> list[tuple[int, int]]:
    result = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = (coord[0] + dx, coord[1] + dy)
        if (0 <= neighbor[0] < img.size[0] and
            0 <= neighbor[1] < img.size[1]):
            result.append(neighbor)
    return result


def get_connected_set_of_like_color(
        coord: tuple[int, int],
        img: Image.Image,
        threshold: float,
        spoken_for: Optional[set[tuple[int, int]]] = None,
) -> set[tuple[int, int]]:
    def include_coord(c: tuple[int, int]) -> bool:
        return euclid_distance(img.getpixel(c), img.getpixel(coord)) < threshold

    return get_connected_set_with_predicate(
        coord,
        lambda c: get_neighbor_coords(c, img),
        include_coord,
        spoken_for=spoken_for,
    )


def draw_borders(component: set[tuple[int, int]],
                 output_img: Image.Image,
                 color=(0, 0, 0)) -> None:
    for coord in component:
        if not all(x in component for x in get_neighbor_coords(coord, output_img)):
            output_img.putpixel(coord, color)


def fill_region(output_img, region: Iterable[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for coord in region:
        output_img.putpixel(coord, color)


def wackify_image(img: Image.Image,
                  starting_tolerance: float = 20.0,
                  increase_tolerance: float = 0.1,
                  time_limit: Optional[float] = None,
                  ) -> tuple[Image.Image, Image.Image]:
    """Makes a weird copy of an image.

    Returns a black-and-white image showing outlines of groups, and an image
    where each groups is filled in with the color of its seed pixel.
    """
    rgb_img = img.convert("RGB")
    t = time.time()
    full_color = Image.new("RGB", rgb_img.size, color=(255, 255, 255))
    # full_color_draw = ImageDraw.Draw(full_color)
    # full_color_draw.rectangle((0, 0, img.size[0], img.size[1]),
    #                           fill=(255, 255, 255),
    #                           outline=(255, 255, 255))
    black_and_white = Image.new("L", full_color.size, color=255)

    uncolored = set((x, y) for x in range(img.size[0])
                    for y in range(img.size[1]))
    colored = set()
    print("About to do graph stuff:", time.time() - t)
    # seeds = [(random.randrange(img.size[0]), random.randrange(img.size[1]))
    #          for _ in range(10000)]
    regions_by_seed = {}
    tolerance = starting_tolerance
    while uncolored:
        # seed = random.choice(list(uncolored))
        seed = next(iter(uncolored))
        print("picked a seed", time.time() - t)
        regions_by_seed[seed] = get_connected_set_of_like_color(
            seed, img, tolerance, spoken_for=colored)
        uncolored.difference_update(regions_by_seed[seed])
        print("did a graph thing:", time.time() - t)
        tolerance += increase_tolerance
        if time_limit is not None and time.time() - t > time_limit:
            break
    print("About to draw:", time.time() - t)
    for region in regions_by_seed.values():
        draw_borders(region, black_and_white, color=0)
    for seed, region in reversed(regions_by_seed.items()):
        fill_region(full_color, region, rgb_img.getpixel(seed))
    print("Done drawing:", time.time() - t)
    return black_and_white, full_color


def main(unused_argv):
    img = Image.open(INFILE_FLAG.value)
    black_and_white_image, color_image = wackify_image(
        img.convert("LAB"),
        starting_tolerance=THRESHOLD_FLAG.value,
        increase_tolerance=THRESHOLD_INCREMENT_FLAG.value,
        time_limit=TIME_LIMIT_FLAG.value,
    )
    if OPEN_IMAGES_FLAG.value:
        black_and_white_image.show()
        color_image.show()
    if OUTLINE_OUTFILE_FLAG.value is not None:
        black_and_white_image.save(OUTLINE_OUTFILE_FLAG.value,
                                   format='png')
    if COLOR_OUTFILE_FLAG.value is not None:
        color_image.save(COLOR_OUTFILE_FLAG.value,
                         format='png')




if __name__ == "__main__":
    app.run(main)
