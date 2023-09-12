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

# Not useful. I wondered if PIL.Image was especially slow for pixel editing,
# and the answer is no. There are just a lot of pixels.
class PixelImage:
    def __init__(self, dimensions: tuple[int, int],
                 fill_color=(255, 255, 255)) -> None:
        # self._data = [(255, 255, 255)] * dimensions[0] * dimensions[1]
        # The duplication is intentional for speed.
        row_0 = [fill_color] * dimensions[1]
        self._data = [row_0] * dimensions[0]
        self._dimensions = dimensions

    def putpixel(self, coord: tuple[int, int],
                  color: tuple[int, int, int]) -> None:
        # The first time a row is modified we instantiate it for real.
        if coord[0] != 0 and self._data[coord[0]] is self._data[0]:
            self._data[coord[0]] = list(self._data[0])

        self._data[coord[0]][coord[1]] = color

        # index = coord[0] + coord[1] * self._dimensions[0]
        # self._data[index] = color

    # def get_data(self) -> list[tuple[int, int, int]]:
    #     return self._data

    def get_array(self) -> list[list[tuple[int, int, int]]]:
        return self._data

    @property
    def size(self):
        return self._dimensions


# TODO: confident person on internet recommends LabDE2000. https://stackoverflow.com/questions/7530627
# So try that. Also try using the scipy image tools, which can convert formats and do other stuff.


def euclid_distance(pt_a: tuple[int | float, ...], pt_b: tuple[int | float, ...]) -> float:
    return sum((xa - xb)**2 for xa, xb in zip(pt_a, pt_b))**0.5


NodeT = TypeVar("NodeT")
# GraphT = TypeVar("GraphT")


def get_connected_component_predicate(
        # graph: GraphT,
        start: NodeT,
        get_neighbors: Callable[[NodeT], Iterable[NodeT]],
        include: Callable[[NodeT], bool],
        spoken_for: Optional[set[NodeT]] = None,
        ) -> set[NodeT]:
    # spoken_for is a set of nodes that are already grouped in a different
    # connected component and shouldn't be considered for this one.
    if spoken_for is None:
        spoken_for = set()  # in this case spoken_for could just be ignored
    scheduled = {start}
    result = set()
    stack = [start]
    while stack:
        current = stack.pop()
        # if current in visited:
        #     continue
        # visited.add(current)
        if include(current):
            result.add(current)
            spoken_for.add(current)
            for neigh in get_neighbors(current):
                if neigh not in scheduled and neigh not in spoken_for:
                    scheduled.add(neigh)
                    stack.append(neigh)
            # stack.extend(get_neighbors(current))
    return result


# def unbounded_neighbor_coords(coord: tuple[int, int]) -> list[tuple[int, int]]:
#     return [(coord[0] + dx, coord[1] + dy)
#             for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]]

def get_neighbor_coords(coord: tuple[int, int],
                        img: Image.Image | PixelImage) -> list[tuple[int, int]]:
    result = []
    # for neighbor in unbounded_neighbor_coords(coord):
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0)]:
        neighbor = (coord[0] + dx, coord[1] + dy)
        if (0 <= neighbor[0] < img.size[0] and
            0 <= neighbor[1] < img.size[1]):
            result.append(neighbor)
    return result



def get_cc_rgb_euclid(coord: tuple[int, int],
                      img: Image.Image,
                      threshold: float,
                      spoken_for: Optional[set[tuple[int, int]]] = None,
                      ) -> set[tuple[int, int]]:
    def include_coord(c: tuple[int, int]) -> bool:
        return euclid_distance(img.getpixel(c), img.getpixel(coord)) < threshold

    return get_connected_component_predicate(
        coord,
        lambda c: get_neighbor_coords(c, img),
        include_coord,
        spoken_for=spoken_for,
    )


def draw_borders(component: set[tuple[int, int]],
                 output_img: Image.Image | PixelImage,
                 color=(0, 0, 0)) -> None:
    for coord in component:
        if not all(x in component for x in get_neighbor_coords(coord, output_img)):
            output_img.putpixel(coord, color)


def fill_region(output_img, region: Iterable[tuple[int, int]], color: tuple[int, int, int]) -> None:
    for coord in region:
        output_img.putpixel(coord, color)


# Good but doesn't keep track of what space needs to be filled in, so leaves white space all over.
# def wackify_image(img: Image.Image) -> Image.Image:
#     """Makes a weird copy of an image."""
#     rgb_img = img.convert("RGB")
#     t = time.time()
#     copy = img.copy().convert("RGB")
#     draw = ImageDraw.Draw(copy)
#     draw.rectangle((0, 0, img.size[0], img.size[1]),
#                    fill=(255, 255, 255),
#                    outline=(255, 255, 255))
#     print("About to do graph stuff:", time.time() - t)
#     seeds = [(random.randrange(img.size[0]), random.randrange(img.size[1]))
#              for _ in range(10000)]
#     regions_by_seed = {}
#     for seed in seeds:
#         if not any(seed in region for region in regions_by_seed.values()):
#             regions_by_seed[seed] = get_cc_rgb_euclid(seed, img, 30.0)
#         print("did a graph thing:", time.time() - t)
#     print("About to draw:", time.time() - t)
#     for region in regions_by_seed.values():
#         draw_borders(region, copy)
#     copy.show()
#     for seed, region in regions_by_seed.items():
#         fill_region(copy, region, rgb_img.getpixel(seed))
#     return copy

def wackify_image(img: Image.Image,
                  starting_tolerance: float = 20.0,
                  increase_tolerance: float = 0.1) -> Image.Image:
    """Makes a weird copy of an image."""

    rgb_img = img.convert("RGB")
    t = time.time()
    copy = img.copy().convert("RGB")
    draw = ImageDraw.Draw(copy)
    draw.rectangle((0, 0, img.size[0], img.size[1]),
                   fill=(255, 255, 255),
                   outline=(255, 255, 255))
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
        regions_by_seed[seed] = get_cc_rgb_euclid(seed, img, tolerance, spoken_for=colored)
        uncolored.difference_update(regions_by_seed[seed])
        print("did a graph thing:", time.time() - t)
        tolerance += increase_tolerance
        if time.time() - t > 200:
            break
    print("About to draw:", time.time() - t)
    for region in regions_by_seed.values():
        draw_borders(region, copy)
    copy.show()
    draw.rectangle((0, 0, img.size[0], img.size[1]),
                   fill=(0, 0, 0),
                   outline=(0, 0, 0))
    for seed, region in reversed(regions_by_seed.items()):
        fill_region(copy, region, rgb_img.getpixel(seed))
    return copy


def main(unused_argv):
    img = Image.open(INFILE_FLAG.value)
    wackify_image(img.convert("LAB")).show()


if __name__ == "__main__":
    app.run(main)
