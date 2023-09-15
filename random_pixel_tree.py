"""
A script that makes a two-color image depicting a random tree of pixels.
The tree grows in random directions but the branches avoid touching each other.
... It looks like a bowl of noodles if you choose noodle colors.
"""
from absl import app
from absl import flags
import random
from typing import Generic, TypeVar
from PIL import Image


_WIDTH_FLAG = flags.DEFINE_integer(
    "width", 100,
    "Width of the result image in pixels.")
_HEIGHT_FLAG = flags.DEFINE_integer(
    "height", 100,
    "Height of the result image in pixels.")
_OUTFILE_FLAG = flags.DEFINE_string(
    "outfile", None,
    "File path to write/overwrite with the result image. PNG format.",
    required=True,
)
_BG_COLOR_FLAG = flags.DEFINE_string(
    "bgcolor", "0,0,0",
    "Background color, as a comma-separated list of 3 ints between 0 and 255.")
_FG_COLOR_FLAG = flags.DEFINE_string(
    "fgcolor", "255,255,255",
    "Foreground color, as a comma-separated list of 3 ints between 0 and 255.")


T = TypeVar("T")


# When you implement Union Find but then realize you don't need it, but you
# don't want to delete your implementation because it's your fave data structure
# class UnionFind(Generic[T]):

#     def __init__(self) -> None:
#         self._parent = {}
#         # Size is tricky because only the root node sizes are accurate.
#         self._size = {}

#     def _add_if_missing(self, elem: T) -> None:
#         if elem not in self._parent:
#             self._parent[elem] = elem
#             self._size[elem] = 1

#     def find(self, elem: T) -> T:
#         """Returns the root of the tree containing elem."""
#         self._add_if_missing(elem)
#         if self._parent[elem] == elem:
#             return elem
#         # Recursion with path compression
#         self._parent[elem] = self.find(self._parent[elem])
#         return self._parent[elem]

#     def union(self, a: T, b: T) -> None:
#         """Merges the trees containing a and b."""
#         a = self.find(a)
#         b = self.find(b)
#         if b == a:
#             return
#         if self._size[a] > self._size[b]:
#             b, a = a, b
#         # b is bigger so we hang a from b
#         self._parent[a] = b
#         self._size[b] += self._size[a]
#         del self._size[a]


def _neighbors(coord: tuple[int, int], width: int, height: int) -> list[tuple[int, int]]:
    result = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            neighbor = (coord[0] + dx, coord[1] + dy)
            if (neighbor == coord or neighbor[0] < 0 or neighbor[0] >= width
                or neighbor[1] < 0 or neighbor[1] >= height):
                continue
            result.append(neighbor)
    return result


def rand_pixel_tree_grid(width: int, height: int):
    grid = [[False] * width for _ in range(height)]
    start = (random.randrange(width), random.randrange(height))
    pending = {start}
    stack = [start]
    iters = 0
    while stack:
        iters += 1
        coord = stack.pop()
        neighbors_already_in = [neigh for neigh in _neighbors(coord, width, height)
                                if grid[neigh[1]][neigh[0]]]
        if len(neighbors_already_in) > 1:
            continue
        grid[coord[1]][coord[0]] = True
        neighbors = _neighbors(coord, width, height)
        random.shuffle(neighbors)
        for neighbor in neighbors:  # NB if you want less density you can truncate neighbors here.
            if neighbor not in pending:
                pending.add(neighbor)
                stack.append(neighbor)
    print(iters)
    return grid


def make_pixel_tree_image(width: int, height: int,
                          background_color = (0, 0, 0),
                          foreground_color = (255, 255, 255)) -> Image.Image:
    grid = rand_pixel_tree_grid(width, height)
    img = Image.new("RGB", (width, height), color=background_color)
    for row in range(height):
        for col in range(width):
            if grid[row][col]:
                img.putpixel((col, row), foreground_color)
    return img


def _parse_color(text: str) -> tuple[int, int, int]:
    r, g, b = [int(c.strip()) for c in text.split(",")]
    return (r, g, b)


def main(unused_argv) -> None:
    make_pixel_tree_image(_WIDTH_FLAG.value, _HEIGHT_FLAG.value,
                          background_color=_parse_color(_BG_COLOR_FLAG.value),
                          foreground_color=_parse_color(_FG_COLOR_FLAG.value),
                          ).save(_OUTFILE_FLAG.value)


if __name__ == "__main__":
    app.run(main)
