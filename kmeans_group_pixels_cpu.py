"""Use k-means to group the pixels of an image by color and then replace each color with the nearest mean color.

Good explanation of k-means:
https://blog.paperspace.com/speed-up-kmeans-numpy-vectorization-broadcasting-profiling/
(I only read the beginning because... I didn't care to use their optimizations.)

Suggested by a friend.
"""

from absl import app
from absl import flags
import random
import sys
from typing import Iterable, Union
from PIL import Image


_INFILE_FLAG = flags.DEFINE_string(
    "infile", None,
    "Input image file path.",
    required=True,
)

_NUM_MEANS_FLAG = flags.DEFINE_integer(
    "k", 10,
    "Number of means for the k-means algorithm. "
    "Running time is O(k * n) where n is the number of pixels in --infile. "
    "Default 10.",
)

_THRESHOLD_FLAG = flags.DEFINE_float(
    "threshold", 10.0,
    "The smaller this is, the longer k-means will take to converge.")


Number = Union[int, float]


def _distance_squared(a: tuple[Number, ...],
                      b: tuple[Number, ...]) -> float:
    return sum((ax - bx)**2 for ax, bx in zip(a, b))


def _assign_to_groups(
        all_points: list[tuple[Number, ...]],
        centroids: list[tuple[float, ...]]
) -> dict[tuple[float, ...]: set[tuple[Number, ...]]]:
    result = {centroid: set() for centroid in centroids}
    for point in all_points:
        result[min(centroids, key=lambda c: _distance_squared(c, point))].add(point)
    return result


def _get_new_mean_centroids(
        groups: Iterable[set[tuple[Number, ...]]],
        # groups: dict[tuple[float, float, float]: set[tuple[int, int, int]]]
) -> list[tuple[float, ...]]:
    result = []
    for group in groups:
        result.append(tuple(
            sum(point[d] for point in group) / len(group)
            for d in range(len(next(iter(group))))))
    return result


def k_means(points: list[tuple[Number, ...]], k: int, epsilon=5.0) -> list[tuple[float, ...]]:
    centroids = random.sample(points, k)
    while True:
        new_centroids = _get_new_mean_centroids(
            _assign_to_groups(points, centroids).values())
        distance_from_convergence = max(_distance_squared(old, new)**0.5
                                        for old, new in zip(centroids, new_centroids))
        # Each iteration takes like 5 minutes so I print the centroids, so you can
        # hardcode the value of centroids at the start to pick up where you left off.
        print(new_centroids)
        print(distance_from_convergence, file=sys.stderr)
        if distance_from_convergence < epsilon:
            return new_centroids
        centroids = new_centroids


def substitute_centroid_colors(
        img: Image.Image,
        points: list[tuple[Number, ...]],
        centroids: list[tuple[Number, ...]],
) -> None:
    # Modifies img in place.
    # Image should be in the same color space (Image.mode) as the pixel data (points).
    # So probably LAB for good results.

    # We assume the first 3 coordinates in a point are pixel values between 0
    # and 255, and other coordinates may be used to represent abstract sorta
    # stuff. (That's also why `points` is a separate parameter from `img`.)
    centroid_to_color = {centroid: tuple(int(x) for x in centroid[:3])
                         for centroid in centroids}
    for index, point in enumerate(points):
        row_col = (index % img.size[0], index // img.size[0])
        centroid = min(centroids, key=lambda c: _distance_squared(c, point))
        img.putpixel(row_col, centroid_to_color[centroid])


def main(unused_argv) -> None:
    img = Image.open(_INFILE_FLAG.value).convert("LAB")
    pixels = list(img.getdata())
    centroids = k_means(pixels, _NUM_MEANS_FLAG.value, epsilon=_THRESHOLD_FLAG.value)

    substitute_centroid_colors(img, pixels, centroids)
    img.show()


if __name__ == "__main__":
    app.run(main)
