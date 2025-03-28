"""Creates a version of an image that looks like all one big long noodle.

Well, a squiggly line.

Usage: python noodlize.py --infile=input_image_file > result.svg

You will need to install the dependencies, including
github.com/Byvire/potato_sauce.

To get the best possible output for a given input image, try messing around
with the flags that control how the grayscale image is turned into a point
cloud. But also consider preprocessing the image, because not all paths to
grayscale yield the same results.

Note that the default settings cause the squiggle to be most dense in the
darkest parts of the image, and least dense in the lightest parts. But you can
invert this by setting --grayscale_most_dense to a lighter value than
--grayscale_least_dense. For example,

python noodlize.py --infile=my_image.jpg \
  --grayscale_most_dense=255 \
  --grayscale_least_dense=0 \
  --bg_color='#000000' \
  --squiggle_color='#ffffff' > out.svg

The output file may be quite large (e.g. a 30 megabyte svg). So I then
recommend running

rsvg-convert -u result.svg > result.png

At which point you may still have a 30 megabyte PNG file. For better lossless
compression, try:

optipng result.png

Or for more substantial but lossy JPG compression, use ImageMagick:

magick convert result.png -quality 10 result.jpg
"""

from absl import app
from absl import flags
import collections
import heapq
import random
import sys
from typing import Callable, Collection, Generic, NamedTuple, Optional, Sequence
import typing
from PIL import Image

from potato_sauce import geom  # github.com/Byvire/potato_sauce



INFILE_FLAG = flags.DEFINE_string(
    "infile", None, "Input image file.",
    required=True,
)

POINTS_ONLY_FLAG = flags.DEFINE_bool(
    "points_only",
    False,
    "Just show the point cloud generated based on grayscale values, and skip "
    "all other steps. "
    "This is useful when adjusting parameters to suit a new image.",
)

BACKGROUND_COLOR_FLAG = flags.DEFINE_string(
    "bg_color",
    "#ffffff",
    "Background color, in CSS format. Default: #ffffff",
)

SQUIGGLE_COLOR_FLAG = flags.DEFINE_string(
    "squiggle_color",
    "#000000",
    "Squiggle color, in CSS format. Default: #000000",
)

POINT_DENSITY_FLAG = flags.DEFINE_float(
    "max_point_density",
    0.1,
    "Probability that a 'completely dark' pixel will become a point in the "
    "random point set that we create. "
    "(The squiggly line we produce is a path through this random point set.) "
    "0.1 is reasonable for a medium-res image, e.g. 1024x1024 pixels. "
    "For higher resolution photos, try lower values like 0.03, and keep an eye "
    "on the printed 'Point cloud size' debug info, which will be directly "
    "proportional to this value. A million points is on the high end and will "
    "take several minutes to triangulate and draw. Default: 0.1"
)

GRAYSCALE_LIGHTEST_FLAG = flags.DEFINE_integer(
    "grayscale_least_dense",
    155,
    "Grayscale value should correspond to 0% squiggle-density. "
    "Grayscale values are from 0-255 but you can experiment with values "
    "outside that range if you want white parts of the input image to be drawn "
    "as less-dense squiggles rather than as empty space. "
    "Default: 155",
)

GRAYSCALE_DARKEST_FLAG = flags.DEFINE_integer(
    "grayscale_most_dense",
    0,
    "Grayscale value that should correspond to maximum squiggle-density. "
    "Values between this and --grayscale_least_dense will have squiggle-"
    "density values iterpolated on a cubic scale. Values outside that range "
    "will be capped to 0% or 100% of --max_point_density. "
    "Default: 0"
)


def image_to_points(img: Image.Image) -> list[geom.Point]:
    """Makes a point set based on grayscale values of pixels in the image.

    The probability that the point (x, y) is in the output set depends how
    dark the pixel at (x, y) is.

    Twiddling with constants in this function can improve the result of the
    script depending on the input image.
    """
    if img.mode != "L":
        img = img.convert("L")
        # Depending on the image, another route to grayscale may result in
        # higher contrast. For example, for the Zakim Bridge photo, I liked:
        # img = img.convert("LAB").getchannel(2)
    points = []
    max_density = POINT_DENSITY_FLAG.value
    assert 0 < max_density <= 1
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            grayscale_val = img.getpixel((x, y))
            fraction_dark = (
                (grayscale_val - GRAYSCALE_LIGHTEST_FLAG.value)
                / (GRAYSCALE_DARKEST_FLAG.value - GRAYSCALE_LIGHTEST_FLAG.value))
            fraction_dark = min(1, max(0, fraction_dark))
            if (random.random() < max_density * fraction_dark ** 5):
                # Adding random.random() helps avoid some edge cases caused by
                # colinear points.
                points.append(geom.Point(x + random.random() * 0.2,
                                         y + random.random() * 0.2))
    print("Point cloud size:", len(points), file=sys.stderr)
    return points


def _get_bounding_triangle(points: Sequence[geom.Point]) -> tuple[geom.Point, geom.Point, geom.Point]:
    # The Bower-Watson algorithm for constructing a Delaunay triangulation only
    # works if you start with one big triangle that bounds the rest of the
    # point set. You remove the three extra points at the end.
    left = min(point.x for point in points)
    right = max(point.x for point in points)
    bottom = min(point.y for point in points)
    top = max(point.y for point in points)
    # The super-triangle points must be far away so they don't disrupt the
    # "real" part of the triangulation.
    a_lot = max(top - bottom, left - right) * 100
    return (
        geom.Point(left - a_lot, top + a_lot),
        geom.Point(left - a_lot * 2, bottom - a_lot),
        geom.Point(right + a_lot, (top - bottom) / 2),
    )


def _polygon_sides(shape: tuple[geom.Point, ...]
                   ) -> tuple[tuple[geom.Point, geom.Point], ...]:
    return tuple(zip(shape, shape[1:] + (shape[0],)))


def _triangle_circumcenter(triangle: tuple[geom.Point, geom.Point, geom.Point]
                           ) -> geom.Point:
    """Finds the circumcenter of a triangle.

    For any triangle, there is a unique circle that touches every vertex of the
    triangle. The circumcenter is the center of that circumcircle.

    Not numerically stable - you can get a different answer if you compute this
    for two different orderings of the same triangle points.
    """
    # The general method works as long as at least two sides of the triangle
    # are neither horizontal nor vertical.
    general_position_segments = [
        (a, b) for a, b in _polygon_sides(triangle)
        if a.x != b.x and a.y != b.y]
    if len(general_position_segments) < 2:
        # One side is horizontal and another is vertical. Special case.
        for (a, b) in _polygon_sides(triangle):
            if a.x == b.x:
                center_y = (a.y + b.y) / 2
            if a.y == b.y:
                center_x = (a.x + b.x) / 2
        return geom.Point(center_x, center_y)
    # name the vertices a, b, and c so that the two sides we'll use share vertex b.
    if general_position_segments[0][0] == general_position_segments[1][1]:
        b, c = general_position_segments[0]
        a = general_position_segments[1][0]
    else:
        assert general_position_segments[0][1] == general_position_segments[1][0]
        a, b = general_position_segments[0]
        c = general_position_segments[1][1]
    bc_bisector_slope = - (c.x - b.x) / (c.y - b.y)
    ab_bisector_slope = - (a.x - b.x) / (a.y - b.y)
    # This comes from writing the equations for the two perpendicular bisector
    # lines and solving for their intersection x coordinate.
    x = (-ab_bisector_slope * (a.x + b.x) + bc_bisector_slope * (c.x + b.x)
         + a.y - c.y) / (2 * (bc_bisector_slope - ab_bisector_slope))
    # Then plug back into the equation for one of the lines.
    y = ab_bisector_slope * (x - (a.x + b.x) / 2) + (a.y + b.y) / 2
    # assertion fails due to floating point numerical instability
    # assert y == bc_bisector_slope * (x - (c.x + b.x) / 2) + (c.y + b.y) / 2
    return geom.Point(x, y)


def _circumcircle_contains_point(triangle: tuple[geom.Point, geom.Point, geom.Point],
                                 point: geom.Point) -> bool:
    """Does this point lie inside the circumcircle of this triangle?"""
    center = _triangle_circumcenter(triangle)
    return (point - center).magnitude() < (triangle[0] - center).magnitude()


def _triangle_contains_point(triangle, point) -> bool:
    """The point is located inside or on an edge of the triangle."""
    return all(geom.ccw(*side, point) >= 0
               for side in _polygon_sides(triangle))


def _find_point_in_triangle_tree(
        tree: dict[tuple[geom.Point, geom.Point, geom.Point],
                   set[tuple[geom.Point, geom.Point, geom.Point]]],
        root: tuple[geom.Point, geom.Point, geom.Point],
        point: geom.Point) -> tuple[geom.Point, geom.Point, geom.Point]:
    """log(n) time point-location strategy for Bower-Watson algorithm.

    By "point location" I mean "given a point, find the triangle containing that
    point."

    Using the map that stores the history of what larger triangles were
    replaced by what smaller triangles, we can find the triangle containing
    the given point.
    """
    cur_node = root
    while cur_node in tree:
        for child in tree[cur_node]:
            if _triangle_contains_point(child, point):
                cur_node = child
                break
    return cur_node


def _neighbor_triangles(triangulation: dict[tuple[geom.Point, geom.Point],
                                            tuple[geom.Point, geom.Point, geom.Point]],
                        triangle: tuple[geom.Point, geom.Point, geom.Point],
                        ) -> list[tuple[geom.Point, geom.Point, geom.Point]]:
    """Gets the triangles sharing a side (ie a halfedge) with the given triangle."""
    neighbors = []
    for a, b in _polygon_sides(triangle):
        if (b, a) in triangulation:
            neighbors.append(triangulation[b, a])
    return neighbors


class BowerWatsonResult(NamedTuple):
    # The Delaunay triangulation of the point set (without the "super
    # triangle").
    #
    # This uses a "half-edge" graph representation: If the triangulation
    # contains triangles ABC and DBA (spelled with points in counterclockwise
    # order), then the halfedge AB maps to ABC and the halfedge BA maps to DBA.
    # IE each halfedge maps to the face where that edge is oriented
    # counterclockwise.
    triangulation: dict[tuple[geom.Point, geom.Point],
                        tuple[geom.Point, geom.Point, geom.Point]]
    # The three extra points added to bound the point set during the computation.
    super_triangle: tuple[geom.Point, geom.Point, geom.Point]
    # The triangulation including the extra faces that contain super_triangle points.
    # (This is handy for computing a version of the Voronoi diagram with no
    # unbounded faces.)
    extended_triangulation: dict[tuple[geom.Point, geom.Point],
                                 tuple[geom.Point, geom.Point, geom.Point]]


def delaunay_triangulation(points: Sequence[geom.Point]) -> BowerWatsonResult:
    """Compute the Delaunay triangulation of a point set.

    Based on Wikipedia's pseudocode for the Bower-Watson algorithm.
    """
    points = list(points)
    random.shuffle(points)  # improves speed from quadratic to n*log(n)
    triangulation = {}
    def _add_triangle(triangle: tuple[geom.Point, geom.Point, geom.Point]):
        assert geom.ccw(*triangle) > 0
        triangulation[(triangle[0], triangle[1])] = triangle
        triangulation[(triangle[1], triangle[2])] = triangle
        triangulation[(triangle[2], triangle[0])] = triangle

    def _remove_triangle(triangle: tuple[geom.Point, geom.Point, geom.Point]):
        del triangulation[(triangle[0], triangle[1])]
        del triangulation[(triangle[1], triangle[2])]
        del triangulation[(triangle[2], triangle[0])]

    super_triangle = _get_bounding_triangle(points)
    _add_triangle(super_triangle)  # NB oriented counterclockwise!
    # The "replacement tree" tracks what (large) triangles were replaced with
    # what other (smaller) triangles and lets us do O(log(n)) point location.
    replacement_tree = {}
    # Add the points to the triangulation one at a time, recovering the
    # Delaunay property (all triangle circumcircles are empty) each time.
    for point in points:
        bad_triangles = set()
        candidate_bads = [_find_point_in_triangle_tree(replacement_tree, super_triangle, point)]
        while candidate_bads:
            triangle = candidate_bads.pop()
            if (_circumcircle_contains_point(triangle, point)
                and triangle not in bad_triangles):
                bad_triangles.add(triangle)
                candidate_bads.extend(_neighbor_triangles(triangulation, triangle))
        orphan_sides = set()
        for triangle in bad_triangles:
            for side in _polygon_sides(triangle):
                if (side[1], side[0]) in orphan_sides:
                    orphan_sides.remove((side[1], side[0]))
                else:
                    orphan_sides.add(side)
        # NB new_triangles will all be oriented counterclockwise
        new_triangles = set(side + (point,) for side in orphan_sides)
        for triangle in bad_triangles:
            _remove_triangle(triangle)
            replacement_tree[triangle] = new_triangles
        for triangle in new_triangles:
            _add_triangle(triangle)
    cleaned_triangulation = {
        edge: triangle for edge, triangle in triangulation.items()
        if not any(point in triangle for point in super_triangle)}
    return BowerWatsonResult(
        triangulation=cleaned_triangulation,
        super_triangle=super_triangle,
        extended_triangulation=triangulation,
    )


T = typing.TypeVar("T")


class PriorityQueue(Generic[T]):
    """It's weird that heapq doesn't provide a class like this"""
    def __init__(self, key: Callable[[T], float]) -> None:
        self._values = []
        self._key = key
        self._tiebreaker = 0

    def insert(self, value: T) -> None:
        heapq.heappush(self._values, (self._key(value), self._tiebreaker, value))
        self._tiebreaker += 1

    def pop(self) -> T:
        _, _, value = heapq.heappop(self._values)
        return value

    def __bool__(self) -> bool:
        return bool(self._values)


def _triangle_perimeter(triangle: tuple[geom.Point, geom.Point, geom.Point]) -> float:
    a, b, c = triangle
    return (a - b).magnitude() + (b - c).magnitude() + (c - a).magnitude()


def long_path_through_triangulation(
        triangulation: dict[tuple[geom.Point, geom.Point],
                            tuple[geom.Point, geom.Point, geom.Point]],
        forbidden_points: Optional[set[geom.Point]] = None,
) -> list[geom.Point]:
    """Greedy algorithm to find a long simple cyclic path in a triangulation.

    forbidden_points is a set of vertices that will not be part of the path.
    """
    if forbidden_points is None:
        forbidden_points = set()

    def _queue_key(halfedge: tuple[geom.Point, geom.Point]) -> float:
        # how much we add to the perimeter if we replace (a,b) with the other
        # two edges of its opposing triangle.
        a, b = halfedge
        return _triangle_perimeter(triangulation[(b, a)]) - (a - b).magnitude()

    queue = PriorityQueue(key=_queue_key)
    # Ok so. We're gonna start with a path that's just one triangle. Every
    # round we'll make our path longer by one edge, by replacing an edge with
    # the other two edges of the same triangle (if possible).
    seed_triangle = next(
        triangle for triangle in triangulation.values()
        if not set(triangle).intersection(forbidden_points))
    path: dict[geom.Point, geom.Point] = {}
    for a, b in _polygon_sides(seed_triangle):
        path[a] = b
        if (b, a) in triangulation:
            queue.insert((a, b))
    while queue:
        a, b = queue.pop()  # edge a->b is in our path
        if path[a] != b:
            continue
        [c] = [point for point in triangulation[(b, a)]
               if point != a and point != b]
        if c in path or c in forbidden_points:
            continue
        path[a] = c
        path[c] = b
        for edge in [(a, c), (c, b)]:
            if edge[::-1] in triangulation:
                queue.insert(edge)
    start = seed_triangle[0]
    current = path[start]
    as_list = [start]
    while current != start:
        as_list.append(current)
        current = path[current]
    return as_list


def delaunay_to_voronoi(
        triangulation: dict[tuple[geom.Point, geom.Point],
                            tuple[geom.Point, geom.Point, geom.Point]],
        super_triangle: tuple[geom.Point, geom.Point, geom.Point],
) -> dict[geom.Point, tuple[geom.Point, ...]]:
    """See https://en.wikipedia.org/wiki/Voronoi_diagram

    Return value maps points (from the triangulation) to the corresponding
    Voronoi cell (as a counterclockwise polygon).

    The result isn't quite a real Voronoi diagram because we use the "super
    triangle" (from the Delaunay triangulation construction) to avoid having
    unbounded cells.
    """
    arbitrary_neighbor = {}
    for a, b in triangulation:
        arbitrary_neighbor[a] = b
    voronoi = {}
    for vertex in arbitrary_neighbor:
        if vertex in super_triangle:
            continue
        start_edge = (vertex, arbitrary_neighbor[vertex])
        cur_edge = start_edge
        face = []
        while True:
            face.append(_triangle_circumcenter(triangulation[cur_edge]))
            [third_point] = [point for point in triangulation[cur_edge]
                             if point not in cur_edge]
            cur_edge = (vertex, third_point)
            if cur_edge == start_edge:
                break
        voronoi[vertex] = tuple(face)
    return voronoi


def _common_voronoi_edge(face0, face1):
    common = set(_polygon_sides(face0)).intersection(
        set(_polygon_sides(face1[::-1])))
    assert len(common) == 1
    return common.pop()


def _sides_where_line_hits_polygon(
        point: geom.Point,
        vector: geom.Vector,
        polygon: tuple[geom.Point, ...]) -> list[tuple[geom.Point, geom.Point]]:
    second_point = point + vector
    result = []
    for a, b in _polygon_sides(polygon):
        if (geom.ccw(point, second_point, a) * geom.ccw(point, second_point, b) < 0
            # if the line hits a vertex, only count one of the edges it forms.
            # And if the polygon side lies completely on the line, ignore it.
            or (geom.ccw(point, second_point, a) == 0
                and not geom.ccw(point, second_point, b) == 0)):
            result.append((a, b))
    return result


def _midpoint(alice: geom.Point, bob: geom.Point) -> geom.Point:
    return geom.Point((alice.x + bob.x) / 2,
                      (alice.y + bob.y) / 2)


def _squiggly_svg(path: Sequence[geom.Point],
                  voronoi: dict[geom.Point, tuple[geom.Point, ...]],
                  size: tuple[float, float],
                  ) -> str:

    start_control_point = path[-1]
    start_edge = _common_voronoi_edge(voronoi[path[-1]], voronoi[path[0]])
    start_point = _midpoint(*start_edge)

    path_commands = [f"M {start_point.x} {start_point.y}"]
    prev_control_point = start_control_point
    entry_edge = start_edge
    entry_point = start_point
    for a, b in zip(path, path[1:]):
        # We draw a quadratic spline from the previous endpoint to the midpoint
        # of the edge shared by the voronoi cells of A and B. The control point
        # is somewhere in/on the voronoi cell of A, and so the quadratic spline,
        # being bounded by the triangle formed of its endpoints and control
        # point, is also bounded by the voronoi cell. Therefore because
        # voronoi cells don't overlap, our splines won't intersect each other.
        exit_edge = _common_voronoi_edge(voronoi[a], voronoi[b])
        exit_point = _midpoint(*exit_edge)
        edges_on_tangent_line = _sides_where_line_hits_polygon(
            entry_point, prev_control_point - entry_point, voronoi[a])
        [control_point_edge] = [side for side in edges_on_tangent_line
                                if side[::-1] != entry_edge]
        control_point = geom.line_intersection(
            entry_point, prev_control_point - entry_point,
            control_point_edge[0], control_point_edge[0] - control_point_edge[1])
        if control_point_edge == exit_edge:
            control_point = _midpoint(control_point, entry_point)
        path_commands.append(
            f"Q {control_point.x} {control_point.y} {exit_point.x} {exit_point.y}")
        entry_point = exit_point
        entry_edge = exit_edge
        prev_control_point = control_point
    result = [
        f'<svg width="{size[0]}" height="{size[1]}" xmlns="http://www.w3.org/2000/svg">']
    result.append(
        f'<rect x="0" y="0" width="{size[0]}" height="{size[1]}" '
        f'fill="{BACKGROUND_COLOR_FLAG.value}" />')
    result.append('<path d="{}" stroke="{}" fill="transparent" />'.format(
        " ".join(path_commands),
        SQUIGGLE_COLOR_FLAG.value,
    ))
    result.append("</svg>")
    return "\n".join(result)


def _load_and_squiggle_image(img_path: str):
    """Loads and squigglifies image, printing SVG to stdout."""
    img = Image.open(img_path)
    points = image_to_points(img)
    bw_result = delaunay_triangulation(points)
    voronoi = delaunay_to_voronoi(bw_result.extended_triangulation,
                                  bw_result.super_triangle)
    external_points = set()
    for vertex, face in voronoi.items():
        # Exclude a vertex from our path if its Voronoi cell isn't strictly
        # inside the image bounds.
        if any(point.x <= 0 or point.y <= 0
               or point.x >= img.size[0] - 1 or point.y >= img.size[1] - 1
               for point in face):
            external_points.add(vertex)
    path = long_path_through_triangulation(bw_result.triangulation, external_points)
    print(_squiggly_svg(path,
                        voronoi,
                        img.size))
    print("Num points", len(points), file=sys.stderr)
    print("Path length", len(path), file=sys.stderr)


def _just_the_points(img_path: str):
    # For fast iteration when experimenting with values in image_to_points
    img = Image.open(img_path)
    points = image_to_points(img)
    img.paste(0xff, (0, 0, img.size[0], img.size[1]))
    for point in points:
        img.putpixel((max(0, int(point.x)), max(0, int(point.y))), 0)
    img.show()


def main(extra_argv):
    if len(extra_argv) > 1:
        raise Exception("Leftover values after flag parsing:", extra_argv[1:])
    if POINTS_ONLY_FLAG.value:
        _just_the_points(INFILE_FLAG.value)
    else:
        _load_and_squiggle_image(INFILE_FLAG.value)


if __name__ == "__main__":
    app.run(main)


# OTHER DEBUGGING STUFF THAT I DID INSTEAD OF UNIT TESTS, heh.
##############################################################

# def triangulation_to_svg(triangulation: dict[tuple[geom.Point, geom.Point],
#                                              tuple[geom.Point, geom.Point, geom.Point]],
#                          width: int,
#                          height: int) -> str:
#     # For debugging Delaunay triangulations
#     lines = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
#     deduped_edges = set(tuple(sorted(edge)) for edge in triangulation)
#     for a, b in deduped_edges:
#         lines.append(
#             f'<line x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}" stroke="black" stroke-width="1" />')
#     lines.append("</svg>")
#     return "\n".join(lines)

# def _load_and_triangulate_image(path: str):
#     img = Image.open(path).convert("L")
#     points = image_to_points(img)
#     triangulation = delaunay_triangulation(points)
#     print(triangulation_to_svg(triangulation, img.size[0], img.size[1]))


# if __name__ == "__main__":
#     _load_and_triangulate_image(sys.argv[1])


# def path_to_svg(path, width, height) -> str:
#     lines = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
#     for a, b in zip(path, path[1:] + [path[0]]):
#         lines.append(
#             f'<line x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}" stroke="black" stroke-width="1" />')
#     lines.append("</svg>")
#     return "\n".join(lines)


# def _load_and_straightsquiggle_image(img_path: str):
#     img = Image.open(img_path).convert("L")
#     points = image_to_points(img)
#     bw_result = delaunay_triangulation(points)
#     triangulation = bw_result.triangulation
#     path = long_path_through_triangulation(triangulation)
#     print(path_to_svg(path, img.size[0], img.size[1]))
#     print("Num points", len(points), file=sys.stderr)
#     print("Path length", len(path), file=sys.stderr)
#     # print(triangulation_to_svg(triangulation, img.size[0], img.size[1]))


# if __name__ == "__main__":
#     _load_and_straightsquiggle_image(sys.argv[1])
