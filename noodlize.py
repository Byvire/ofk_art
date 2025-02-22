# Creates a version of an image that looks like all one big long noodle.


# Convert to grayscale.
# Find Delaunay triangulation.
# Find hamiltonian (or at least long) path through Delaunay triangulation.
#

import random
import sys
from typing import Sequence
from PIL import Image

from potato_sauce import geom  # github.com/Byvire/potato_sauce




def _get_bounding_triangle(points: Sequence[geom.Point]) -> tuple[geom.Point, geom.Point, geom.Point]:
    # The Bower-Watson algorithm for constructing a Delaunay triangulation only
    # works if you start with one big triangle that bounds the rest of the
    # point set. You remove the three extra points at the end.
    left = min(point.x for point in points)
    right = max(point.x for point in points)
    bottom = min(point.y for point in points)
    top = max(point.y for point in points)
    return (
        geom.Point(left - 1, top + (top - bottom + 1) * 3),
        geom.Point(left - 10, top - (top - bottom + 1) * 3),
        geom.Point(right + (right - left + 1) * 3, (top - bottom) / 2),
    )


def _triangle_sides(triangle: tuple[geom.Point, geom.Point, geom.Point]
                    ) -> tuple[tuple[geom.Point, geom.Point],
                               tuple[geom.Point, geom.Point],
                               tuple[geom.Point, geom.Point]]:
    return ((triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]))

def _triangle_circumcenter(triangle: tuple[geom.Point, geom.Point, geom.Point]
                           ) -> geom.Point:
    """Finds the circumcenter of a triangle.

    For any triangle, there is a unique circle that touches every vertex of the
    triangle. The circumcenter is the center of that circumcircle.
    """
    # The general method works as long as at least two sides of the triangle
    # are neither horizontal nor vertical.
    general_position_segments = [
        (a, b) for a, b in _triangle_sides(triangle)
        if a.x != b.x and a.y != b.y]
    if len(general_position_segments) < 2:
        # One side is horizontal and another is vertical. Special case.
        for (a, b) in _triangle_sides(triangle):
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
    center = _triangle_circumcenter(triangle)
    return (point - center).magnitude() < (triangle[0] - center).magnitude()



# This triangulation procedure is based on Wikipedia's pseudocode for the
# Bower-Watson algorithm.


def delaunay_triangulation(points: Sequence[geom.Point]
                           # If the triangulation contains triangles ABC and
                           # DBA (spelled with points in counterclockwise
                           # order), then the pair AB maps to ABC and the pair BA maps to DBA.
                           # IE each directed edge maps to the face where the
                           # edge is oriented counterclockwise.
                           ) -> dict[tuple[geom.Point, geom.Point],
                                     tuple[geom.Point, geom.Point, geom.Point]]:
    triangulation = {}
    def _add_triangle(triangle: tuple[geom.Point, geom.Point, geom.Point]):
        assert geom.ccw(*triangle)
        triangulation[(triangle[0], triangle[1])] = triangle
        triangulation[(triangle[1], triangle[2])] = triangle
        triangulation[(triangle[2], triangle[0])] = triangle

    def _remove_triangle(triangle: tuple[geom.Point, geom.Point, geom.Point]):
        del triangulation[(triangle[0], triangle[1])]
        del triangulation[(triangle[1], triangle[2])]
        del triangulation[(triangle[2], triangle[0])]

    super_triangle = _get_bounding_triangle(points)
    _add_triangle(super_triangle)

    # Add the points to the triangulation one at a time, recovering the
    # Delaunay property (all triangle circumcircles are empty) each time.
    for point in points:
        bad_triangles = set()
        for triangle in set(triangulation.values()):
            # Looping over all triangles is inefficient but simpler than doing
            # fast point location.
            if _circumcircle_contains_point(triangle, point):
                bad_triangles.add(triangle)
                _remove_triangle(triangle)
        orphan_sides = set()
        for triangle in bad_triangles:
            for side in _triangle_sides(triangle):
                if (side[1], side[0]) in orphan_sides:
                    orphan_sides.remove((side[1], side[0]))
                else:
                    orphan_sides.add(side)
        for side in orphan_sides:
            _add_triangle(side + (point,))
    for triangle in set(triangulation.values()):
        if any(point in triangle for point in super_triangle):
            _remove_triangle(triangle)
    return triangulation



def grayscale_image_to_points(img: Image.Image) -> list[geom.Point]:
    points = []
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if random.random() < 0.01 * (255 - img.getpixel((x, y))) / 255:
                points.append(geom.Point(x, y))
    print("Point cloud size:", len(points), file=sys.stderr)
    return points



def triangulation_to_svg(triangulation: dict[tuple[geom.Point, geom.Point],
                                             tuple[geom.Point, geom.Point, geom.Point]],
                         width: int,
                         height: int) -> str:
    lines = [f'<svg width="{width}" height="{height}" xmlns="http://www.w3.org/2000/svg">']
    deduped_edges = set(tuple(sorted(edge)) for edge in triangulation)
    for a, b in deduped_edges:
        lines.append(
            f'<line x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}" stroke="black" stroke-width="1" />')
    lines.append("</svg>")
    return "\n".join(lines)


def _load_and_triangulate_image(path: str):
    img = Image.open(path).convert("L")
    points = grayscale_image_to_points(img)
    triangulation = delaunay_triangulation(points)
    print(triangulation_to_svg(triangulation, img.size[0], img.size[1]))


if __name__ == "__main__":
    _load_and_triangulate_image(sys.argv[1])



# if __name__ == "__main__":
#     # TEST OF DELAUNAY TRIANGULATION. IT WORKS LOL
#     WIDTH = 512
#     HEIGHT = 512
#     NUM_POINTS = 1000
#     points = [geom.Point(x, y)
#               for x, y in  zip(
#                       random.choices(range(1, WIDTH - 1), k=NUM_POINTS),
#                       random.choices(range(1, HEIGHT - 1), k=NUM_POINTS))]
#     points = list(set(points))
#     triangulation = delaunay_triangulation(points)

#     print(f'<svg width="{WIDTH}" height="{HEIGHT}" xmlns="http://www.w3.org/2000/svg">')
#     for a, b in triangulation:
#         print(f'<line x1="{a.x}" y1="{a.y}" x2="{b.x}" y2="{b.y}" stroke="black" stroke-width="1" />')
#     # for triangle in set(triangulation.values()):
#     #     center = _triangle_circumcenter(triangle)
#     #     radius = (triangle[0] - center).magnitude()
#     #     print(f'<circle cx="{center[0]}" cy="{center[1]}" r="{radius}" fill="green" fill-opacity="0.3" />')

#     print("</svg>")
