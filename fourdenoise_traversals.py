"""Standalone traversal indices and reversible array unfolding."""

from copy import deepcopy
from functools import lru_cache
import numpy as np


def _row_major_indices(shape):
    """Return row-major ``(y, x)`` traversal indices for a 2D grid."""
    y_size, x_size = shape
    return np.array(
        [(y_idx, x_idx) for y_idx in range(y_size) for x_idx in range(x_size)],
        dtype=int,
    )


def _serpentine_indices(shape):
    """
    Return left-to-right/right-to-left alternating row traversal indices.

    This keeps consecutive elements adjacent when crossing from one row to the
    next, which is useful for scan paths that physically snake through the grid.
    """
    y_size, x_size = shape
    indices = []
    for y_idx in range(y_size):
        if y_idx % 2 == 0:
            x_range = range(x_size)
        else:
            x_range = range(x_size - 1, -1, -1)
        indices.extend((y_idx, x_idx) for x_idx in x_range)
    return np.array(indices, dtype=int)


def _spiral_indices(shape):
    """
    Return top-left clockwise inward spiral traversal indices for a 2D grid.

    The path starts across the top row, moves down the right column, then left
    across the bottom row, up the left column, and repeats inward.
    """
    y_size, x_size = shape
    top = 0
    bottom = y_size - 1
    left = 0
    right = x_size - 1
    indices = []

    while top <= bottom and left <= right:
        for x_idx in range(left, right + 1):
            indices.append((top, x_idx))
        top += 1

        for y_idx in range(top, bottom + 1):
            indices.append((y_idx, right))
        right -= 1

        if top <= bottom:
            for x_idx in range(right, left - 1, -1):
                indices.append((bottom, x_idx))
            bottom -= 1

        if left <= right:
            for y_idx in range(bottom, top - 1, -1):
                indices.append((y_idx, left))
            left += 1

    return np.array(indices, dtype=int)


def _diagonal_zigzag_indices(shape):
    """
    Traverse anti-diagonals while alternating direction on each diagonal.

    This is not a fractal space-filling curve, but it is a simple
    locality-preserving baseline used in image compression and scanning.
    """
    y_size, x_size = shape
    indices = []
    for diag in range(y_size + x_size - 1):
        y_start = max(0, diag - x_size + 1)
        y_stop = min(y_size - 1, diag)
        diagonal = [(y_idx, diag - y_idx) for y_idx in range(y_start, y_stop + 1)]
        if diag % 2 == 0:
            diagonal.reverse()
        indices.extend(diagonal)
    return np.array(indices, dtype=int)


def _hilbert_d2yx(side, distance):
    """
    Convert Hilbert distance to ``(y, x)`` for a power-of-2 square.

    This is the standard iterative d2xy Hilbert algorithm. It is written in
    ``(x, y)`` internally and returned as ``(y, x)`` to match array indexing.
    """
    x_coord = 0
    y_coord = 0
    t_val = int(distance)
    scale = 1

    while scale < side:
        rx = 1 & (t_val // 2)
        ry = 1 & (t_val ^ rx)
        if ry == 0:
            if rx == 1:
                x_coord = scale - 1 - x_coord
                y_coord = scale - 1 - y_coord
            x_coord, y_coord = y_coord, x_coord

        x_coord += scale * rx
        y_coord += scale * ry
        t_val //= 4
        scale *= 2

    return y_coord, x_coord


def _hilbert_indices(shape):
    """Return a discrete Hilbert traversal for a power-of-2 square."""
    side_y, side_x = shape
    if side_y != side_x or not _is_power_of(side_y, 2):
        raise ValueError("Hilbert traversal requires a power-of-2 square.")
    return np.array(
        [_hilbert_d2yx(side_y, distance) for distance in range(side_y * side_y)],
        dtype=int,
    )


def _morton_code(y_idx, x_idx):
    """Interleave the binary bits of ``y`` and ``x`` into a Morton code."""
    code = 0
    bit = 0
    max_val = max(int(y_idx), int(x_idx))
    while (1 << bit) <= max_val:
        code |= ((x_idx >> bit) & 1) << (2 * bit)
        code |= ((y_idx >> bit) & 1) << (2 * bit + 1)
        bit += 1
    return code


def _morton_indices(shape):
    """
    Return Morton/Z-order traversal for a power-of-2 square.

    Coordinates are sorted by interleaving the binary bits of row and column
    indices. This is less continuous than Hilbert, but useful as a hierarchical
    locality-preserving baseline.
    """
    side_y, side_x = shape
    if side_y != side_x or not _is_power_of(side_y, 2):
        raise ValueError("Morton/Z-order traversal requires a power-of-2 square.")
    coords = [(y_idx, x_idx) for y_idx in range(side_y) for x_idx in range(side_x)]
    coords.sort(key=lambda coord: _morton_code(coord[0], coord[1]))
    return np.array(coords, dtype=int)


def _peano_indices(shape):
    """
    Return a recursive Peano traversal for a power-of-3 square.

    The first-order path traverses a 3x3 square in a ``2``-shaped motif. Higher
    orders recursively orient each child block so adjacent children connect by a
    nearest-neighbor step. This alternates ``2``- and ``S``-like motifs across
    the grid, avoiding the disconnected repeated-block pattern that a naive
    serpentine recursion would produce.
    """
    side_y, side_x = shape
    if side_y != side_x or not _is_power_of(side_y, 3):
        raise ValueError("Peano traversal requires a power-of-3 square.")

    order = 0
    side = side_y
    while side > 1:
        side //= 3
        order += 1

    return np.array(_peano_path(order), dtype=int)


def _opposite_corner(corner):
    """Return the diagonally opposite corner name for a square block."""
    opposites = {
        'NW': 'SE',
        'SE': 'NW',
        'NE': 'SW',
        'SW': 'NE',
    }
    return opposites[corner]


def _peano_base_path(start_corner='NW', end_corner='SE'):
    """
    Return an oriented 3x3 Peano base path between opposite corners.

    The canonical orientation starts at ``NW`` and ends at ``SE``:
    right, right, down, left, left, down, right, right. Reflections provide the
    other opposite-corner orientations while preserving nearest-neighbor steps.
    """
    if _opposite_corner(start_corner) != end_corner:
        raise ValueError(
            "Peano child curves require opposite start and end corners."
        )

    path = np.array(
        [
            (0, 0), (0, 1), (0, 2),
            (1, 2), (1, 1), (1, 0),
            (2, 0), (2, 1), (2, 2),
        ],
        dtype=int,
    )

    if start_corner in {'SW', 'SE'}:
        path[:, 0] = 2 - path[:, 0]
    if start_corner in {'NE', 'SE'}:
        path[:, 1] = 2 - path[:, 1]

    return path


def _peano_next_entry_corner(direction, exit_corner):
    """Return the child entry corner compatible with a neighboring block step."""
    compatible_entries = {
        (0, 1): {'NE': 'NW', 'SE': 'SW'},
        (0, -1): {'NW': 'NE', 'SW': 'SE'},
        (1, 0): {'SW': 'NW', 'SE': 'NE'},
        (-1, 0): {'NW': 'SW', 'NE': 'SE'},
    }
    direction = tuple(int(v) for v in direction)
    if direction not in compatible_entries:
        raise ValueError(f"Invalid Peano block direction {direction}.")
    if exit_corner not in compatible_entries[direction]:
        raise ValueError(
            f"Peano child exit corner '{exit_corner}' is incompatible with "
            f"block direction {direction}."
        )
    return compatible_entries[direction][exit_corner]


def _peano_path(order, start_corner='NW', end_corner='SE'):
    """Return recursive Peano coordinates for a ``3**order`` square."""
    if order < 0:
        raise ValueError("order must be non-negative.")
    if order == 0:
        return ((0, 0),)

    macro_path = _peano_base_path(start_corner, end_corner)
    if order == 1:
        return tuple(map(tuple, macro_path))

    sub_side = 3 ** (order - 1)
    coords = []
    child_start = start_corner

    for block_idx, block in enumerate(macro_path):
        if block_idx < len(macro_path) - 1:
            direction = macro_path[block_idx + 1] - block
            child_end = _opposite_corner(child_start)
            next_child_start = _peano_next_entry_corner(direction, child_end)
        else:
            child_end = end_corner
            next_child_start = None
            if _opposite_corner(child_start) != child_end:
                raise RuntimeError(
                    "Could not orient the final Peano child block continuously."
                )

        child = _peano_path(order - 1, child_start, child_end)
        offset = block * sub_side
        coords.extend(
            (y_idx + offset[0], x_idx + offset[1])
            for y_idx, x_idx in child
        )
        child_start = next_child_start

    return tuple(coords)


def _corner_coordinate(corner, side):
    """Return the local ``(y, x)`` coordinate of a named square corner."""
    corners = {
        'NW': (0, 0),
        'NE': (0, side - 1),
        'SW': (side - 1, 0),
        'SE': (side - 1, side - 1),
    }
    return corners[corner]


def _compatible_corner_pairs(direction):
    """Return adjacent exit/entry corner pairs for neighboring 3x3 blocks."""
    if direction == (0, 1):
        return [('NE', 'NW'), ('SE', 'SW')]
    if direction == (0, -1):
        return [('NW', 'NE'), ('SW', 'SE')]
    if direction == (1, 0):
        return [('SW', 'NW'), ('SE', 'NE')]
    if direction == (-1, 0):
        return [('NW', 'SW'), ('NE', 'SE')]
    raise ValueError(f"Invalid block direction {direction}.")


def _base_meander_path(start_corner, end_corner):
    """Find a 3x3 Hamiltonian path between two square corners."""
    start = _corner_coordinate(start_corner, 3)
    end = _corner_coordinate(end_corner, 3)
    if start == end:
        raise ValueError("start_corner and end_corner must differ.")

    path = [start]
    visited = {start}

    def neighbor_score(cell):
        y_idx, x_idx = cell
        onward = 0
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (y_idx + dy, x_idx + dx)
            if (
                0 <= neighbor[0] < 3
                and 0 <= neighbor[1] < 3
                and neighbor not in visited
            ):
                onward += 1
        distance_to_end = abs(y_idx - end[0]) + abs(x_idx - end[1])
        return onward, distance_to_end

    def dfs(cell):
        if len(path) == 9:
            return cell == end

        y_idx, x_idx = cell
        candidates = []
        for dy, dx in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
            neighbor = (y_idx + dy, x_idx + dx)
            if 0 <= neighbor[0] < 3 and 0 <= neighbor[1] < 3:
                if neighbor in visited:
                    continue
                if neighbor == end and len(path) != 8:
                    continue
                candidates.append(neighbor)

        candidates.sort(key=neighbor_score)
        for neighbor in candidates:
            visited.add(neighbor)
            path.append(neighbor)
            if dfs(neighbor):
                return True
            path.pop()
            visited.remove(neighbor)
        return False

    if not dfs(start):
        raise RuntimeError(
            f"Could not construct 3x3 meander path from {start_corner} to {end_corner}."
        )
    return tuple(path)


def _meander_block_template(start_corner, end_corner):
    """
    Build a 3x3 block traversal with corner-aware child orientations.

    This alternates local 2-like and S-like motifs so neighboring child curves
    connect continuously across block boundaries.
    """
    corner_blocks = {
        'NW': (0, 0),
        'NE': (0, 2),
        'SW': (2, 0),
        'SE': (2, 2),
    }
    start_block = corner_blocks[start_corner]
    end_block = corner_blocks[end_corner]
    path = [(start_block, start_corner, None)]
    visited = {start_block}

    def ordered_neighbors(block):
        y_idx, x_idx = block
        if y_idx % 2 == 0:
            directions = [(0, 1), (1, 0), (0, -1), (-1, 0)]
        else:
            directions = [(0, -1), (1, 0), (0, 1), (-1, 0)]

        for direction in directions:
            neighbor = (y_idx + direction[0], x_idx + direction[1])
            if 0 <= neighbor[0] < 3 and 0 <= neighbor[1] < 3:
                yield neighbor, direction

    def dfs(current_block, current_start):
        if len(path) == 9:
            if current_block != end_block or current_start == end_corner:
                return False
            path[-1] = (current_block, current_start, end_corner)
            return True

        for next_block, direction in ordered_neighbors(current_block):
            if next_block in visited:
                continue
            remaining = 9 - len(path)
            if next_block == end_block and remaining != 1:
                continue

            for exit_corner, entry_corner in _compatible_corner_pairs(direction):
                if exit_corner == current_start:
                    continue
                path[-1] = (current_block, current_start, exit_corner)
                visited.add(next_block)
                path.append((next_block, entry_corner, None))
                if dfs(next_block, entry_corner):
                    return True
                path.pop()
                visited.remove(next_block)
                path[-1] = (current_block, current_start, None)

        return False

    if not dfs(start_block, start_corner):
        raise RuntimeError(
            f"Could not construct meander block from {start_corner} to {end_corner}."
        )
    return tuple(path)


def _peano_meander_path(order, start_corner='NW', end_corner='SE'):
    """
    Return a recursive Peano-meander path for ``3**order`` square grids.

    This is not a mathematically strict Peano curve. It is a Peano-like
    ternary meander that fills powers-of-3 grids with continuous nearest-neighbor
    steps while alternating child-curve orientation for better block-to-block
    continuity.
    """
    if order == 1:
        return _base_meander_path(start_corner, end_corner)

    sub_side = 3 ** (order - 1)
    coords = []
    for (block_y, block_x), child_start, child_end in _meander_block_template(
        start_corner,
        end_corner,
    ):
        child = _peano_meander_path(order - 1, child_start, child_end)
        coords.extend(
            (y_idx + block_y * sub_side, x_idx + block_x * sub_side)
            for y_idx, x_idx in child
        )
    return tuple(coords)


def _peano_meander_indices(shape):
    """
    Return a Peano-like ternary meander traversal for a power-of-3 square.

    This curve is useful when a continuous power-of-3 locality-preserving path is
    desired, while avoiding the repeated local motif in the simpler Peano-style
    implementation.
    """
    side_y, side_x = shape
    if side_y != side_x or not _is_power_of(side_y, 3):
        raise ValueError("Peano-meander traversal requires a power-of-3 square.")

    order = 0
    side = side_y
    while side > 1:
        side //= 3
        order += 1

    return np.array(_peano_meander_path(order), dtype=int)


_MEANDER4_BLOCK_SIZE = 4


_MEANDER4_CORNERS = {
    'NW': (0, 0),
    'NE': (0, _MEANDER4_BLOCK_SIZE - 1),
    'SW': (_MEANDER4_BLOCK_SIZE - 1, 0),
    'SE': (_MEANDER4_BLOCK_SIZE - 1, _MEANDER4_BLOCK_SIZE - 1),
}


_MEANDER4_SIDE_EXIT_CORNERS = {
    'N': ('NW', 'NE'),
    'E': ('NE', 'SE'),
    'S': ('SW', 'SE'),
    'W': ('NW', 'SW'),
}


_MEANDER4_DIRECTION_TO_SIDE = {
    (0, 1): 'E',
    (0, -1): 'W',
    (1, 0): 'S',
    (-1, 0): 'N',
}


_MEANDER4_NEXT_ENTRY = {
    ((0, 1), 'NE'): 'NW',
    ((0, 1), 'SE'): 'SW',
    ((0, -1), 'NW'): 'NE',
    ((0, -1), 'SW'): 'SE',
    ((1, 0), 'SW'): 'NW',
    ((1, 0), 'SE'): 'NE',
    ((-1, 0), 'NW'): 'SW',
    ((-1, 0), 'NE'): 'SE',
}


_MEANDER4_BASE_NW_NE = np.array(
    [
        (0, 0),
        (1, 0), (2, 0), (3, 0),
        (3, 1), (3, 2), (3, 3),
        (2, 3), (1, 3),
        (1, 2),
        (2, 2),
        (2, 1),
        (1, 1), (0, 1),
        (0, 2), (0, 3),
    ],
    dtype=int,
)


def _meander4_transform_path(path, transform_name):
    """Apply a square symmetry transform to a 4x4 meander tile path."""
    size = _MEANDER4_BLOCK_SIZE
    y_idx = path[:, 0]
    x_idx = path[:, 1]

    if transform_name == 'identity':
        transformed = np.column_stack([y_idx, x_idx])
    elif transform_name == 'rot90':
        transformed = np.column_stack([x_idx, size - 1 - y_idx])
    elif transform_name == 'rot180':
        transformed = np.column_stack([size - 1 - y_idx, size - 1 - x_idx])
    elif transform_name == 'rot270':
        transformed = np.column_stack([size - 1 - x_idx, y_idx])
    elif transform_name == 'flip_y':
        transformed = np.column_stack([size - 1 - y_idx, x_idx])
    elif transform_name == 'flip_x':
        transformed = np.column_stack([y_idx, size - 1 - x_idx])
    elif transform_name == 'diag':
        transformed = np.column_stack([x_idx, y_idx])
    elif transform_name == 'anti_diag':
        transformed = np.column_stack([size - 1 - x_idx, size - 1 - y_idx])
    else:
        raise ValueError(f"Unknown meander-4 transform '{transform_name}'.")

    return transformed.astype(int)


def _meander4_corner_name(coord):
    """Return the named 4x4 block corner at ``coord``."""
    coord = tuple(int(v) for v in coord)
    for name, corner in _MEANDER4_CORNERS.items():
        if coord == corner:
            return name
    return None


def _meander4_tile_paths():
    """
    Return oriented 4x4 meander tile paths between adjacent corners.

    The canonical tile is the user-specified path from ``NW`` to ``NE``:
    down x3, right x3, up x2, left, down, left, up x2, right x2. Direct square
    symmetries generate the remaining orientations while preserving that motif.
    """
    transforms = (
        'identity',
        'rot90',
        'rot180',
        'rot270',
        'flip_y',
        'flip_x',
        'diag',
        'anti_diag',
    )
    tile_paths = {}
    for transform_name in transforms:
        path = _meander4_transform_path(_MEANDER4_BASE_NW_NE, transform_name)
        start_corner = _meander4_corner_name(path[0])
        end_corner = _meander4_corner_name(path[-1])
        tile_paths[(start_corner, end_corner)] = path
    return tile_paths


_MEANDER4_TILE_PATHS = _meander4_tile_paths()


def _meander4_corners_are_adjacent(first_corner, second_corner):
    """Return True when two 4x4 block corners share a block side."""
    first_y, first_x = _MEANDER4_CORNERS[first_corner]
    second_y, second_x = _MEANDER4_CORNERS[second_corner]
    distance = abs(first_y - second_y) + abs(first_x - second_x)
    return distance == _MEANDER4_BLOCK_SIZE - 1


def _meander4_choose_exit_corner(entry_corner, direction):
    """Choose the exit corner for a block-to-block step."""
    side = _MEANDER4_DIRECTION_TO_SIDE[direction]
    for exit_corner in _MEANDER4_SIDE_EXIT_CORNERS[side]:
        if (
            exit_corner != entry_corner
            and _meander4_corners_are_adjacent(entry_corner, exit_corner)
        ):
            return exit_corner
    raise ValueError(
        f"No meander-4 exit corner for entry='{entry_corner}', direction={direction}."
    )


def _meander4_indices(shape):
    """
    Return a 4x4 Greek-key meander traversal for dimensions divisible by 4.

    The array is divided into 4x4 tiles. The first tile follows the exact local
    step pattern ``down x3, right x3, up x2, left, down, left, up x2, right x2``.
    Tiles are then visited along an inward spiral, with each tile orientation
    chosen so neighboring tiles connect by a one-pixel step. Rectangular
    ``4*Ny`` by ``4*Nx`` grids are supported.
    """
    height, width = tuple(int(v) for v in shape)
    if height <= 0 or width <= 0:
        raise ValueError("meander-4 traversal shape must be positive.")
    if height % _MEANDER4_BLOCK_SIZE != 0 or width % _MEANDER4_BLOCK_SIZE != 0:
        raise ValueError(
            "meander-4 traversal requires both dimensions to be multiples of 4."
        )

    block_rows = height // _MEANDER4_BLOCK_SIZE
    block_cols = width // _MEANDER4_BLOCK_SIZE
    block_order = [tuple(coord) for coord in _spiral_indices((block_rows, block_cols))]
    coords = []
    entry_corner = 'NW'

    for block_idx, (block_y, block_x) in enumerate(block_order):
        if block_idx < len(block_order) - 1:
            next_block = block_order[block_idx + 1]
            direction_to_next = (
                next_block[0] - block_y,
                next_block[1] - block_x,
            )
            exit_corner = _meander4_choose_exit_corner(
                entry_corner,
                direction_to_next,
            )
            next_entry_corner = _MEANDER4_NEXT_ENTRY[
                (direction_to_next, exit_corner)
            ]
        else:
            for candidate_corner in ('NE', 'SE', 'SW', 'NW'):
                if (
                    candidate_corner != entry_corner
                    and _meander4_corners_are_adjacent(
                        entry_corner,
                        candidate_corner,
                    )
                ):
                    exit_corner = candidate_corner
                    break
            next_entry_corner = None

        local_path = _MEANDER4_TILE_PATHS[(entry_corner, exit_corner)]
        offset = np.array(
            [
                block_y * _MEANDER4_BLOCK_SIZE,
                block_x * _MEANDER4_BLOCK_SIZE,
            ]
        )
        coords.extend(
            (y_idx + offset[0], x_idx + offset[1])
            for y_idx, x_idx in local_path
        )
        entry_corner = next_entry_corner

    return np.array(coords, dtype=int)


_MEANDER5_BLOCK_SIZE = 5


_MEANDER5_SIDE_COORDS = {
    'N': (0, 2),
    'E': (2, 4),
    'S': (4, 2),
    'W': (2, 0),
    'C': (2, 2),
}


_MEANDER5_DIRECTION_TO_SIDE = {
    (0, 1): 'E',
    (0, -1): 'W',
    (1, 0): 'S',
    (-1, 0): 'N',
}


_MEANDER5_OPPOSITE_SIDE = {
    'N': 'S',
    'S': 'N',
    'E': 'W',
    'W': 'E',
}


_MEANDER5_STEP_DIRECTIONS = ((0, 1), (1, 0), (0, -1), (-1, 0))


_MEANDER5_SPIRAL_RANK = {
    tuple(coord): idx
    for idx, coord in enumerate(_spiral_indices((_MEANDER5_BLOCK_SIZE, _MEANDER5_BLOCK_SIZE)))
}


@lru_cache(maxsize=None)
def _meander5_block_path(entry_side, exit_side):
    """
    Return a 5x5 Greek-key block traversal between two block endpoints.

    The endpoints lie on side midpoints, or on the center for the final block.
    A small depth-first Hamiltonian search is used once per entry/exit pair and
    cached; the scoring favors perimeter-to-interior squared turns, giving the
    Greek-key hook motif while preserving one-pixel steps.
    """
    if entry_side not in _MEANDER5_SIDE_COORDS:
        raise ValueError(f"Invalid meander-5 entry side '{entry_side}'.")
    if exit_side not in _MEANDER5_SIDE_COORDS:
        raise ValueError(f"Invalid meander-5 exit side '{exit_side}'.")

    block_size = _MEANDER5_BLOCK_SIZE
    start = _MEANDER5_SIDE_COORDS[entry_side]
    end = _MEANDER5_SIDE_COORDS[exit_side]
    if start == end:
        raise ValueError("meander-5 entry and exit endpoints must differ.")

    total = block_size * block_size
    visited = {start}
    path = [start]

    def connectivity_ok():
        remaining = {
            (y_idx, x_idx)
            for y_idx in range(block_size)
            for x_idx in range(block_size)
            if (y_idx, x_idx) not in visited
        }
        if not remaining:
            return True

        stack = [next(iter(remaining))]
        seen = {stack[0]}
        while stack:
            y_idx, x_idx = stack.pop()
            for dy, dx in _MEANDER5_STEP_DIRECTIONS:
                neighbor = (y_idx + dy, x_idx + dx)
                if neighbor in remaining and neighbor not in seen:
                    seen.add(neighbor)
                    stack.append(neighbor)
        return len(seen) == len(remaining)

    def onward_count(cell):
        y_idx, x_idx = cell
        count = 0
        for dy, dx in _MEANDER5_STEP_DIRECTIONS:
            neighbor = (y_idx + dy, x_idx + dx)
            if (
                0 <= neighbor[0] < block_size
                and 0 <= neighbor[1] < block_size
                and neighbor not in visited
            ):
                if neighbor == end and len(path) != total - 1:
                    continue
                count += 1
        return count

    def neighbor_score(cell, previous):
        py, px = previous
        cy, cx = cell
        straight_penalty = 0
        if len(path) >= 2:
            ay, ax = path[-2]
            old_direction = (py - ay, px - ax)
            new_direction = (cy - py, cx - px)
            straight_penalty = int(old_direction == new_direction)

        return (
            onward_count(cell),
            _MEANDER5_SPIRAL_RANK[cell],
            straight_penalty,
            abs(cy - end[0]) + abs(cx - end[1]),
        )

    def dfs(cell):
        if len(path) == total:
            return cell == end

        y_idx, x_idx = cell
        candidates = []
        for dy, dx in _MEANDER5_STEP_DIRECTIONS:
            neighbor = (y_idx + dy, x_idx + dx)
            if not (0 <= neighbor[0] < block_size and 0 <= neighbor[1] < block_size):
                continue
            if neighbor in visited:
                continue
            if neighbor == end and len(path) != total - 1:
                continue
            candidates.append(neighbor)

        candidates.sort(key=lambda neighbor: neighbor_score(neighbor, cell))
        for neighbor in candidates:
            visited.add(neighbor)
            path.append(neighbor)
            if connectivity_ok() and dfs(neighbor):
                return True
            path.pop()
            visited.remove(neighbor)
        return False

    if not dfs(start):
        raise RuntimeError(
            f"Could not construct meander-5 block path {entry_side}->{exit_side}."
        )
    return tuple(path)


def _meander5_indices(shape):
    """
    Return a Greek-key meander traversal for dimensions divisible by 5.

    The array is divided into 5x5 tiles. Tiles are visited along an inward
    spiral, and each tile is filled by a small squared Greek-key hook whose
    entry and exit sides are oriented to connect continuously to neighboring
    tiles. Rectangular ``5*Ny`` by ``5*Nx`` grids are supported.
    """
    height, width = tuple(int(v) for v in shape)
    if height <= 0 or width <= 0:
        raise ValueError("meander-5 traversal shape must be positive.")
    if height % _MEANDER5_BLOCK_SIZE != 0 or width % _MEANDER5_BLOCK_SIZE != 0:
        raise ValueError(
            "meander-5 traversal requires both dimensions to be multiples of 5."
        )

    block_rows = height // _MEANDER5_BLOCK_SIZE
    block_cols = width // _MEANDER5_BLOCK_SIZE
    block_order = [tuple(coord) for coord in _spiral_indices((block_rows, block_cols))]
    coords = []

    for block_idx, (block_y, block_x) in enumerate(block_order):
        if len(block_order) == 1:
            entry_side = 'W'
            exit_side = 'C'
        elif block_idx == 0:
            next_block = block_order[block_idx + 1]
            direction_to_next = (
                next_block[0] - block_y,
                next_block[1] - block_x,
            )
            exit_side = _MEANDER5_DIRECTION_TO_SIDE[direction_to_next]
            entry_side = _MEANDER5_OPPOSITE_SIDE[exit_side]
        else:
            previous_block = block_order[block_idx - 1]
            direction_to_previous = (
                previous_block[0] - block_y,
                previous_block[1] - block_x,
            )
            entry_side = _MEANDER5_DIRECTION_TO_SIDE[direction_to_previous]
            if block_idx < len(block_order) - 1:
                next_block = block_order[block_idx + 1]
                direction_to_next = (
                    next_block[0] - block_y,
                    next_block[1] - block_x,
                )
                exit_side = _MEANDER5_DIRECTION_TO_SIDE[direction_to_next]
            else:
                exit_side = 'C'

        local_path = _meander5_block_path(entry_side, exit_side)
        offset = np.array(
            [
                block_y * _MEANDER5_BLOCK_SIZE,
                block_x * _MEANDER5_BLOCK_SIZE,
            ]
        )
        coords.extend(
            (y_idx + offset[0], x_idx + offset[1])
            for y_idx, x_idx in local_path
        )

    return np.array(coords, dtype=int)


def _moore_indices(shape):
    """
    Placeholder for a closed Hilbert-like Moore traversal.

    Moore curves are useful, but implementing a clear closed discrete variant is
    a separate step from the current unfolding refactor.
    """
    raise NotImplementedError(
        "method='moore' is registered but not implemented yet. "
        "Use method='hilbert' for an open Hilbert traversal."
    )


_FULL_SHAPE_TRAVERSAL_METHODS = {
    'row_major',
    'serpentine',
    'spiral',
    'diagonal_zigzag',
}


_CURVE_TRAVERSAL_METHODS = {
    'hilbert',
    'morton',
    'peano',
    'peano_meander',
    'moore',
}


_BLOCK_TRAVERSAL_METHODS = {
    'meander-4',
    'meander-5',
}


_TRAVERSAL_INDEX_GENERATORS = {
    'row_major': _row_major_indices,
    'serpentine': _serpentine_indices,
    'spiral': _spiral_indices,
    'diagonal_zigzag': _diagonal_zigzag_indices,
    'hilbert': _hilbert_indices,
    'morton': _morton_indices,
    'peano': _peano_indices,
    'peano_meander': _peano_meander_indices,
    'meander-4': _meander4_indices,
    'meander-5': _meander5_indices,
    'moore': _moore_indices,
}


_TRAVERSAL_METHOD_ALIASES = {
    'z_order': 'morton',
    'meander_4': 'meander-4',
    'meander_5': 'meander-5',
}


def _is_power_of(value, base):
    """Return True when ``value`` is an integer power of ``base``."""
    value = int(value)
    if value < 1:
        return False
    while value % base == 0:
        value //= base
    return value == 1


def _largest_power_leq(value, base):
    """Return the largest power of ``base`` less than or equal to ``value``."""
    value = int(value)
    if value < 1:
        raise ValueError("value must be positive.")
    power = 1
    while power * base <= value:
        power *= base
    return power


def _largest_multiple_leq(value, factor):
    """Return the largest positive multiple of ``factor`` not exceeding ``value``."""
    value = int(value)
    factor = int(factor)
    if value < factor:
        raise ValueError(
            f"value must be at least {factor} for this traversal method."
        )
    return (value // factor) * factor


def _smallest_power_geq(value, base):
    """Return the smallest power of ``base`` greater than or equal to ``value``."""
    value = int(value)
    if value < 1:
        raise ValueError("value must be positive.")
    power = 1
    while power < value:
        power *= base
    return power


def _nearest_power(value, base):
    """
    Return the nearest power of ``base`` to ``value``.

    Ties choose the smaller side to avoid introducing interpolated data unless
    the user explicitly requests upsampling.
    """
    lower = _largest_power_leq(value, base)
    upper = _smallest_power_geq(value, base)
    if abs(value - lower) <= abs(upper - value):
        return lower
    return upper


def _curve_base_for_method(method):
    """Return the natural integer base for a square-compatible traversal."""
    if method == 'peano':
        return 3
    if method == 'peano_meander':
        return 3
    return 2


def _block_size_for_method(method):
    """Return the required tile size for block-compatible traversals."""
    if method == 'meander-4':
        return _MEANDER4_BLOCK_SIZE
    if method == 'meander-5':
        return _MEANDER5_BLOCK_SIZE
    raise ValueError(f"method='{method}' is not a block traversal method.")


def _normalize_traversal_method(method):
    """Normalize method aliases such as ``z_order`` to their implementation."""
    if not isinstance(method, str):
        raise ValueError("method must be a string.")
    method = method.lower()
    return _TRAVERSAL_METHOD_ALIASES.get(method, method)


def _validate_unfold_shape(shape):
    """Validate and normalize a 4D-STEM tensor shape."""
    if shape is None:
        raise ValueError("original_shape is required to undo an unfolding.")
    if len(shape) != 4:
        raise ValueError(
            "Unfolding expects a 4D-STEM shape in the convention "
            "(Ry, Rx, Ky, Kx)."
        )
    shape = tuple(int(v) for v in shape)
    if any(v <= 0 for v in shape):
        raise ValueError("All original_shape dimensions must be positive.")
    return shape


def _normalize_unfold_request(domain='real', method='row_major'):
    """Validate the modern unfolding API: separate domain and method names."""
    if domain is None:
        domain = 'real'
    if method is None:
        method = 'row_major'
    if not isinstance(domain, str):
        raise ValueError("domain must be a string.")

    domain = domain.lower()
    method = _normalize_traversal_method(method)
    if domain not in ('real', 'reciprocal', 'both'):
        raise ValueError("domain must be one of 'real', 'reciprocal', or 'both'.")

    valid_methods = set(_TRAVERSAL_INDEX_GENERATORS) | {'coordinate_aligned'}
    if method not in valid_methods:
        valid = ', '.join(sorted(valid_methods | set(_TRAVERSAL_METHOD_ALIASES)))
        raise ValueError(f"method must be one of: {valid}.")
    if method == 'coordinate_aligned' and domain == 'both':
        raise ValueError("method='coordinate_aligned' supports domain='real' or 'reciprocal', not 'both'.")
    if domain == 'both' and method not in {'row_major', 'morton'}:
        raise NotImplementedError(
            "domain='both' currently supports method='row_major' and "
            "method='morton'. Other traversal pairings need an explicit design."
        )

    return domain, method


def _validate_resize_side(method, resize_side):
    """Validate a user-provided side length for a curve method."""
    resize_side = int(resize_side)
    base = _curve_base_for_method(method)
    if resize_side <= 0 or not _is_power_of(resize_side, base):
        raise ValueError(
            f"resize_side must be a positive power of {base} for method='{method}'."
        )
    return resize_side


def _select_resize_side(shape, method, resize_side=None, resize_side_mode='nearest'):
    """Select a compatible square side for curve resize mode."""
    if resize_side is not None:
        return _validate_resize_side(method, resize_side)

    resize_side_mode = resize_side_mode.lower()
    if resize_side_mode not in ('nearest', 'downsample', 'upsample'):
        raise ValueError("resize_side_mode must be 'nearest', 'downsample', or 'upsample'.")

    target = min(shape)
    base = _curve_base_for_method(method)
    if resize_side_mode == 'nearest':
        return _nearest_power(target, base)
    if resize_side_mode == 'downsample':
        return _largest_power_leq(target, base)
    return _smallest_power_geq(target, base)


def _center_crop_metadata(shape, method):
    """Return centered compatible crop metadata for special traversals."""
    height, width = tuple(int(v) for v in shape)
    if method in _BLOCK_TRAVERSAL_METHODS:
        block_size = _block_size_for_method(method)
        grid_y = _largest_multiple_leq(height, block_size)
        grid_x = _largest_multiple_leq(width, block_size)
    else:
        side = _largest_power_leq(min(height, width), _curve_base_for_method(method))
        grid_y = side
        grid_x = side

    y0 = (height - grid_y) // 2
    x0 = (width - grid_x) // 2
    y1 = y0 + grid_y
    x1 = x0 + grid_x
    return (grid_y, grid_x), y0, y1, x0, x1


def _excess_indices_for_crop(shape, y0, y1, x0, x1):
    """Return coordinates outside a centered compatible square crop."""
    height, width = shape
    excess = [
        (y_idx, x_idx)
        for y_idx in range(height)
        for x_idx in range(width)
        if not (y0 <= y_idx < y1 and x0 <= x_idx < x1)
    ]
    return np.array(excess, dtype=int).reshape((-1, 2))


def _validate_traversal_indices(indices, shape, method):
    """Ensure traversal coordinates are unique and within ``shape``."""
    indices = np.asarray(indices, dtype=int)
    if indices.ndim != 2 or indices.shape[1] != 2:
        raise ValueError(f"Traversal method '{method}' must return an (N, 2) array.")
    if indices.size == 0:
        raise ValueError(f"Traversal method '{method}' returned no coordinates.")

    height, width = shape
    if np.any(indices[:, 0] < 0) or np.any(indices[:, 0] >= height):
        raise ValueError(f"Traversal method '{method}' returned out-of-bounds y coordinates.")
    if np.any(indices[:, 1] < 0) or np.any(indices[:, 1] >= width):
        raise ValueError(f"Traversal method '{method}' returned out-of-bounds x coordinates.")

    if len(set(map(tuple, indices))) != len(indices):
        raise ValueError(f"Traversal method '{method}' returned duplicate coordinates.")
    return indices


def _get_traversal_indices(shape, method, curve_shape_strategy='center_crop',
                           resize_side=None, resize_side_mode='nearest'):
    """
    Return traversal indices and metadata for a 2D coordinate grid.

    Full-shape methods visit every coordinate in ``shape``. Curve methods use
    either a centered compatible square crop or a resized compatible square.
    Block methods such as ``meander-4`` and ``meander-5`` use a centered
    compatible rectangular crop whose dimensions are multiples of the required
    block size.
    """
    method = _normalize_traversal_method(method)
    shape = tuple(int(v) for v in shape)
    if len(shape) != 2 or any(v <= 0 for v in shape):
        raise ValueError("Traversal shape must be a pair of positive integers.")
    if method not in _TRAVERSAL_INDEX_GENERATORS:
        valid = ', '.join(sorted(set(_TRAVERSAL_INDEX_GENERATORS) | set(_TRAVERSAL_METHOD_ALIASES)))
        raise ValueError(f"method='{method}' is not a traversal method. Valid methods are: {valid}.")

    if method in _FULL_SHAPE_TRAVERSAL_METHODS:
        indices = np.asarray(_TRAVERSAL_INDEX_GENERATORS[method](shape), dtype=int)
        expected_size = shape[0] * shape[1]
        if indices.shape != (expected_size, 2):
            raise ValueError(
                f"Traversal method '{method}' returned shape {indices.shape}; "
                f"expected {(expected_size, 2)}."
            )
        indices = _validate_traversal_indices(indices, shape, method)
        return indices, {
            'curve_shape_strategy': 'full_shape',
            'traversal_shape': shape,
        }

    if method == 'moore':
        _moore_indices(shape)

    special_methods = _CURVE_TRAVERSAL_METHODS | _BLOCK_TRAVERSAL_METHODS
    if method not in special_methods:
        raise ValueError(f"Unknown traversal method '{method}'.")

    if curve_shape_strategy is None:
        curve_shape_strategy = 'center_crop'
    if not isinstance(curve_shape_strategy, str):
        raise ValueError("curve_shape_strategy must be a string.")
    curve_shape_strategy = curve_shape_strategy.lower()

    if curve_shape_strategy == 'center_crop':
        grid_shape, y0, y1, x0, x1 = _center_crop_metadata(shape, method)
        local_indices = np.asarray(
            _TRAVERSAL_INDEX_GENERATORS[method](grid_shape),
            dtype=int,
        )
        if local_indices.shape != (grid_shape[0] * grid_shape[1], 2):
            raise ValueError(
                f"Traversal method '{method}' did not cover its selected "
                f"{grid_shape} grid exactly once."
            )
        indices = local_indices + np.array([y0, x0])
        indices = _validate_traversal_indices(indices, shape, method)
        traversal_metadata = {
            'curve_shape_strategy': 'center_crop',
            'curve_grid_shape': grid_shape,
            'crop_slices': {'y': (y0, y1), 'x': (x0, x1)},
            'kept_indices': indices,
            'excess_indices': _excess_indices_for_crop(shape, y0, y1, x0, x1),
            'traversal_shape': shape,
        }
        if method in _BLOCK_TRAVERSAL_METHODS:
            block_size = _block_size_for_method(method)
            traversal_metadata.update({
                'block_size': block_size,
                'block_grid_shape': (
                    grid_shape[0] // block_size,
                    grid_shape[1] // block_size,
                ),
            })
        return indices, traversal_metadata

    if curve_shape_strategy == 'resize':
        if method in _BLOCK_TRAVERSAL_METHODS:
            block_size = _block_size_for_method(method)
            raise NotImplementedError(
                f"method='{method}' currently supports "
                "curve_shape_strategy='center_crop' only. Its compatible "
                "domain is a centered rectangle with dimensions divisible by "
                f"{block_size}."
            )
        side = _select_resize_side(
            shape,
            method,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        indices = np.asarray(
            _TRAVERSAL_INDEX_GENERATORS[method]((side, side)),
            dtype=int,
        )
        if indices.shape != (side * side, 2):
            raise ValueError(
                f"Traversal method '{method}' did not cover its resized "
                f"{side}x{side} grid exactly once."
            )
        indices = _validate_traversal_indices(indices, (side, side), method)
        return indices, {
            'curve_shape_strategy': 'resize',
            'curve_grid_shape': (side, side),
            'resize_side': side,
            'resize_side_mode': resize_side_mode,
            'resized_traversal_shape': (side, side),
            'traversal_shape': (side, side),
        }

    raise ValueError("curve_shape_strategy must be 'center_crop' or 'resize'.")


def _inverse_transpose_order(order):
    """Return the inverse permutation for a transpose order."""
    inverse = [0] * len(order)
    for idx, axis in enumerate(order):
        inverse[axis] = idx
    return tuple(inverse)


def _copy_traversal_metadata(prefix, metadata):
    """Copy traversal metadata, optionally namespaced for domain='both'."""
    if prefix is None:
        return dict(metadata)
    return {f"{prefix}_{key}": value for key, value in metadata.items()}


def _build_unfold_metadata(original_shape, domain='real', method='row_major',
                           curve_shape_strategy='center_crop',
                           preserve_excess=True, resize_side=None,
                           resize_side_mode='nearest', resize_method='linear',
                           preserve_original=False, working_shape=None):
    """Build metadata needed to undo an unfolding exactly when possible."""
    original_shape = _validate_unfold_shape(original_shape)
    working_shape = _validate_unfold_shape(working_shape or original_shape)
    domain, method = _normalize_unfold_request(domain=domain, method=method)
    ry, rx, ky, kx = working_shape

    metadata = {
        'version': 2,
        'original_shape': original_shape,
        'working_shape': working_shape,
        'domain': domain,
        'method': method,
        'preserve_excess': bool(preserve_excess),
        'preserve_original': bool(preserve_original),
    }

    if method == 'coordinate_aligned':
        if domain == 'real':
            transpose_order = (0, 2, 1, 3)
            intermediate_shape = (ry, ky, rx, kx)
            output_shape = (ry * ky, rx * kx)
            interpretation = (
                "Rows combine real-space y with reciprocal-space y; columns "
                "combine real-space x with reciprocal-space x."
            )
        else:
            transpose_order = (2, 0, 3, 1)
            intermediate_shape = (ky, ry, kx, rx)
            output_shape = (ky * ry, kx * rx)
            interpretation = (
                "Rows combine reciprocal-space y with real-space y; columns "
                "combine reciprocal-space x with real-space x."
            )

        metadata.update({
            'representation': 'coordinate_aligned_matrix',
            'curve_shape_strategy': 'not_applicable',
            'transpose_order': transpose_order,
            'inverse_transpose_order': _inverse_transpose_order(transpose_order),
            'intermediate_shape': intermediate_shape,
            'output_shape': output_shape,
            'interpretation': interpretation,
        })
        return metadata

    if domain == 'real':
        indices, traversal_meta = _get_traversal_indices(
            (ry, rx),
            method,
            curve_shape_strategy=curve_shape_strategy,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        metadata.update(_copy_traversal_metadata(None, traversal_meta))
        metadata.update({
            'representation': 'real_stack',
            'traversal_indices': indices,
            'output_shape': (len(indices), ky, kx),
            'resize_method': resize_method,
            'resized_shape': working_shape if traversal_meta['curve_shape_strategy'] == 'resize' else None,
            'interpretation': (
                "The real-space scan grid is traversed into a stack of "
                "diffraction patterns."
            ),
        })
    elif domain == 'reciprocal':
        indices, traversal_meta = _get_traversal_indices(
            (ky, kx),
            method,
            curve_shape_strategy=curve_shape_strategy,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        metadata.update(_copy_traversal_metadata(None, traversal_meta))
        metadata.update({
            'representation': 'reciprocal_stack',
            'traversal_indices': indices,
            'output_shape': (len(indices), ry, rx),
            'resize_method': resize_method,
            'resized_shape': working_shape if traversal_meta['curve_shape_strategy'] == 'resize' else None,
            'interpretation': (
                "The reciprocal-space diffraction grid is traversed into a "
                "stack of real-space images."
            ),
        })
    else:
        real_indices, real_meta = _get_traversal_indices(
            (ry, rx),
            method,
            curve_shape_strategy=curve_shape_strategy,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        reciprocal_indices, reciprocal_meta = _get_traversal_indices(
            (ky, kx),
            method,
            curve_shape_strategy=curve_shape_strategy,
            resize_side=resize_side,
            resize_side_mode=resize_side_mode,
        )
        metadata.update({
            'representation': 'both_matrix',
            'real_traversal_indices': real_indices,
            'reciprocal_traversal_indices': reciprocal_indices,
            'output_shape': (len(real_indices), len(reciprocal_indices)),
            'resize_method': resize_method,
            'resized_shape': working_shape if real_meta['curve_shape_strategy'] == 'resize' else None,
            'interpretation': (
                "Rows traverse real-space positions and columns traverse "
                "reciprocal-space pixels."
            ),
        })
        metadata.update(_copy_traversal_metadata('real', real_meta))
        metadata.update(_copy_traversal_metadata('reciprocal', reciprocal_meta))
        metadata['curve_shape_strategy'] = real_meta['curve_shape_strategy']

    return metadata


def _require_metadata(metadata):
    """Validate unfolding metadata and return a shallow copy."""
    if metadata is None:
        raise ValueError(
            "metadata is required for undo=True. Use return_metadata=True when "
            "unfolding, or call HyperData.unfold(undo=True) on an unfolded "
            "HyperData object that still has attached metadata."
        )
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be a dictionary returned by unfold(...).")

    required = {'original_shape', 'domain', 'method', 'representation', 'output_shape'}
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"metadata is missing required field(s): {', '.join(missing)}.")

    clean = dict(metadata)
    clean['original_shape'] = _validate_unfold_shape(clean['original_shape'])
    clean['working_shape'] = _validate_unfold_shape(
        clean.get('working_shape', clean['original_shape'])
    )
    clean['output_shape'] = tuple(int(v) for v in clean['output_shape'])
    return clean


def _validate_unfolded_shape(array, metadata):
    """Raise a helpful error if an unfolded array does not match metadata."""
    expected_shape = tuple(metadata['output_shape'])
    if array.shape != expected_shape:
        raise ValueError(
            "Unfolded array shape does not match metadata: "
            f"got {array.shape}, expected {expected_shape}."
        )


def _extract_excess_values(array, metadata):
    """Store excluded data required for exact center-crop undo."""
    representation = metadata['representation']
    if metadata.get('curve_shape_strategy') != 'center_crop':
        return
    if not metadata.get('preserve_excess', False):
        return

    if representation == 'real_stack':
        excess_indices = np.asarray(metadata['excess_indices'], dtype=int)
        metadata['excess_values'] = array[
            excess_indices[:, 0],
            excess_indices[:, 1],
            :,
            :,
        ].copy()
    elif representation == 'reciprocal_stack':
        excess_indices = np.asarray(metadata['excess_indices'], dtype=int)
        values = np.empty(
            (len(excess_indices), array.shape[0], array.shape[1]),
            dtype=array.dtype,
        )
        for idx, (ky_idx, kx_idx) in enumerate(excess_indices):
            values[idx] = array[:, :, ky_idx, kx_idx]
        metadata['excess_values'] = values
    elif representation == 'both_matrix':
        real_excess = np.asarray(metadata['real_excess_indices'], dtype=int)
        reciprocal_excess = np.asarray(
            metadata['reciprocal_excess_indices'], dtype=int,
        )
        real_kept = np.asarray(metadata['real_traversal_indices'], dtype=int)
        metadata['excess_values'] = {
            'real': array[
                real_excess[:, 0], real_excess[:, 1], :, :
            ].copy(),
            'reciprocal': array[
                real_kept[:, 0, None],
                real_kept[:, 1, None],
                reciprocal_excess[None, :, 0],
                reciprocal_excess[None, :, 1],
            ].copy(),
        }
        metadata['excess_values_encoding'] = 'separated_domains'


def _unfold_payload_arrays(metadata):
    """Yield large saved values; traversal coordinates remain independent."""
    for key in ('original_values', 'excess_values'):
        value = metadata.get(key)
        if isinstance(value, np.ndarray):
            yield value
        elif isinstance(value, dict):
            yield from (
                item for item in value.values()
                if isinstance(item, np.ndarray)
            )


def _clone_unfold_metadata(metadata, *, share_payload=False):
    """Copy metadata without recopying immutable undo payloads by default."""
    if metadata is None:
        return None
    if not share_payload:
        return deepcopy(metadata)
    memo = {}
    for values in _unfold_payload_arrays(metadata):
        values.setflags(write=False)
        memo[id(values)] = values
    return deepcopy(metadata, memo)


def _crop_restore_shape(metadata):
    """Return the shape restored when center-crop excess was not preserved."""
    representation = metadata['representation']
    working_shape = metadata['working_shape']
    ry, rx, ky, kx = working_shape

    if representation == 'real_stack':
        side_y, side_x = metadata['curve_grid_shape']
        return (side_y, side_x, ky, kx)
    if representation == 'reciprocal_stack':
        side_y, side_x = metadata['curve_grid_shape']
        return (ry, rx, side_y, side_x)
    if representation == 'both_matrix':
        real_side_y, real_side_x = metadata['real_curve_grid_shape']
        reciprocal_side_y, reciprocal_side_x = metadata['reciprocal_curve_grid_shape']
        return (real_side_y, real_side_x, reciprocal_side_y, reciprocal_side_x)
    raise ValueError(f"Unsupported crop restore representation '{representation}'.")


def _local_crop_indices(indices, crop_slices):
    """Convert original-array crop coordinates to local cropped coordinates."""
    indices = np.asarray(indices, dtype=int)
    y0 = crop_slices['y'][0]
    x0 = crop_slices['x'][0]
    return indices - np.array([y0, x0])


def _restore_real_stack(array, metadata):
    """Undo a real-domain stack unfolding."""
    indices = np.asarray(metadata['traversal_indices'], dtype=int)
    strategy = metadata.get('curve_shape_strategy', 'full_shape')

    if strategy == 'center_crop' and not metadata.get('preserve_excess', False):
        restored = np.empty(_crop_restore_shape(metadata), dtype=array.dtype)
        local_indices = _local_crop_indices(indices, metadata['crop_slices'])
        restored[local_indices[:, 0], local_indices[:, 1], :, :] = array
        return restored

    if strategy == 'resize' and metadata.get('preserve_original', False):
        return np.array(metadata['original_values'], copy=True)

    restored_shape = metadata['working_shape']
    restored = np.empty(restored_shape, dtype=array.dtype)

    if strategy == 'center_crop' and metadata.get('preserve_excess', False):
        excess_indices = np.asarray(metadata['excess_indices'], dtype=int)
        restored[excess_indices[:, 0], excess_indices[:, 1], :, :] = metadata['excess_values']

    restored[indices[:, 0], indices[:, 1], :, :] = array
    return restored


def _restore_reciprocal_stack(array, metadata):
    """Undo a reciprocal-domain stack unfolding."""
    indices = np.asarray(metadata['traversal_indices'], dtype=int)
    strategy = metadata.get('curve_shape_strategy', 'full_shape')

    if strategy == 'center_crop' and not metadata.get('preserve_excess', False):
        restored = np.empty(_crop_restore_shape(metadata), dtype=array.dtype)
        local_indices = _local_crop_indices(indices, metadata['crop_slices'])
        for flat_idx, (ky_idx, kx_idx) in enumerate(local_indices):
            restored[:, :, ky_idx, kx_idx] = array[flat_idx]
        return restored

    if strategy == 'resize' and metadata.get('preserve_original', False):
        return np.array(metadata['original_values'], copy=True)

    restored_shape = metadata['working_shape']
    restored = np.empty(restored_shape, dtype=array.dtype)

    if strategy == 'center_crop' and metadata.get('preserve_excess', False):
        excess_indices = np.asarray(metadata['excess_indices'], dtype=int)
        for flat_idx, (ky_idx, kx_idx) in enumerate(excess_indices):
            restored[:, :, ky_idx, kx_idx] = metadata['excess_values'][flat_idx]

    for flat_idx, (ky_idx, kx_idx) in enumerate(indices):
        restored[:, :, ky_idx, kx_idx] = array[flat_idx]
    return restored


def _restore_both_matrix(array, metadata):
    """Undo a 2D mixed real/reciprocal unfolding."""
    real_indices = np.asarray(metadata['real_traversal_indices'], dtype=int)
    reciprocal_indices = np.asarray(metadata['reciprocal_traversal_indices'], dtype=int)
    strategy = metadata.get('curve_shape_strategy', 'full_shape')

    if strategy == 'center_crop' and not metadata.get('preserve_excess', False):
        restored = np.empty(_crop_restore_shape(metadata), dtype=array.dtype)
        real_local = _local_crop_indices(real_indices, metadata['real_crop_slices'])
        reciprocal_local = _local_crop_indices(
            reciprocal_indices,
            metadata['reciprocal_crop_slices'],
        )
    else:
        if strategy == 'resize' and metadata.get('preserve_original', False):
            return np.array(metadata['original_values'], copy=True)
        if strategy == 'center_crop' and metadata.get('preserve_excess', False):
            encoding = metadata.get('excess_values_encoding', 'full_tensor')
            if encoding == 'full_tensor':
                # Saved files from the original implementation use this form.
                restored = np.array(metadata['excess_values'], copy=True)
            elif encoding == 'separated_domains':
                restored = np.empty(metadata['working_shape'], dtype=array.dtype)
                real_excess = np.asarray(metadata['real_excess_indices'], dtype=int)
                reciprocal_excess = np.asarray(
                    metadata['reciprocal_excess_indices'], dtype=int,
                )
                values = metadata['excess_values']
                restored[real_excess[:, 0], real_excess[:, 1], :, :] = values['real']
                for flat_idx, (ry_idx, rx_idx) in enumerate(real_indices):
                    restored[
                        ry_idx, rx_idx,
                        reciprocal_excess[:, 0], reciprocal_excess[:, 1],
                    ] = values['reciprocal'][flat_idx]
            else:
                raise ValueError(
                    f"Unsupported excess-values encoding '{encoding}'."
                )
        else:
            restored = np.empty(metadata['working_shape'], dtype=array.dtype)
        real_local = real_indices
        reciprocal_local = reciprocal_indices

    for real_flat_idx, (ry_idx, rx_idx) in enumerate(real_local):
        restored[
            ry_idx,
            rx_idx,
            reciprocal_local[:, 0],
            reciprocal_local[:, 1],
        ] = array[real_flat_idx]
    return restored


def _infer_square_side(value, label):
    """Infer a square side length from a flattened grid size."""
    value = int(value)
    side = int(np.sqrt(value))
    if side * side != value:
        raise ValueError(
            f"Cannot infer a square {label} grid from {value} elements. "
            "Pass original_shape explicitly."
        )
    return side


def _infer_row_major_original_shape(array, domain, original_shape=None):
    """Infer or validate the original 4D shape for row-major undo without metadata."""
    domain, _ = _normalize_unfold_request(domain=domain, method='row_major')

    if original_shape is not None:
        original_shape = tuple(int(v) for v in original_shape)
        if len(original_shape) == 4:
            return _validate_unfold_shape(original_shape)
        if len(original_shape) == 2 and domain == 'real' and array.ndim == 3:
            ry, rx = original_shape
            return _validate_unfold_shape((ry, rx, array.shape[1], array.shape[2]))
        if len(original_shape) == 2 and domain == 'reciprocal' and array.ndim == 3:
            ky, kx = original_shape
            return _validate_unfold_shape((array.shape[1], array.shape[2], ky, kx))
        raise ValueError(
            "original_shape must be (Ry, Rx, Ky, Kx). For 3D row-major "
            "domain='real' undo, (Ry, Rx) is also accepted; for "
            "domain='reciprocal' undo, (Ky, Kx) is also accepted."
        )

    if domain == 'real' and array.ndim == 3:
        ry = rx = _infer_square_side(array.shape[0], 'real-space')
        return (ry, rx, array.shape[1], array.shape[2])

    if domain == 'reciprocal' and array.ndim == 3:
        ky = kx = _infer_square_side(array.shape[0], 'reciprocal-space')
        return (array.shape[1], array.shape[2], ky, kx)

    if domain == 'both' and array.ndim == 2:
        ry = rx = _infer_square_side(array.shape[0], 'real-space')
        ky = kx = _infer_square_side(array.shape[1], 'reciprocal-space')
        return (ry, rx, ky, kx)

    raise ValueError(
        "metadata-less undo requires original_shape unless a square row-major "
        "shape can be inferred from the unfolded array."
    )


def _build_row_major_undo_metadata(array, domain='real', method='row_major',
                                   original_shape=None):
    """Build minimal row-major metadata for undo when no unfold metadata exists."""
    domain, method = _normalize_unfold_request(domain=domain, method=method)
    if method != 'row_major':
        raise ValueError(
            "metadata-less undo only supports method='row_major'. Pass metadata "
            "for non-row-major traversal methods."
        )

    original_shape = _infer_row_major_original_shape(
        array,
        domain=domain,
        original_shape=original_shape,
    )
    return _build_unfold_metadata(
        original_shape=original_shape,
        working_shape=original_shape,
        domain=domain,
        method='row_major',
        preserve_excess=False,
    )


def _unfold_array(array, domain='real', method='row_major',
                  curve_shape_strategy='center_crop', preserve_excess=True,
                  resize_side=None, resize_side_mode='nearest',
                  resize_method='linear', preserve_original=False,
                  original_shape=None, original_values=None, undo=False,
                  metadata=None, return_metadata=False):
    """
    Unfold or restore a 4D-STEM tensor using explicit domain/method metadata.

    Examples
    --------
    >>> original = np.arange(5*7*4*6).reshape(5, 7, 4, 6)
    >>> unfolded, meta = _unfold_array(original, domain='real',
    ...                                method='hilbert',
    ...                                preserve_excess=True,
    ...                                return_metadata=True)
    >>> restored = _unfold_array(unfolded, undo=True, metadata=meta)
    >>> np.array_equal(restored, original)
    True
    """
    array = np.asarray(array)

    if undo:
        if metadata is None:
            metadata = _build_row_major_undo_metadata(
                array,
                domain=domain,
                method=method,
                original_shape=original_shape,
            )
        metadata = _require_metadata(metadata)
        _validate_unfolded_shape(array, metadata)
        representation = metadata['representation']

        if representation == 'coordinate_aligned_matrix':
            restored = array.reshape(metadata['intermediate_shape'])
            restored = np.transpose(restored, metadata['inverse_transpose_order'])

        elif representation == 'real_stack':
            restored = _restore_real_stack(array, metadata)

        elif representation == 'reciprocal_stack':
            restored = _restore_reciprocal_stack(array, metadata)

        elif representation == 'both_matrix':
            restored = _restore_both_matrix(array, metadata)

        else:
            raise ValueError(f"Unsupported metadata representation '{representation}'.")

        if return_metadata:
            return restored, metadata
        return restored

    if array.ndim != 4:
        raise ValueError(
            "Forward unfolding expects a 4D-STEM tensor with shape "
            "(Ry, Rx, Ky, Kx)."
        )

    metadata = _build_unfold_metadata(
        original_shape=original_shape or array.shape,
        working_shape=array.shape,
        domain=domain,
        method=method,
        curve_shape_strategy=curve_shape_strategy,
        preserve_excess=preserve_excess,
        resize_side=resize_side,
        resize_side_mode=resize_side_mode,
        resize_method=resize_method,
        preserve_original=preserve_original,
    )

    if preserve_original and original_values is not None:
        metadata['original_values'] = np.array(original_values, copy=True)

    _extract_excess_values(array, metadata)
    payloads = tuple(_unfold_payload_arrays(metadata))
    metadata['preserved_values_nbytes'] = sum(values.nbytes for values in payloads)
    for values in payloads:
        values.setflags(write=False)
    representation = metadata['representation']

    if representation == 'coordinate_aligned_matrix':
        unfolded = np.transpose(array, metadata['transpose_order'])
        unfolded = unfolded.reshape(metadata['output_shape'])

    elif representation == 'real_stack':
        indices = metadata['traversal_indices']
        unfolded = array[indices[:, 0], indices[:, 1], :, :]

    elif representation == 'reciprocal_stack':
        indices = metadata['traversal_indices']
        unfolded = np.empty(metadata['output_shape'], dtype=array.dtype)
        for flat_idx, (ky_idx, kx_idx) in enumerate(indices):
            unfolded[flat_idx] = array[:, :, ky_idx, kx_idx]

    elif representation == 'both_matrix':
        real_indices = metadata['real_traversal_indices']
        reciprocal_indices = metadata['reciprocal_traversal_indices']
        unfolded = np.empty(metadata['output_shape'], dtype=array.dtype)
        for flat_idx, (ry_idx, rx_idx) in enumerate(real_indices):
            unfolded[flat_idx] = array[
                ry_idx,
                rx_idx,
                reciprocal_indices[:, 0],
                reciprocal_indices[:, 1],
            ]

    else:
        raise ValueError(f"Unsupported unfolding representation '{representation}'.")

    metadata['output_shape'] = unfolded.shape

    if return_metadata:
        return unfolded, metadata
    return unfolded
