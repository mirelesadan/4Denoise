"""Traversal-path visualization using the shared traversal registry."""

from numbers import Integral
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.patches import Rectangle
import numpy as np
from fourdenoise_traversals import (
    _BLOCK_TRAVERSAL_METHODS,
    _TRAVERSAL_INDEX_GENERATORS,
    _block_size_for_method,
    _get_traversal_indices,
    _moore_indices,
    _normalize_traversal_method,
)


def plot_traversals(method='all', grid_shape=(64, 64), *, ncols=3,
                    cmap='turbo', linewidth=1.5, show_grid=True,
                    mark_endpoints=True, show=True):
    """Plot one or all implemented 2D unfolding traversal paths.

    ``grid_shape`` is the maximum requested grid, not a promise that every
    method uses every cell. Full-shape methods traverse it entirely. Curve and
    block methods traverse their largest compatible centered crop within it;
    shaded cells are excluded. Each title shows both the used and maximum grid
    shapes. This visualizer does not resize data or require a ``HyperData``.

    Parameters
    ----------
    method : str, optional
        An implemented traversal name or ``'all'``. The latter plots the ten
        implemented canonical methods in registry order. ``'z_order'`` and
        ``'meander_4'``/``'meander_5'`` are accepted aliases for single plots.
        ``'moore'`` is not yet implemented; ``'coordinate_aligned'`` is an axis
        rearrangement rather than a 2D traversal.
    grid_shape : int or tuple[int, int], optional
        Maximum grid height and width, in ``(y, x)`` order. An integer requests
        a square. Compatible curves may use only a centered part of this grid.
    ncols : int, optional
        Maximum number of subplot columns. A single method uses one column.
    cmap : str or Colormap, optional
        Matplotlib colormap for the path's normalized start-to-end progress.
    linewidth : float, optional
        Width of the colored traversal line in points.
    show_grid, mark_endpoints : bool, optional
        Draw cell boundaries and colored start/end markers, respectively.
    show : bool, optional
        Display the figure with ``matplotlib.pyplot.show`` when True.

    Returns
    -------
    tuple[Figure, ndarray]
        Matplotlib figure and a 2D array of axes. Unused axes in ``'all'``
        mode are hidden. A block method that cannot fit inside ``grid_shape``
        gets an explanatory panel in ``'all'`` mode; a single-method request
        raises the underlying size error.

    Examples
    --------
    >>> fig, axes = plot_traversals('all', (64, 64), ncols=3)
    >>> fig, axes = plot_traversals('peano', 81, linewidth=2, show=False)
    """
    if isinstance(grid_shape, Integral) and not isinstance(grid_shape, (bool, np.bool_)):
        grid_shape = (grid_shape, grid_shape)
    else:
        grid_shape = np.asarray(grid_shape, dtype=object)
        if grid_shape.shape != (2,):
            raise ValueError("grid_shape must be a positive integer or (height, width).")
    if any(
        isinstance(size, (bool, np.bool_))
        or not isinstance(size, (Integral, np.integer)) or size < 1
        for size in grid_shape
    ):
        raise ValueError("grid_shape must contain positive integers.")
    height, width = map(int, grid_shape)

    if isinstance(ncols, (bool, np.bool_)) or not isinstance(ncols, Integral) or ncols < 1:
        raise ValueError("ncols must be a positive integer.")
    if isinstance(linewidth, (bool, np.bool_)) or not isinstance(
        linewidth, (Integral, float, np.floating)
    ) or not np.isfinite(linewidth) or linewidth <= 0:
        raise ValueError("linewidth must be a positive finite number.")

    requested_method = _normalize_traversal_method(method)
    implemented = tuple(
        name for name in _TRAVERSAL_INDEX_GENERATORS if name != 'moore'
    )
    if requested_method == 'all':
        methods = implemented
    elif requested_method == 'moore':
        _moore_indices((height, width))
    elif requested_method in implemented:
        if (
            requested_method in _BLOCK_TRAVERSAL_METHODS
            and min(height, width) < _block_size_for_method(requested_method)
        ):
            _get_traversal_indices((height, width), requested_method)
        methods = (requested_method,)
    else:
        valid = ', '.join(implemented)
        raise ValueError(
            f"method must be 'all' or one of: {valid}. "
            "coordinate_aligned is not a 2D traversal."
        )

    plot_cols = min(int(ncols), len(methods))
    plot_rows = int(np.ceil(len(methods) / plot_cols))
    fig, axes = plt.subplots(
        plot_rows, plot_cols, squeeze=False,
        figsize=(4.8 * plot_cols, 4.8 * plot_rows), layout='constrained',
    )
    progress_norm = Normalize(vmin=0, vmax=1)

    for ax, name in zip(axes.flat, methods):
        ax.set_xlim(-0.5, width - 0.5)
        ax.set_ylim(height - 0.5, -0.5)
        ax.set_aspect('equal')
        ax.set_facecolor('white')
        if show_grid:
            grid_lines = [
                ((x - 0.5, -0.5), (x - 0.5, height - 0.5))
                for x in range(width + 1)
            ] + [
                ((-0.5, y - 0.5), (width - 0.5, y - 0.5))
                for y in range(height + 1)
            ]
            ax.add_collection(LineCollection(
                grid_lines, colors='0.75', linewidths=0.3, alpha=0.45, zorder=0.5,
            ))
        ax.add_patch(Rectangle(
            (-0.5, -0.5), width, height, fill=False,
            edgecolor='0.45', linewidth=0.8, zorder=1,
        ))

        if name in _BLOCK_TRAVERSAL_METHODS and min(height, width) < _block_size_for_method(name):
            block_size = _block_size_for_method(name)
            ax.set_title(f'{name}\nunavailable / max {height}x{width}')
            ax.text(
                (width - 1) / 2, (height - 1) / 2,
                f'Requires at least {block_size}x{block_size}',
                ha='center', va='center', fontsize=9,
            )
            ax.set_axis_off()
            continue

        indices, details = _get_traversal_indices(
            (height, width), name, curve_shape_strategy='center_crop',
        )
        used_height, used_width = details.get('curve_grid_shape', (height, width))
        crop_slices = details.get('crop_slices')
        if crop_slices is not None:
            y0, y1 = crop_slices['y']
            x0, x1 = crop_slices['x']
            excluded_regions = (
                (-0.5, -0.5, width, y0),
                (-0.5, y1 - 0.5, width, height - y1),
                (-0.5, y0 - 0.5, x0, y1 - y0),
                (x1 - 0.5, y0 - 0.5, width - x1, y1 - y0),
            )
            for x, y, region_width, region_height in excluded_regions:
                if region_width > 0 and region_height > 0:
                    ax.add_patch(Rectangle(
                        (x, y), region_width, region_height,
                        facecolor='0.88', edgecolor='none', alpha=0.7, zorder=0.2,
                    ))
            ax.add_patch(Rectangle(
                (x0 - 0.5, y0 - 0.5), used_width, used_height,
                fill=False, edgecolor='0.35', linestyle='--',
                linewidth=0.9, zorder=3,
            ))

        xy = indices[:, [1, 0]].astype(np.float32)
        if len(xy) > 1:
            segments = np.stack((xy[:-1], xy[1:]), axis=1)
            path = LineCollection(
                segments, cmap=cmap, norm=progress_norm,
                linewidths=linewidth, zorder=2,
            )
            path.set_array(np.linspace(0, 1, len(segments), dtype=np.float32))
            ax.add_collection(path)
        if mark_endpoints:
            ax.scatter(
                xy[0, 0], xy[0, 1], s=28, marker='o', c='limegreen',
                edgecolors='black', linewidths=0.5, zorder=4,
            )
            ax.scatter(
                xy[-1, 0], xy[-1, 1], s=34, marker='X', c='crimson',
                edgecolors='black', linewidths=0.5, zorder=4,
            )
        ax.set_title(f'{name}\nused {used_height}x{used_width} / max {height}x{width}')
        ax.set_axis_off()

    for ax in list(axes.flat)[len(methods):]:
        ax.set_visible(False)
    colorbar = fig.colorbar(
        ScalarMappable(norm=progress_norm, cmap=cmap),
        ax=list(axes.flat)[:len(methods)], shrink=0.7, pad=0.02,
    )
    colorbar.set_label('Traversal progress (start to end)')
    if show:
        plt.show()
    return fig, axes
