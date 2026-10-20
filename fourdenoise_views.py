"""Private rendering for RealSpace and ReciprocalSpace public views.

Public methods keep their signatures and documentation in fourdenoise.
These renderers depend on view attributes, not the main module or denoising.
"""

import numpy as np
from numbers import Integral
from scipy.ndimage import map_coordinates
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
from matplotlib.colors import ListedColormap, to_rgba
from mpl_toolkits.axes_grid1 import make_axes_locatable
from fourdenoise_geometry import _normalize_unit_mode, _real_spacing_pair, _resolve_unit_mode

def _show_reciprocal_space(self,
         power: float = 1,
         title: str = 'Diffraction Pattern',
         logScale: bool = True,
         axes: bool = True,
         vmin=None,
         vmax=None,
         figsize=(10, 10),
         aspect=None,
         cmap: str = 'turbo',
         coords: np.ndarray | None = None,
         axis_units='auto',
         y=None,
         x=None,
         overlay_mask=None,
         overlay_color='green',
         overlay_alpha=0.3,
         mode='cartesian',
         grid=False,
         grid_ticks=None,
         radial_ticks=None,
         angular_ticks=None,
         grid_color='white',
         grid_alpha=0.35,
         scale_bar=None,
         scale_bar_position=None,
         scale_bar_color='white',
         scale_bar_label=True,
         **scatter_kwargs):
    """Render a ReciprocalSpace view without owning its data."""
    if self.array.ndim != 2:
        raise ValueError("ReciprocalSpace.show requires a 2D array.")
    if 'units' in scatter_kwargs or 'conv_factor' in scatter_kwargs:
        raise TypeError(
            "Use axis_units='auto', 'pixels', or 'calibrated' instead of "
            "passing units/conv_factor to show()."
        )
    if mode not in ('cartesian', 'polar'):
        raise ValueError("mode must be 'cartesian' or 'polar'.")
    if self.is_cepstral and title == 'Diffraction Pattern':
        title = 'EWPC'
    if mode == 'polar' and grid_ticks is not None:
        raise ValueError("grid_ticks applies only to rectangular mode.")
    if mode == 'cartesian' and (radial_ticks is not None or angular_ticks is not None):
        raise ValueError("radial_ticks and angular_ticks require mode='polar'.")
    if grid_ticks is not None:
        if isinstance(grid_ticks, (Integral, np.integer)) and not isinstance(
            grid_ticks, (bool, np.bool_)
        ):
            grid_ticks = (grid_ticks, grid_ticks)
        if not isinstance(grid_ticks, (tuple, list, np.ndarray)) or len(grid_ticks) != 2:
            raise ValueError("grid_ticks must be an integer or a (y, x) pair.")
        grid_ticks = tuple(
            self._plot_tick_count(count, 'grid_ticks', minimum=2)
            for count in grid_ticks
        )
    if radial_ticks is not None:
        radial_ticks = self._plot_tick_count(radial_ticks, 'radial_ticks')
    if angular_ticks is not None:
        angular_ticks = self._plot_tick_count(angular_ticks, 'angular_ticks')
    try:
        grid_alpha = float(grid_alpha)
    except (TypeError, ValueError) as exc:
        raise ValueError("grid_alpha must be between 0 and 1.") from exc
    if not np.isfinite(grid_alpha) or not 0 <= grid_alpha <= 1:
        raise ValueError("grid_alpha must be between 0 and 1.")
    to_rgba(grid_color)
    to_rgba(scale_bar_color)
    if scale_bar_position is None:
        scale_bar_position = 'center' if mode == 'polar' else 'left'
    if scale_bar_position not in ('left', 'center', 'right'):
        raise ValueError("scale_bar_position must be 'left', 'center', or 'right'.")
    if scale_bar is not None:
        if isinstance(scale_bar, (bool, np.bool_)):
            raise ValueError("scale_bar must be a positive length.")
        try:
            scale_bar = float(scale_bar)
        except (TypeError, ValueError) as exc:
            raise ValueError("scale_bar must be a positive length.") from exc
        if not np.isfinite(scale_bar) or scale_bar <= 0:
            raise ValueError("scale_bar must be a positive length.")
    if coords is not None and (y is not None or x is not None):
        raise ValueError("Provide either coords or y and x, not both.")
    if (y is None) != (x is None):
        raise ValueError("Supply both y and x for scatter points.")
    if y is not None:
        y_values = np.atleast_1d(np.asarray(y, dtype=float))
        x_values = np.atleast_1d(np.asarray(x, dtype=float))
        if y_values.ndim != 1 or x_values.ndim != 1:
            raise ValueError("y and x must be scalars or 1D arrays.")
        if y_values.shape != x_values.shape:
            raise ValueError("y and x must contain the same number of points.")
        coords = np.column_stack((y_values, x_values))
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape == (2,):
            coords = coords.reshape(1, 2)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (N, 2).")
        if not np.all(np.isfinite(coords)):
            raise ValueError("Scatter coordinates must be finite.")
    if overlay_mask is not None:
        overlay_mask = np.asarray(overlay_mask, dtype=bool)
        if overlay_mask.shape != self.shape:
            raise ValueError(
                f"overlay_mask must have shape {self.shape}; got "
                f"{overlay_mask.shape}."
            )
        if not np.isfinite(overlay_alpha) or not 0 <= overlay_alpha <= 1:
            raise ValueError("overlay_alpha must be between 0 and 1.")

    if self.is_polar:
        (
            extent,
            theta_unit_text,
            radius_unit_text,
            radius_axis_start,
            radius_sample_step,
        ) = self._polar_axis_info(axis_units=axis_units)
        conv_factor = None
    else:
        units, conv_factor, _ = _resolve_unit_mode(
            axis_units, self.units, self.conv_factor, label='axis_units',
        )
        axis_unit_text = self._format_unit_text(units)
        extent = self._axis_extent(conv_factor=conv_factor)

    image_data = self.array
    plot_overlay = overlay_mask
    if mode == 'polar':
        if self.is_polar:
            theta_edges = np.deg2rad(
                np.linspace(extent[0], extent[1], self.shape[1] + 1)
            )
            radius_edges = np.linspace(extent[3], extent[2], self.shape[0] + 1)
            radius_max = float(extent[2])
        else:
            ky, kx = self.shape
            center_y, center_x = self._display_center()
            radius_max_pixels = min(center_y, center_x, ky - 1 - center_y, kx - 1 - center_x)
            if radius_max_pixels <= 0:
                raise ValueError("A circular view requires at least a 2x2 image.")
            n_radius = max(2, int(np.ceil(radius_max_pixels)) + 1)
            n_angle = max(24, int(np.ceil(2 * np.pi * radius_max_pixels)))
            radius_samples = np.linspace(0, radius_max_pixels, n_radius)
            angle_samples = np.linspace(0, 2 * np.pi, n_angle, endpoint=False)
            sample_y = center_y + radius_samples[:, None] * np.sin(angle_samples)
            sample_x = center_x + radius_samples[:, None] * np.cos(angle_samples)
            sampling_coords = np.vstack((sample_y.ravel(), sample_x.ravel()))
            image_data = map_coordinates(
                self.array, sampling_coords, order=1, mode='nearest',
            ).reshape(n_radius, n_angle)
            if overlay_mask is not None:
                plot_overlay = map_coordinates(
                    overlay_mask.astype(np.uint8), sampling_coords,
                    order=0, mode='nearest',
                ).reshape(n_radius, n_angle).astype(bool)
            plot_scale = 1.0 if conv_factor is None else conv_factor
            radius_max = radius_max_pixels * plot_scale
            radius_edges = np.linspace(0, radius_max, n_radius + 1)
            theta_edges = np.linspace(0, 2 * np.pi, n_angle + 1)
            radius_unit_text = axis_unit_text
        if not np.isfinite(radius_max) or radius_max <= 0:
            raise ValueError("Polar display requires a positive radial range.")

    with np.errstate(divide='ignore', invalid='ignore'):
        processed_data = power * np.log(image_data) if logScale else image_data ** power
    finite_values = np.asarray(processed_data)[np.isfinite(processed_data)]
    if finite_values.size == 0:
        raise ValueError("Diffraction pattern has no finite values to display.")
    if vmin is None:
        vmin = np.min(finite_values)
    if vmax is None:
        vmax = np.max(finite_values)
    processed_data = np.ma.masked_invalid(processed_data)

    if mode == 'polar':
        fig, ax = plt.subplots(figsize=figsize, subplot_kw={'projection': 'polar'})
        ax.grid(False)
        ax.set_theta_zero_location('E')
        ax.set_theta_direction(-1)
        im1 = ax.pcolormesh(
            theta_edges, radius_edges, processed_data,
            cmap=cmap, vmin=vmin, vmax=vmax, shading='flat', zorder=1,
        )
        ax.set_rlim(0, radius_max)
        if plot_overlay is not None:
            mask_cmap = ListedColormap([
                (0, 0, 0, 0), to_rgba(overlay_color, overlay_alpha),
            ])
            ax.pcolormesh(
                theta_edges, radius_edges, plot_overlay.astype(np.uint8),
                cmap=mask_cmap, vmin=0, vmax=1, shading='flat', zorder=2,
            )
    else:
        fig, ax = plt.subplots(figsize=figsize)
        im1 = ax.imshow(
            processed_data, vmin=vmin, vmax=vmax, cmap=cmap,
            extent=extent, origin='upper', zorder=1,
        )
        if plot_overlay is not None:
            rgba = np.empty((*self.shape, 4), dtype=np.float32)
            rgba[:] = to_rgba(overlay_color)
            rgba[..., 3] = plot_overlay * overlay_alpha
            ax.imshow(
                rgba, extent=extent, origin='upper',
                interpolation='nearest', zorder=2,
            )
        if aspect is not None:
            ax.set_aspect(aspect)

    if coords is not None and coords.size > 0:
        if self.is_polar:
            metadata = self.polar_metadata or {}
            theta_step = metadata.get('theta_step', 360.0 / self.shape[1])
            theta_min = metadata.get('theta_sample_start', metadata.get('theta_range', (0.0, 360.0))[0])
            angle_coords = theta_min + coords[:, 1] * theta_step
            radius_coords = radius_axis_start + coords[:, 0] * radius_sample_step
            x_coords = np.deg2rad(angle_coords) if mode == 'polar' else angle_coords
            y_coords = radius_coords
        else:
            plot_scale = 1.0 if conv_factor is None else conv_factor
            center_y, center_x = self._display_center()
            delta_y = coords[:, 0] - center_y
            delta_x = coords[:, 1] - center_x
            if mode == 'polar':
                x_coords = np.mod(np.arctan2(delta_y, delta_x), 2 * np.pi)
                y_coords = np.hypot(delta_y, delta_x) * plot_scale
            else:
                x_coords = delta_x * plot_scale
                y_coords = -delta_y * plot_scale
        scatter_kwargs.setdefault('zorder', 3)
        ax.scatter(x_coords, y_coords, **scatter_kwargs)

    if mode == 'polar':
        n_radial = radial_ticks if radial_ticks is not None else (5 if grid else None)
        n_angular = angular_ticks if angular_ticks is not None else (12 if grid else None)
        if n_radial is not None:
            radius_locations = np.linspace(0, radius_max, n_radial + 1)[1:]
            ax.set_yticks(radius_locations)
            ax.set_yticklabels([f"{value:g}" for value in radius_locations])
        if n_angular is not None:
            angle_locations = np.linspace(0, 360, n_angular, endpoint=False)
            ax.set_xticks(np.deg2rad(angle_locations))
            ax.set_xticklabels([f"{value:g}°" for value in angle_locations])
        if grid and axes:
            ax.grid(color=grid_color, alpha=grid_alpha, linewidth=0.8)
        elif grid:
            angles_for_lines = np.linspace(0, 2 * np.pi, 361)
            for radius in radius_locations:
                ax.plot(
                    angles_for_lines,
                    np.full_like(angles_for_lines, radius),
                    color=grid_color, alpha=grid_alpha, linewidth=0.8, zorder=2.5,
                )
            for angle in np.deg2rad(angle_locations):
                ax.plot(
                    [angle, angle], [0, radius_max],
                    color=grid_color, alpha=grid_alpha, linewidth=0.8, zorder=2.5,
                )
        if axes:
            radial_symbol = 'c_r' if self.is_cepstral else 'k_r'
            ax.set_rlabel_position(145)
            for tick_label in ax.get_yticklabels():
                tick_label.set_color('white')
                tick_label.set_path_effects([
                    path_effects.Stroke(linewidth=2, foreground='black'),
                    path_effects.Normal(),
                ])
            ax.set_title(
                f"{title}  (${radial_symbol}$: {radius_unit_text})",
                fontsize=16, pad=28,
            )
        else:
            ax.set_axis_off()
        bar_unit_text = radius_unit_text
    else:
        tick_counts = grid_ticks if grid_ticks is not None else ((6, 6) if grid else None)
        if tick_counts is not None:
            n_y, n_x = tick_counts
            x_locations = np.linspace(extent[0], extent[1], n_x)
            y_locations = np.linspace(extent[2], extent[3], n_y)
            ax.set_xticks(x_locations)
            ax.set_yticks(y_locations)
        if grid and axes:
            ax.grid(color=grid_color, alpha=grid_alpha, linewidth=0.8)
        elif grid:
            for location in x_locations:
                ax.axvline(location, color=grid_color, alpha=grid_alpha,
                           linewidth=0.8, zorder=2.5)
            for location in y_locations:
                ax.axhline(location, color=grid_color, alpha=grid_alpha,
                           linewidth=0.8, zorder=2.5)
        if axes:
            if self.is_polar:
                ax.set_xlabel(rf"$k_\theta$ ({theta_unit_text})", fontsize=14)
                ax.set_ylabel(rf"$k_r$ ({radius_unit_text})", fontsize=14)
            elif self.is_cepstral:
                ax.set_xlabel(rf"$c_x$ ({axis_unit_text})", fontsize=14)
                ax.set_ylabel(rf"$c_y$ ({axis_unit_text})", fontsize=14)
            else:
                ax.set_xlabel(rf"$k_x$ ({axis_unit_text})", fontsize=14)
                ax.set_ylabel(rf"$k_y$ ({axis_unit_text})", fontsize=14)
            ax.set_title(title, fontsize=18)
        else:
            ax.set_axis_off()
        bar_unit_text = theta_unit_text if self.is_polar else axis_unit_text

    if axes:
        if mode == 'polar':
            cb = fig.colorbar(im1, ax=ax, pad=0.1, shrink=0.8)
        else:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.05)
            cb = fig.colorbar(im1, cax=cax)
        cb.ax.tick_params(labelsize=15)
        if logScale:
            cbar_title = (
                "log(Intensity)" if power == 1
                else f"log(Intensity)\n(Power = {power})"
            )
        else:
            cbar_title = (
                "Intensity" if power == 1
                else f"Intensity\n(Power = {power})"
            )
        cb.set_label(cbar_title, fontsize=14)

    if scale_bar is not None:
        self._draw_scale_bar(
            ax, scale_bar, bar_unit_text, scale_bar_color,
            scale_bar_position, scale_bar_label, mode,
            extent=extent, radius_max=radius_max if mode == 'polar' else None,
        )
    if mode == 'polar':
        fig.tight_layout(pad=1.5)
    plt.show()


def _show_real_space(self,
         title: str = 'Real-Space Image',
         axes: bool = True,
         grid: bool = False,
         num_div=10,
         gridColor: str = 'black',
         vmin=None,
         vmax=None,
         figsize=(8, 8),
         aspect=None,
         cmap: str = 'gray',
         coords: np.ndarray | None = None,
         axis_units='auto',
         *, ax=None, show=True, colorbar=None, colorbar_label=None,
         grid_color=None, grid_ticks=None,
         scale_bar=None, scale_bar_position='left',
         scale_bar_color='white', scale_bar_label=True,
         coords_units='pixels', y=None, x=None,
         percentiles=None, symmetric=False,
         **scatter_kwargs):
    """Render a RealSpace view without owning its data."""
    if 'units' in scatter_kwargs or 'conv_factor' in scatter_kwargs:
        raise TypeError(
            "RealSpace.show uses axis_units='auto', 'pixels', or "
            "'calibrated'. Set the object scale with set_scale() instead "
            "of passing units/conv_factor to show()."
        )
    if coords is not None and (y is not None or x is not None):
        raise ValueError("Provide either coords or y and x, not both.")
    if (y is None) != (x is None):
        raise ValueError("Supply both y and x for scatter points.")
    if y is not None:
        y_values = np.atleast_1d(np.asarray(y, dtype=float))
        x_values = np.atleast_1d(np.asarray(x, dtype=float))
        if y_values.ndim != 1 or x_values.ndim != 1 or y_values.shape != x_values.shape:
            raise ValueError("y and x must be equally sized scalar or 1D arrays.")
        coords = np.column_stack((y_values, x_values))
    if coords is not None:
        coords = np.asarray(coords, dtype=float)
        if coords.shape == (2,):
            coords = coords.reshape(1, 2)
        if coords.size == 0:
            coords = np.empty((0, 2), dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError("coords must have shape (N, 2) in (y, x) order.")
        if not np.all(np.isfinite(coords)):
            raise ValueError("Scatter coordinates must be finite.")
    elif scatter_kwargs:
        raise TypeError("Scatter styling requires coords or both y and x.")

    input_mode = _normalize_unit_mode(coords_units, label='coords_units')
    if input_mode == 'auto':
        input_mode = 'pixels'
    if input_mode == 'calibrated' and self.conv_factor is None:
        raise ValueError("coords_units='calibrated' requires image calibration.")
    units, conv_factor, _ = _resolve_unit_mode(
        axis_units, self.units, self.conv_factor, label='axis_units',
    )
    axis_unit_text = self._format_unit_text(units)
    extent = self._axis_extent(conv_factor=conv_factor)
    display_sy, display_sx = _real_spacing_pair(conv_factor)
    display_origin = self.origin if conv_factor is not None else (0.0, 0.0)

    if grid_color is None:
        grid_color = gridColor
    to_rgba(grid_color)
    to_rgba(scale_bar_color)
    if scale_bar_position not in ('left', 'center', 'right'):
        raise ValueError("scale_bar_position must be 'left', 'center', or 'right'.")
    if scale_bar is not None:
        if isinstance(scale_bar, (bool, np.bool_)):
            raise ValueError("scale_bar must be positive and finite.")
        try:
            scale_bar = float(scale_bar)
        except (TypeError, ValueError) as exc:
            raise ValueError("scale_bar must be positive and finite.") from exc
        if not np.isfinite(scale_bar) or scale_bar <= 0:
            raise ValueError("scale_bar must be positive and finite.")
        if scale_bar > 0.88 * abs(extent[1] - extent[0]):
            raise ValueError("scale_bar is too long for the displayed image.")

    def tick_counts(value, label, minimum):
        if isinstance(value, (Integral, np.integer)) and not isinstance(value, (bool, np.bool_)):
            value = (value, value)
        if not isinstance(value, (tuple, list, np.ndarray)) or len(value) != 2:
            raise ValueError(f"{label} must be an integer or (y, x) pair.")
        if any(
            isinstance(count, (bool, np.bool_))
            or not isinstance(count, (Integral, np.integer)) or count < minimum
            for count in value
        ):
            raise ValueError(f"{label} must contain integers >= {minimum}.")
        return tuple(int(count) for count in value)

    if grid_ticks is not None:
        ny_ticks, nx_ticks = tick_counts(grid_ticks, 'grid_ticks', 1)
    elif num_div is None:
        ny_ticks = nx_ticks = None
    else:
        ny_div, nx_div = tick_counts(num_div, 'num_div', 0)
        ny_ticks = ny_div + 1 if ny_div else None
        nx_ticks = nx_div + 1 if nx_div else None

    finite_values = np.asarray(self.array)[np.isfinite(self.array)]
    if finite_values.size == 0:
        raise ValueError("Real-space image has no finite values to display.")
    if percentiles is not None:
        try:
            limits = np.asarray(percentiles, dtype=float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "percentiles must satisfy 0 <= low < high <= 100."
            ) from exc
        if (
            limits.shape != (2,) or not np.all(np.isfinite(limits))
            or limits[0] < 0 or limits[1] > 100 or limits[0] >= limits[1]
        ):
            raise ValueError("percentiles must satisfy 0 <= low < high <= 100.")
        low, high = np.percentile(finite_values, limits)
        vmin = low if vmin is None else vmin
        vmax = high if vmax is None else vmax
    if symmetric:
        low = np.min(finite_values) if vmin is None else vmin
        high = np.max(finite_values) if vmax is None else vmax
        limit = max(abs(low), abs(high))
        vmin, vmax = -limit, limit

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        if not hasattr(ax, 'imshow') or not hasattr(ax, 'figure'):
            raise TypeError("ax must be a Matplotlib Axes instance.")
        fig = ax.figure
    im1 = ax.imshow(
        self.array, vmin=vmin, vmax=vmax, cmap=cmap,
        extent=extent, origin='upper',
    )

    if aspect is not None:
        ax.set_aspect(aspect)

    if coords is not None and coords.size:
        if input_mode == 'calibrated':
            source_sy, source_sx = self.pixel_size
            pixel_y = (coords[:, 0] - self.origin[0]) / source_sy
            pixel_x = (coords[:, 1] - self.origin[1]) / source_sx
        else:
            pixel_y, pixel_x = coords[:, 0], coords[:, 1]
        x_positions = display_origin[1] + pixel_x * display_sx
        y_positions = display_origin[0] + pixel_y * display_sy
        ax.scatter(x_positions, y_positions, **scatter_kwargs)

    x_positions = (
        ax.get_xticks() if nx_ticks is None else np.linspace(
            display_origin[1],
            display_origin[1] + (self.shape[1] - 1) * display_sx,
            nx_ticks,
        )
    )
    y_positions = (
        ax.get_yticks() if ny_ticks is None else np.linspace(
            display_origin[0],
            display_origin[0] + (self.shape[0] - 1) * display_sy,
            ny_ticks,
        )
    )

    if axes:
        ax.set_xlabel(f"x ({axis_unit_text})", fontsize=14)
        ax.set_ylabel(f"y ({axis_unit_text})", fontsize=14)
        if nx_ticks is not None:
            ax.set_xticks(x_positions)
        if ny_ticks is not None:
            ax.set_yticks(y_positions)
        if grid:
            ax.grid(color=grid_color)
    else:
        if grid:
            for xpos in x_positions:
                ax.axvline(xpos, color=grid_color, linewidth=0.7, zorder=2)
            for ypos in y_positions:
                ax.axhline(ypos, color=grid_color, linewidth=0.7, zorder=2)
        ax.set_axis_off()

    if title is not None:
        ax.set_title(title, fontsize=18)
    if colorbar is None:
        colorbar = axes
    if colorbar:
        cb = fig.colorbar(im1, ax=ax)
        label = colorbar_label or self.quantity
        if colorbar_label is None and self.value_units is not None:
            label += f" ({self.value_units})"
        cb.set_label(label, fontsize=14)
    if scale_bar is not None:
        self._draw_scale_bar(
            ax, scale_bar, axis_unit_text, scale_bar_color,
            scale_bar_position, scale_bar_label, extent,
        )

    if show:
        plt.show()
    return fig, ax
