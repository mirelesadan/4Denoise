"""
abTEM helper functions.

Author: Adan J Mireles
Date: April 2024.
Last updated: June 2024.
"""

from abtem import *
import numpy as np
import matplotlib.pyplot as plt
import cv2

"""
    gpts : one or two int, optional
        Number of grid points in `x` and `y` describing each slice of the potential. 
        Provide either "sampling" (spacing between consecutive grid points) or 
        "gpts" (total number of grid points).
    
    sampling : one or two float, optional
        Sampling of the potential in `x` and `y` [1 / Å]. Provide either "sampling" or "gpts".
"""

def abTEM2numpy(abTEM_dp_object, resize_dims=None):
    """Return diffraction data as a NumPy array, optionally resizing its last two axes.

    ``resize_dims`` uses OpenCV's ``(width, height)`` order. Leading scan axes
    are retained and each diffraction pattern is resized independently.
    """
    measurement = abTEM_dp_object
    if measurement.is_lazy:
        computed = measurement.compute()
        measurement = measurement if computed is None else computed

    array = np.asarray(measurement.array)
    if resize_dims is None:
        return array

    if (
        not isinstance(resize_dims, (tuple, list, np.ndarray))
        or len(resize_dims) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or value <= 0
            for value in resize_dims
        )
    ):
        raise ValueError("resize_dims must be a positive integer (width, height) pair.")
    if array.ndim < 2 or array.dtype.kind not in 'iuf':
        raise ValueError("Diffraction data must be a real-valued array with at least 2 axes.")

    width, height = map(int, resize_dims)
    working = array.astype(np.result_type(array.dtype, np.float32), copy=False)
    if working.ndim == 2:
        return cv2.resize(working, (width, height), interpolation=cv2.INTER_LINEAR)

    output = np.empty(working.shape[:-2] + (height, width), dtype=working.dtype)
    for index in np.ndindex(working.shape[:-2]):
        output[index] = cv2.resize(
            working[index], (width, height), interpolation=cv2.INTER_LINEAR,
        )
    return output
    

def simulate_diff(atom_model, substrate=None, interlayer_dist=2, vacuum=2,
                  theta_tilt=0, phi_rot=0, rotZ=0, units='deg',
                  energy=80e3, semiangle_cutoff=0.5, sampling_potential=0.1, 
                  slice_thickness=0.5, sampling_probe=None, gpts=1000, device='cpu', 
                  max_angle=None, get_probe_info=False, show_model=False,
                  frozen_phonons=False, num_configs=10, sigmas=0.1, seed=100, ensemble_mean=True,
                  grid=None,auto_crop=True, 
                  crop_dim=None, com=None, pbc=True):
    """
    Simulate a diffraction pattern from an atomic model with options for rotation, tilting, and substrate integration.

    Parameters
    ----------
    atom_model : ASE.Atoms
        Atomic configuration of the material of interest.
    substrate : ASE.Atoms, optional
        Atomic configuration of the substrate.
    interlayer_dist : float, optional
        Distance between the atom model and the substrate in angstroms.
    vacuum : float, optional
        Amount of vacuum padding around the atom model's cell in angstroms.
    theta_tilt : float, optional
        Tilt angle around the phi_rot axis.
    phi_rot : float, optional
        Rotation angle of the axis about which theta_tilt is measured.
    rotZ : float, optional
        Additional rotation around the Z-axis.
    energy : float, optional
        Electron beam energy in electron volts.
    semiangle_cutoff : float, optional
        Semi-angle cutoff of the electron probe in milliradians.
    sampling_potential : float, optional
        Sampling interval for the electrostatic potential in angstroms.
    slice_thickness : float, optional
        Thickness of each potential slice in angstroms.
    sampling_probe : float, optional
        Sampling interval for the electron probe in angstroms.
    gpts : int, optional
        Number of grid points for the potential.
    device : str, optional
        Computation device to use ('cpu' or 'gpu').
    max_angle : float, optional
        Maximum scattering angle for the diffraction pattern simulation.
    units : str, optional
        Units for tilt and rotation angles ('deg' for degrees or 'rad' for radians).
    get_probe_info : bool, optional
        If True, returns a plot of the probe's profile and its FWHM.
    show_model : bool, optional
        If True, displays the final atomic model.
    frozen_phonons : bool, optional
        If True, applies frozen phonon configurations.
    num_configs : int, optional
        Number of frozen phonon configurations.
    sigmas : float, optional
        Standard deviation of atomic displacements for frozen phonons.
    seed : int, optional
        Seed for random number generation in frozen phonons.
    ensemble_mean : bool, optional
        If True, averages results from all frozen phonon configurations.
    grid : tuple, optional
        Specifies the start and end points for a grid scan in the simulation.
    auto_crop : bool, optional
        Automatically crops the diffraction pattern to sqrt(1/2) times the 
        maximum angle. This corresponds to cropping a circle to its inscribed square.

    Returns
    -------
    abtem.measurements.DiffractionPatterns
        Simulated convergent beam electron diffraction pattern.

    Notes
    -----
    Substrate integration does not involve rotation, only alignment based on 
    the specified interlayer distance and vacuum padding.
    """
    
    # Check whether the sample was tilted or rotated and apply changes to material model
    # if theta_tilt != 0 or rotZ != 0:
    atom_model = rot_lattice(atom_model, theta_tilt, phi_rot, rotZ, units, show=False, crop_dim=crop_dim, com=com, pbc=pbc)
    
    # Check whether the material model has a substrate
    if substrate is not None:
                  
        # Adjust position of material of interest with respect to the substrate
        atom_model.set_cell(substrate.cell)
        atom_model.center()
        
        # We ensure that the lowermost atom in the material of interest is at most 'interlayer_dist' from the substrate
        atom_model.positions[:, 2] += substrate.cell[2,2] - np.min(atom_model.positions[:,2]) + interlayer_dist
        
        # We combine the substrate and the material of interest into a single atom model
        atom_model = atom_model + substrate
        atom_model.center(axis=2, vacuum=vacuum)        
    
    # Apply thermal vibrations
    if frozen_phonons:
        atom_model_static = atom_model.copy()
        atom_model = FrozenPhonons(atom_model, num_configs=num_configs, 
                                  sigmas=sigmas, seed=seed, ensemble_mean=True)    
           
    potential = Potential(atom_model, sampling=sampling_potential,
                          gpts=gpts, slice_thickness=slice_thickness,
                          parametrization='kirkland', device=device, periodic=True)
    
    probe = Probe(energy=energy, semiangle_cutoff=semiangle_cutoff,)
    
    if get_probe_info:
        probe.grid.match(potential)
        probe.profiles().show();
        print(f"Probe FWHM = {probe.profiles().width().compute()} Å")
    
    # 4D-STEM Data Simulation
    if grid is not None:
        grid_scan = GridScan(
                    potential=potential,
                    start=grid[0],
                    end=grid[1],
                    sampling=probe.aperture.nyquist_sampling,
                    fractional=True)
        
        detector = PixelatedDetector()
        measurements = probe.scan(potential, scan=grid_scan, detectors=detector)
        
        return measurements        

    # Single Diffraction Pattern Simulation
    else: 
        
        # Run multislice algorithm
        probe_exit_wave = probe.multislice(potential=potential,)
                
        # Crop out diffraction data without information and disregard 10% of the outermost data
        if auto_crop:
            max_angle = np.min(probe_exit_wave.diffraction_patterns().max_angles)*np.sqrt(1/2)*0.9
            cbed_diffraction_pattern = probe_exit_wave.diffraction_patterns(max_angle=max_angle)
        elif max_angle is not None:
            cbed_diffraction_pattern = probe_exit_wave.diffraction_patterns(max_angle=max_angle)
        else:
            cbed_diffraction_pattern = probe_exit_wave.diffraction_patterns()
        
        if show_model:
            if frozen_phonons:
                show_atoms(atom_model_static, plane='xz', title='Side View of Final Model')
            else:
                show_atoms(atom_model, plane='xz', title='Side View of Final Model')
        
        if frozen_phonons:
            return cbed_diffraction_pattern.mean(0)
        else:
            return cbed_diffraction_pattern


def rot_lattice(atom_model, theta_tilt=0, phi_rot=0, rotZ=0, units='deg', show=True, crop_dim=None, com=None, pbc=True, cou=False):
    """
    Rotate the lattice of an atomic model about specified angles and axes.

    This function performs a rotation transformation on the atomic model's lattice,
    based on given tilt and rotational parameters. The rotation can be performed in
    degrees or radians and can include an optional rotation around the Z-axis.

    Parameters
    ----------
    atom_model : ase.atoms.Atoms object
        The atomic model to be transformed.
    theta_tilt : float, optional
        The tilt angle around the computed axis. Default is 0.
    phi_rot : float, optional
        The rotation angle around the z-axis to define the tilt axis. Default is 0.
    rotZ : float, optional
        Additional rotation around the Z-axis. Default is None.
    units : str, optional
        Units for angular measurements. Accepts 'deg' for degrees or 'rad' for radians. Default is 'deg'.
    show : bool, optional
        If True, plot the top and side views of the rotated lattice. Default is True.

    Returns
    -------
    tilted_atom_model : ase.atoms.Atoms object
        A new atomic model that has been rotated according to the specified parameters.

    Example
    --------
    >>> atom_model = read("my_atom_model.cif")
    >>> tilted_model = rot_lattice(atom_model, theta_tilt=30, phi_rot=45, plot=False)
    >>> print(tilted_model.positions)  # Outputs rotated positions of the atomic model

    Notes
    -----
    The rotations are applied first about the phi axis and optionally about the Z-axis.
    Rotation angles need to be specified in either degrees or radians based on 
    the 'units' parameter and must be the same units for all inputs.
    """

    if units == 'deg':
        # Convert tilt and rotation to radians if necessary
        phi_rot = np.radians(phi_rot)  # Only convert phi_rot if units are in degrees

    elif units != 'rad':
        # Validate units and convert theta_tilt if not already in degrees
        theta_tilt = np.degrees(theta_tilt)
        
        if rotZ != 0:
            rotZ = np.degrees(rotZ)
        
        raise ValueError("units must be 'rad' or 'deg'.")

    # Create a copy of the model and apply rotations
    tilted_atom_model = atom_model.copy()
    if cou:
        tilted_atom_model.rotate(theta_tilt,
                             (np.cos(phi_rot), np.sin(phi_rot), 0),
                             center='COM', 
                             rotate_cell=True
                            )
    else:
        tilted_atom_model.rotate(theta_tilt,
                             (np.cos(phi_rot), np.sin(phi_rot), 0),
                             #center='COU', 
                             #rotate_cell=True
                            )
        
    if rotZ != 0:
        if cou:
            tilted_atom_model.rotate(rotZ, 'z', 
                                 center='COM', 
                                 rotate_cell=True
                                )
        else:
            tilted_atom_model.rotate(rotZ, 'z', 
                                 # center='COU', 
                                 # rotate_cell=True
                                )
        
    if crop_dim:
        a,b,c = crop_dim
    
        # Define half-dimensions of the box
        half_dimensions = {
            "x_half": a / 2,
            "y_half": b / 2,
            "z_half": c / 2
        }
        
        # Define the bounds for the cropping box centered at COM
        crop_box = {
            "x_min": com[0] - half_dimensions["x_half"],
            "x_max": com[0] + half_dimensions["x_half"],
            "y_min": com[1] - half_dimensions["y_half"],
            "y_max": com[1] + half_dimensions["y_half"],
            "z_min": com[2] - half_dimensions["z_half"],
            "z_max": com[2] + half_dimensions["z_half"]
        }
        
        # Create a mask to select atoms within the box
        positions = tilted_atom_model.positions
        mask = (
            (positions[:, 0] >= crop_box["x_min"]) & (positions[:, 0] <= crop_box["x_max"]) &
            (positions[:, 1] >= crop_box["y_min"]) & (positions[:, 1] <= crop_box["y_max"]) &
            (positions[:, 2] >= crop_box["z_min"]) & (positions[:, 2] <= crop_box["z_max"])
        )
        
        # Use the mask to select atoms and create a cropped Atoms object
        tilted_atom_model = tilted_atom_model[mask]
        
        # Translate the lattice so the minimum coordinate is at (0, 0, 0)
        min_coords = np.min(tilted_atom_model.positions, axis=0)
        tilted_atom_model.positions -= min_coords
        
        # Update the cell dimensions of the cropped lattice
        tilted_atom_model.set_cell([a, b, c])
        tilted_atom_model.set_pbc(pbc)  # Ensure periodic boundary conditions are set
    
    # Optionally show the rotated model
    if show:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        show_atoms(tilted_atom_model, ax=ax1, title='Top view', show_cell=False)
        show_atoms(tilted_atom_model, ax=ax2, plane='xz', title='Side view',show_cell=False)

    return tilted_atom_model

def add_poisson_noise(array, counts, rng=None):
    """Sample a Poisson diffraction image with the requested expected total counts."""
    values = np.asarray(array, dtype=float)
    if not np.all(np.isfinite(values)):
        raise ValueError("array must contain only finite values.")
    if isinstance(counts, (bool, np.bool_)) or not np.isscalar(counts):
        raise ValueError("counts must be a finite non-negative number.")
    counts = float(counts)
    if not np.isfinite(counts) or counts < 0:
        raise ValueError("counts must be a finite non-negative number.")

    signal = np.clip(values, 0, None)
    total = signal.sum()
    if total == 0:
        return np.zeros(values.shape, dtype=np.int64)
    generator = np.random.default_rng() if rng is None else rng
    return generator.poisson(signal * (counts / total))
    

def show_atoms_top_and_side(atom_model,
                              figsize: tuple = (12,4),
                              show_cell: bool = True,
                              title1: str ='Top View',
                              title2: str ='Side View',
                              ):
    """Draw top (xy) and side (xz) views of an atomic model.

    Parameters
    ----------
    atom_model : ase.Atoms
        Atomic structure passed to abTEM's ``show_atoms``.
    figsize : tuple of float, optional
        Width and height of the two-panel Matplotlib figure in inches.
    show_cell : bool, optional
        Draw the unit cell in both views.
    title1, title2 : str, optional
        Titles for the top and side panels, respectively.

    Notes
    -----
    This function creates a figure as a side effect and does not return it.
    """

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    show_atoms(atom_model, ax=ax1, plane='xy', title=title1,show_cell=show_cell)
    show_atoms(atom_model, ax=ax2, plane='xz', title=title2,show_cell=show_cell)
    

def plotDP(data,
           log01=False,
           title='Simulated Diffraction Pattern',
           axes01=True,
           vmin=0.000005,
           vmax=0.0005,
           r=None,
           cmap='turbo'):
    """
    Visualize a desired diffraction pattern
    """

    diff_shape = data.shape

    # Visualize
    plt.figure(figsize=(10, 10))
    # -12 rotation yields aligned pattern
    #im1 = plt.imshow(rotate((left_midpoint),0)*mask,vmin = 0, vmax = 1850, cmap='turbo')
    
    if log01:
        
        if r is not None:
            im1 = plt.imshow(np.log(data[int(diff_shape[0]/2)-r:int(diff_shape[0]/2)+r,
                                         int(diff_shape[1]/2)-r:int(diff_shape[1]/2)+r]),
                             vmin=np.log(vmin),
                             vmax=np.log(vmax),
                             cmap='turbo')
            
        else:
            im1 = plt.imshow(np.log(data),
                             vmin=np.log(vmin),
                             vmax=np.log(vmax),
                             cmap='turbo')
    else:
        
        if r is not None:
            im1 = plt.imshow(data[int(diff_shape[0]/2)-r:int(diff_shape[0]/2)+r,
                                  int(diff_shape[1]/2)-r:int(diff_shape[1]/2)+r],
                             vmin=vmin,
                             vmax=vmax,
                             cmap=cmap)
        else:
            im1 = plt.imshow(data,
                             vmin=vmin,
                             vmax=vmax,
                             cmap=cmap)            
    if axes01:

        plt.axis('on')
        
        if r is not None:
            plt.xticks(np.arange(0, 2*r, int(r/10)))
            plt.yticks(np.arange(0, 2*r, int(r/10)))

        # plt.grid()
        ax = plt.gca()
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)   

        cb = plt.colorbar(im1, cax=cax)
        cb.ax.tick_params(labelsize=15) 
        ax.set_title(title, fontsize=18)
        plt.title("Intensity", fontsize=14)

    else:

        plt.axis('off')

    plt.show()


def simDP(atom_model,
          theta_tilt=0,
          phi_rot=0,
          plot=True,
          vmin=0.000005,
          vmax=0.0005,
          r=None,
          crop=False,
          resize_dim=None,
          gpts=1000,
          energy=80e3, 
          semiangle_cutoff=0.5, 
          sampling_potential=0.1,
          slice_thickness=1,
          sampling_probe=0.04,
          log01=False,
          cmap='turbo',
          device='cpu',
          max_angle=None,
          max_frequency=None,
          rotZ=None,
          title='Simulated Diffraction Pattern'):
    """Simulate one diffraction pattern and optionally display it.

    Parameters
    ----------
    atom_model : ase.Atoms
        Atomic structure supplied to :func:`simulate_diff`.
    theta_tilt, phi_rot : float, optional
        Model tilt and azimuth in degrees, following the default units of
        :func:`simulate_diff`.
    plot : bool, optional
        Display the simulated pattern using :func:`plotDP`.
    vmin, vmax : float, optional
        Display limits for the plot; they do not change the returned data.
    r : int or None, optional
        Half-width of the centered plot crop and, when ``crop=True``, of the
        returned array crop. Required if ``crop=True``.
    crop : bool, optional
        Crop the returned array around its center after simulation.
    resize_dim : tuple or None, optional
        Currently unused; this function does not resize its output.
    gpts : int, optional
        Grid points passed to :func:`simulate_diff`.
    energy : float, optional
        Electron energy in eV.
    semiangle_cutoff : float, optional
        Probe convergence semiangle in mrad.
    sampling_potential, slice_thickness, sampling_probe : float, optional
        Simulation sampling and slice parameters in angstroms.
    device : str, optional
        Computation device passed to :func:`simulate_diff`.
    max_angle : float or None, optional
        Maximum scattering angle passed to :func:`simulate_diff`.
    max_frequency : None, optional
        Unsupported. Passing a value raises ``ValueError``; use
        ``max_angle`` to constrain the simulated angular range.
    rotZ : float or None, optional
        In-plane model rotation in degrees; ``None`` means no rotation.
    log01 : bool, optional
        Display the logarithm of intensity when plotting.
    cmap : str, optional
        Matplotlib colormap for the plot.
    title : str, optional
        Plot title.

    Returns
    -------
    numpy.ndarray
        Simulated diffraction intensity array, transposed relative to the
        abTEM measurement. It is cropped only when ``crop=True``.
    """

    if max_frequency is not None:
        raise ValueError("max_frequency is not supported; use max_angle instead.")
    if crop and r is None:
        raise ValueError("r must be provided when crop=True.")

    cbed_pattern = simulate_diff(atom_model,
                                 theta_tilt=theta_tilt, phi_rot=phi_rot, 
                                 energy=energy, semiangle_cutoff=semiangle_cutoff, 
                                 sampling_potential=sampling_potential,slice_thickness=slice_thickness,
                                 sampling_probe=sampling_probe,gpts=gpts,device=device,
                                 max_angle=max_angle, rotZ=0 if rotZ is None else rotZ)
    
    # ky, kx = cbed_pattern.array.shape
    # (cbed_pattern**0.15).show()
    # print(f"CBED pattern dimensions: ({ky}, {kx})")
    
    ky, kx = cbed_pattern.array.shape
    # kmin = np.min([ky, kx])
    # kround = kmin - (kmin % 10)
    
    # if resize_dim is not None:
    #     diff_image = cv2.resize(cbed_pattern.array, resize_dim)
    #     diff_image = diff_image.T
    
    # else:
    diff_image = cbed_pattern.array.T

    if plot:

        plotDP(diff_image, log01=log01, vmin=vmin, vmax=vmax, r=r, cmap=cmap, title=title)
        # cbed_pattern.block_direct().show()
    if crop:

        diff_shape = diff_image.shape

        diff_image = diff_image[(diff_shape[0]//2)-r:(diff_shape[0]//2)+r,
                                (diff_shape[1]//2)-r:(diff_shape[1]//2)+r]

    return diff_image

def make_mask(center, r, mask_dim=(128,128)):
    """Return the main circular mask, retaining the simulation helper's float dtype."""
    from fourdenoise import make_mask as main_make_mask

    coordinates, radii = _select_peak_inputs([center], r, None, name='center')
    if (
        not isinstance(mask_dim, (tuple, list, np.ndarray))
        or np.ndim(mask_dim) != 1
        or len(mask_dim) != 2
    ):
        raise ValueError("mask_dim must be a positive integer (height, width) pair.")
    dimensions = tuple(_positive_peak_count(value, 'mask_dim') for value in mask_dim)
    return main_make_mask(coordinates[0], radii[0], mask_dim=dimensions).astype(float)


def _positive_peak_count(value, name):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer.")
    return int(value)


def _reciprocal_pattern(data, name):
    from fourdenoise import ReciprocalSpace

    try:
        array = np.asarray(data)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a nonempty real-valued 2D diffraction pattern.") from exc
    if array.ndim != 2 or 0 in array.shape or array.dtype.kind not in 'biuf':
        raise ValueError(f"{name} must be a nonempty real-valued 2D diffraction pattern.")
    return ReciprocalSpace(array)


def _select_peak_inputs(coords, r, spots_order, num_spots_per_order=6,
                        *, name='coords', allow_selected=False):
    """Adapt simulation order selections to the main peak-analysis inputs."""
    from fourdenoise import _peak_radii

    try:
        coordinates = np.asarray(coords)
        if np.iscomplexobj(coordinates):
            raise ValueError("Coordinates must be real.")
        coordinates = np.asarray(coordinates, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must have shape (N, 2) with finite (ky, kx) coordinates.") from exc
    if (
        coordinates.ndim != 2
        or coordinates.shape[1] != 2
        or not np.all(np.isfinite(coordinates))
    ):
        raise ValueError(f"{name} must have shape (N, 2) with finite (ky, kx) coordinates.")

    count = _positive_peak_count(num_spots_per_order, 'num_spots_per_order')
    selection = slice(None)
    selected_count = len(coordinates)
    if spots_order is not None:
        order = _positive_peak_count(spots_order, 'spots_order')
        if order > 4:
            raise ValueError("spots_order must be 1, 2, 3, 4 (all three orders), or None.")
        start, stop = (0, 3 * count) if order == 4 else ((order - 1) * count, order * count)
        selected_count = stop - start
        if not (allow_selected and len(coordinates) == selected_count):
            if len(coordinates) < stop:
                raise ValueError(f"{name} must contain at least {stop} peaks for spots_order={order}.")
            selection = slice(start, stop)

    try:
        radii = np.asarray(r)
        if np.iscomplexobj(radii):
            raise ValueError("Radii must be real.")
        radii = np.asarray(radii, dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError("r must be a scalar or contain one finite non-negative radius per peak.") from exc
    if not np.all(np.isfinite(radii)) or np.any(radii < 0):
        raise ValueError("r must contain only finite non-negative radii.")
    if radii.ndim == 1 and radii.shape == (len(coordinates),):
        radii = _peak_radii(radii, len(coordinates))[selection]
    else:
        radii = _peak_radii(radii, selected_count)
    return coordinates[selection], radii

def getCenters(dp_data, spots_order=None, r=6, coords=None, num_spots_per_order=6,
               plotSpots=False, iterations=1):
    """Refine simulation peaks using :meth:`fourdenoise.ReciprocalSpace.get_centers`.

    ``coords`` contains ``(ky, kx)`` pairs. Orders 1-3 select consecutive
    ``num_spots_per_order`` groups; order 4 selects all three groups, and
    ``None`` selects every coordinate. ``r`` is a scalar (default 6) or one
    radius per input or selected peak. Each iteration refines the previous
    centers using the main helper's center-of-mass and plotting semantics.
    """
    pattern = _reciprocal_pattern(dp_data, 'dp_data')
    iterations = _positive_peak_count(iterations, 'iterations')
    centers, radii = _select_peak_inputs(coords, r, spots_order, num_spots_per_order)
    for _ in range(iterations):
        centers = pattern.get_centers(radii, ref_coords=centers, show=plotSpots)
    return centers


def getSpotCenter(ky, kx, r, diff_data, plotSpot=False, iterations=1):
    """Refine one ``(ky, kx)`` peak with the main local-window CoM helper.

    ``r`` may be fractional. ``iterations`` repeats the refinement, and
    ``plotSpot`` is forwarded to the main helper on every iteration.
    """
    pattern = _reciprocal_pattern(diff_data, 'diff_data')
    iterations = _positive_peak_count(iterations, 'iterations')
    coordinates, radii = _select_peak_inputs([[ky, kx]], r, None)
    ky, kx = coordinates[0]
    for _ in range(iterations):
        ky, kx = pattern.get_spotCenter(ky, kx, radii[0], plotSpot=plotSpot)
    return ky, kx

def circular_mask(center_y, center_x, radius):
    """Return a Boolean disk mask on a ``2*radius`` square grid.

    Parameters
    ----------
    center_y, center_x : int
        Disk center in the generated mask's row and column coordinates.
    radius : int
        Disk radius in pixels; the grid extends from zero to twice this size.

    Returns
    -------
    numpy.ndarray of bool
        Pixels at distance at most ``radius`` from the specified center.
    """
    from fourdenoise import circular_mask as main_circular_mask

    return main_circular_mask(center_y, center_x, radius)

def rem_negs(data):
    """
    Function to replace negative values in diffraction data
    """

    data[data < 1] = 1

    return data

def getIntensities(data, spots_order=None, center=None, r=6, coords=None):
    """Integrate simulation peaks using the main clipped circular windows.

    ``spots_order`` selects six-spot groups 1-3, or all eighteen spots for
    order 4. ``None`` selects every peak. ``center`` supplies refined
    ``(ky, kx)`` pairs, either all peaks or an already selected group; when
    supplied, ``coords`` is unused. Otherwise ``coords`` is refined by the
    main helper before integration. ``r`` is a scalar or one radius per
    input or selected peak.
    """
    pattern = _reciprocal_pattern(data, 'data')
    coordinates, radii = _select_peak_inputs(
        coords if center is None else center, r, spots_order,
        name='coords' if center is None else 'center', allow_selected=center is not None,
    )
    if center is None:
        return pattern.get_intensities(radii, ref_coords=coordinates)
    return pattern.get_intensities(radii, centers=coordinates)
