# simulate_diffusion.py

# Copyright (c) 2020-2026, Christoph Gohlke
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

# %% [markdown]
"""# Simulate diffusion on a grid using Python

by [Christoph Gohlke](https://www.cgohlke.com/)

Published July, 2020. Last updated February 25, 2026.

This notebook is released under the BSD 3-Clause license.

Source code is available on [GitHub](
https://github.com/cgohlke/simulate_diffusion).

## References

This [Jupyter Notebook](https://jupyter.org/) uses the
[Python](https://www.python.org) programming language, the
[numpy](https://numpy.org/) library for n-dimensional array-programming and
linear algebra, the
[numba](https://numba.pydata.org/) Python compiler, and the
[matplotlib](https://matplotlib.org/) library for plotting.

The [mean square displacement](
https://en.wikipedia.org/wiki/Mean_squared_displacement) (MSD) of
Brownian particles in n-dimensions Euclidean space is:

$$
MSD = \left<\left(x_1(t) - x_1(0)\right)^2 +
      \left(x_2(t) - x_2(0)\right)^2 + ... +
      \left(x_n(t) - x_n(0)\right)^2\right>
    = 2nDt
$$

where $t$ is the time, $D$ the diffusion coefficient, $n$ the dimension,
$x_n(t)$ the cartesian coordinate of a particle in dimension $n$ at time $t$,
and $\left<\right>$ the ensemble average over all particles.

"""

# include simulate_diffusion_1d.py


# %% [markdown]
"""
Import libraries and modules used in this document:
"""


# %%
import copy
import math

import ipywidgets
import matplotlib
import numba
import numpy
from matplotlib import pyplot

rng = numpy.random.default_rng(12345678)  # random number generator

# %% [markdown]
"""
## Diffusion on a n-dimensional grid

In two and three dimensions the same random-walk model applies: at each time
step a particle moves one grid step along exactly one randomly chosen axis
(or stays still). The MSD generalises to $n$ dimensions as:

$$MSD(t) = 2nDt$$

where $n$ is the number of spatial dimensions. The diffusion coefficient $D$
is therefore recovered as $D = \text{slope} \,/\, (2n)$.

The 1D look-up table (LUT) extends naturally: `directions` becomes a 2D array
of shape `(sampling_period, dimensions)`, where each row is a displacement
vector such as `[1, 0, 0]`, `[0, -1, 0]`, or `[0, 0, 0]` for no move.
One random integer per step indexes a row to give the full displacement.

To make the code more modular, manageable, extensible, and reusable, it is
refactored into small functions for simulation, particle counting,
analysis, and plotting.
Hook functions allow customizing the initialization and restriction of
particle movements. A simple detector "particle counter box" and methods
to analyze and plot particle counts are added.

Diffusion in one, two, and three dimensions are simulated and compared.

"""


# %%
def simulate_diffusion(
    dimensions,
    duration,
    sampling_period,
    number_particles,
    diffusion_speed,
    diffusion_model,
    diffusion_model_args,
    positions_init,
    positions_init_args,
):
    """Return nD positions of all particles for duration of simulation."""
    assert 0 < dimensions < 8
    assert sampling_period > diffusion_speed * (dimensions + 1) * 2

    # generate a look-up-table of directions to move.
    # this table will be indexed by random numbers in range 0 to
    # `sampling_period`
    directions = numpy.zeros((sampling_period, dimensions), dtype=numpy.int32)

    # generate combinations of all possible relative moves in all dimensions
    all_possible_directions = numpy.stack(
        numpy.meshgrid(*([-1, 0, 1],) * dimensions), -1
    ).reshape(-1, dimensions)

    index = 0
    for direction in all_possible_directions:
        if numpy.sum(numpy.abs(direction)) != 1:
            # particles can move only in one dimension per sampling_period
            continue

        # move the particle in the specified direction if random number is
        # between `index` and `index + diffusion_speed * dimensions`
        directions[index : index + diffusion_speed * dimensions] = direction

        index += diffusion_speed * dimensions

    # get a random number between 0 and `sampling_period`
    # for all particles and sampling periods in the duration of the simulation
    random_numbers = rng.integers(
        sampling_period, size=(number_particles, duration)
    )

    # index the first axis in the `directions` look-up-table with the random
    # numbers to obtain the relative moves of all particles for all sampling
    # periods
    random_moves = numpy.take(directions, random_numbers, axis=0)

    # set the initial positions of particles using a hook function
    positions_init(random_moves, **positions_init_args)

    if diffusion_model is None:
        return random_moves

    # calculate the positions of particles from the random moves using a
    # hook function
    return diffusion_model(random_moves, **diffusion_model_args)


def positions_init_origin(random_moves):
    """Set in-place initial position of particles to origin."""
    random_moves[:, 0] = 0


def diffusion_model_unconstrained(random_moves, **kwargs):
    """Diffusion with no constraints."""
    return numpy.cumsum(random_moves, axis=1)


def particle_counter_box(positions, counter_shape=None, counter_position=None):
    """Return number of particles in observation box over time.

    Also return the indices of particles that were counted.

    """
    dimensions = positions.shape[-1]
    if counter_shape is None:
        counter_shape = (1,) * dimensions  # one element
    if counter_position is None:
        counter_position = (0,) * dimensions  # center
    lower = tuple(
        p - s // 2
        for p, s in zip(counter_position, counter_shape, strict=True)
    )
    upper = tuple(
        p + s // 2 + s % 2
        for p, s in zip(counter_position, counter_shape, strict=True)
    )
    in_box = numpy.all((positions >= lower) & (positions < upper), axis=-1)
    particle_counts = numpy.sum(in_box, axis=0)
    particles_counted = numpy.nonzero(numpy.any(in_box, axis=1))
    return particle_counts, particles_counted


def calculate_msd_d(positions):
    """Return mean square displacement and D of simulated positions.

    MSD is computed as displacement squared relative to each particle's
    initial position, so randomized starting positions are handled correctly.

    """
    number_particles, duration, dimensions = positions.shape
    msd = numpy.mean(
        numpy.square(positions - positions[:, 0:1, :]), axis=(0, -1)
    )
    time = numpy.arange(duration)[..., numpy.newaxis]
    slope = numpy.linalg.lstsq(time, msd, rcond=None)[0][0]
    d = slope / (2 * dimensions)
    return msd, d


def plot_positions(positions, selection=None, ax=None, title=None, label=None):
    """Plot positions of selected particles over duration of simulation."""
    number_particles, duration, dimensions = positions.shape
    if selection is None:
        selection = slice(1)  # first particle
    threed = dimensions > 2 and not isinstance(selection, int)
    ax_ = ax
    if ax is None:
        fig = pyplot.figure(figsize=(7.0, 7.0) if threed else None)
        ax = fig.add_subplot(111, projection='3d' if threed else None)
    if title is None:
        title = 'Selected particle positions'
    ax.set_title(title)
    if isinstance(selection, int):
        time = numpy.arange(duration)
        ax.set_xlabel('time')
        ax.set_ylabel('position')
        label = '' if label is None else label + ' '
        for i, dim in zip(
            range(dimensions - 1, -1, -1), 'xyzwvuts', strict=False
        ):
            ax.plot(time, positions[selection, :, i], label=label + dim)
    elif dimensions == 1:
        time = numpy.arange(duration)
        ax.set_xlabel('time')
        ax.set_ylabel('x')
        for pos in positions[selection]:
            ax.plot(time, pos, label=label)
    elif dimensions == 2:
        ax.set_aspect('equal')
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        for pos in positions[selection]:
            ax.plot(pos[:, 1], pos[:, 0], label=label)
    else:
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        for pos in positions[selection]:
            ax.plot(pos[:, 2], pos[:, 1], pos[:, 0], label=label)
    if label is not None:
        ax.legend()
    if ax_ is None:
        pyplot.show()


def plot_msd(msd, d, dimensions=None, ax=None, labels=('simulation', 'fit')):
    """Plot MSD and line fit."""
    duration = msd.shape[0]
    time = numpy.arange(duration)
    ax_ = ax
    if ax is None:
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
    ax.set_title('MSD and line fit')
    ax.set_xlabel('time')
    ax.set_ylabel('MSD')
    try:
        label0, label1 = labels
    except Exception:
        label0, label1 = None, None
    if dimensions:
        ax.plot(time, msd, '.', label=label0)
        ax.plot(time, d * 2 * dimensions * time, '-', lw=3, label=label1)
    else:
        ax.plot(time, msd, label=label0)
    if label0 or label1:
        ax.legend()
    if ax_ is None:
        pyplot.show()


def plot_particle_counts(particle_counts, ax=None, label=None):
    """Plot number of detected particles over time."""
    duration = particle_counts.shape[0]
    time = numpy.arange(duration)
    ax_ = ax
    if ax is None:
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
    ax.set_title('Detected particles')
    ax.set_xlabel('time')
    ax.set_ylabel('count')
    ax.plot(time, particle_counts, label=label)
    if label:
        ax.legend()
    if ax_ is None:
        pyplot.show()


def example_nd_simulations():
    """Compare diffusion in 1, 2, and 3 dimensions."""
    # create three empty plots
    plots = []
    for _ in range(3):
        fig = pyplot.figure()
        plots.append(fig.add_subplot(111))

    # iterate over dimensions 1 to 3
    for dimensions in range(1, 4):
        # define simulation parameters
        simulation_args = {
            'dimensions': dimensions,
            'duration': 2500,
            'sampling_period': 1000,
            'number_particles': 1000,
            'diffusion_speed': 10,
            'positions_init': positions_init_origin,
            'positions_init_args': {},
            'diffusion_model': diffusion_model_unconstrained,
            'diffusion_model_args': {},
        }

        particle_counter_args = {
            'counter_position': (0,) * dimensions,
            'counter_shape': (10,) * dimensions,
        }

        # run simulation of model
        positions = simulate_diffusion(**simulation_args)

        # count particles
        particle_counts, particles_counted = particle_counter_box(
            positions, **particle_counter_args
        )

        # analyze positions and counted particles
        msd, D = calculate_msd_d(positions)

        # plot results of simulation and analysis
        label = f'{dimensions}D'
        plot_positions(positions, 0, ax=plots[0], label=label)
        plot_particle_counts(particle_counts, ax=plots[1], label=label)
        plot_msd(
            msd,
            D,
            ax=plots[2],
            labels=(
                f'{label} D={D * simulation_args["sampling_period"]:.3f}',
                None,
            ),
        )

    pyplot.show()


# %time example_nd_simulations()


# %% [markdown]
"""
## Constrained diffusion in a box

To demonstrate diffusion under restrictions, particles are placed in a box.
Particles can diffuse unconstrained within the box. Three different cases are
explored when particles hit a box boundary:

* particles cannot leave the box.
* particles leaving the box immediately enter on the opposite side.
* particles leaving the box cannot re-enter (they are "absorbed" by the
  boundaries).

Hook functions are defined for each case and passed to the simulation function.

"""


# %%
def diffusion_model_box_closed(random_moves, box_shape):
    """Diffusion in a box. Particles cannot leave box."""
    positions = random_moves.copy()  # modify in-place
    _number_particles, duration, _dimensions = positions.shape
    lower = tuple(-s // 2 for s in box_shape)
    upper = tuple(s // 2 + s % 2 - 1 for s in box_shape)
    for time in range(1, duration):
        temp = positions[:, time]
        temp += positions[:, time - 1]  # cumsum axis=1
        numpy.clip(temp, lower, upper, out=temp)
    return positions


def diffusion_model_box_cyclic(random_moves, box_shape):
    """Diffusion in a box. Particles leaving box enter on opposite side."""
    positions = random_moves.copy()
    _number_particles, duration, _dimensions = positions.shape
    lower = tuple(-s // 2 for s in box_shape)
    for time in range(1, duration):
        temp = positions[:, time]
        temp += positions[:, time - 1]  # cumsum axis=1
        temp -= lower
        temp %= box_shape
        temp += lower
    return positions


def diffusion_model_box_absorbing(random_moves, box_shape):
    """Diffusion in a box. Particles leaving box never re-enter."""
    number_particles, duration, _dimensions = random_moves.shape
    positions = numpy.cumsum(random_moves, axis=1)
    lower = tuple(-s // 2 for s in box_shape)
    upper = tuple(s // 2 + s % 2 for s in box_shape)
    leaving_box = numpy.argmax(
        numpy.any((positions < lower) | (positions >= upper), axis=-1), axis=-1
    )
    for particle in range(number_particles):
        index = leaving_box[particle]
        if 0 < index < duration - 1:
            positions[particle, index:] = positions[particle, index + 1]
    return positions


def example_box_model_simulations(dimensions=3):
    """Compare box diffusion models."""
    # define simulation parameters
    box_model_args = {'box_shape': (20,) * dimensions}

    simulation_args = {
        'dimensions': dimensions,
        'duration': 2500,
        'sampling_period': 1000,
        'number_particles': 1000,
        'diffusion_speed': 10,
        # 'diffusion_model': will be set later
        'diffusion_model_args': box_model_args,
        'positions_init': positions_init_origin,
        'positions_init_args': {},
    }

    particle_counter_args = {
        'counter_position': (0,) * dimensions,
        'counter_shape': (10,) * dimensions,
    }

    # create empty plots
    fig = pyplot.figure(figsize=(7.0, 7.0) if dimensions > 2 else None)
    plot0 = fig.add_subplot(111, projection='3d' if dimensions > 2 else None)
    fig = pyplot.figure()
    plot1 = fig.add_subplot(111)
    fig = pyplot.figure()
    plot2 = fig.add_subplot(111)

    # iterate over diffusion models
    for diffusion_model in (
        diffusion_model_unconstrained,
        diffusion_model_box_closed,
        diffusion_model_box_cyclic,
        diffusion_model_box_absorbing,
    ):
        # run simulation of model
        positions = simulate_diffusion(
            diffusion_model=diffusion_model, **simulation_args
        )

        # count particles
        particle_counts, _particles_counted = particle_counter_box(
            positions, **particle_counter_args
        )

        # analyze positions and counted particles
        msd, D = calculate_msd_d(positions)

        # plot results of simulation and analysis
        model = diffusion_model.__name__[16:]
        plot_positions(positions, ax=plot0, label=model)
        plot_particle_counts(particle_counts, ax=plot1, label=model)
        plot_msd(
            msd,
            D,
            ax=plot2,
            labels=(
                f'{model} D={D * simulation_args["sampling_period"]:.3f}',
                None,
            ),
        )

    pyplot.show()


# %time example_box_model_simulations()


# %% [markdown]
"""
## Membrane raft diffusion model

A membrane raft is approximated by a cylinder of a certain radius
(in y, x dimensions) and thickness (in z dimension). It is placed at the
origin. Particles moving into the cylinder are retained for some sampling
periods and then released on the negative z side. A particle counter detector,
elongated in the positive z direction, is placed at a distance in the y
dimension (x=0, z=0).

The average trajectory of all detected particles, the number of particles in
the observation volume over time and the MSD over time are plotted.

"""


# %%
@numba.jit(nopython=True)
def diffusion_model_raft(random_moves, raft_shape, raft_delay):
    """Membrane raft diffusion model."""
    positions = random_moves.copy()
    number_particles, duration, dimensions = positions.shape
    assert dimensions == 3
    assert raft_delay >= 0
    length, radius = raft_shape
    radius *= radius
    for particle in range(number_particles):
        zyx = positions[particle, 0].copy()
        t = 1
        while t < duration:
            zyx += positions[particle, t]
            if (
                zyx[0] >= 0
                and zyx[0] < length
                and zyx[1] * zyx[1] + zyx[2] * zyx[2] < radius
            ):
                # particle entered raft, glue it for `raft_delay`
                zyx[0] = 0
                positions[particle, t : t + raft_delay] = zyx
                t += raft_delay
                if t < duration:
                    # release particle on -z side
                    zyx[0] = -1
                    positions[particle, t] = zyx
            else:
                positions[particle, t] = zyx
            t += 1
    return positions


def example_raft_simulation():
    """Run simulation of membrane raft model."""
    raft_model_args = {'raft_shape': (2, 10), 'raft_delay': 20}

    simulation_args = {
        'dimensions': 3,
        'duration': 5000,
        'sampling_period': 1000,
        'number_particles': 5000,
        'diffusion_speed': 10,
        'diffusion_model': diffusion_model_raft,
        'diffusion_model_args': raft_model_args,
        'positions_init': positions_init_origin,
        'positions_init_args': {},
    }

    particle_counter_args = {
        'counter_position': (50, 15, 0),
        'counter_shape': (100, 1, 1),
    }

    positions = simulate_diffusion(**simulation_args)

    particle_counts, particles_counted = particle_counter_box(
        positions, **particle_counter_args
    )

    msd, D = calculate_msd_d(positions)

    average_trajectory = numpy.average(positions[particles_counted], axis=0)
    plot_positions(
        average_trajectory[numpy.newaxis],
        0,
        title='Average trajectory of detected particles',
    )
    plot_particle_counts(particle_counts)

    return positions  # to analyze further


# %time POSITIONS = example_raft_simulation()


# %% [markdown]
"""
### Plot the spatial distribution of particles over time

For visualizing the spatial distribution of all particles over the duration
of the simulation, a multi-dimensional histogram of particle positions is
calculated. The y and x dimensions are reduced to one radial dimension
(`radius = hypot(x, y)`). The number of particles at a certain sampling period,
z-position, and radius is counted and then normalized by the number of voxels
at a certain radius. The three-dimensional histogram is plotted in log-scale
as a series of color-coded 2D images using interactive Jupyter widgets.

"""


# %%
@numba.jit(nopython=True)
def _histogram_zr(positions, axes=(0, 1, 2)):
    """Return (z, r) histograms of particles over duration of simulation."""
    number_particles, duration, dimensions = positions.shape
    zax, yax, xax = axes
    assert dimensions == 3
    radial = positions[..., :2].copy()
    zmin = radial[..., zax].min()
    radius = numpy.hypot(positions[..., yax], positions[..., xax])
    radius += 0.5
    radial[..., 0] -= zmin
    radial[..., 1] = radius
    zmax = radial[..., 0].max()
    rmax = radial[..., 1].max()
    hist = numpy.zeros((duration, zmax + 1, rmax + 1), dtype=numpy.uint32)
    for p in range(number_particles):
        for t in range(duration):
            z = radial[p, t, 0]
            r = radial[p, t, 1]
            hist[t, z, r] += 1
    return hist, (zmin, 0)


def _histogram_zr_norm(radius):
    """Return number of particles at radius for uniform distribution."""
    yx = numpy.mgrid[:radius, :radius]
    r = numpy.hypot(yx[0], yx[1])
    r += 0.5
    r = r.astype(numpy.int32)
    norm = numpy.bincount(r.ravel())
    norm = norm[:radius]
    norm[1:] *= 4
    return norm


def plot_histogram_zr(positions, axes=(0, 1, 2)):
    """Interactively plot histograms of (z, r) positions as log-image."""
    number_particles, duration, dimensions = positions.shape
    hist, (zmin, _) = _histogram_zr(positions, axes=axes)
    norm = _histogram_zr_norm(hist.shape[-1])
    hist = numpy.log10(numpy.where(hist > 0.0, hist, numpy.nan) / norm)

    cmap = copy.copy(matplotlib.colormaps.get_cmap('bwr'))
    cmap.set_bad(color='black')
    vmax = math.log10(number_particles)

    def _plot(time):
        pyplot.figure(figsize=(4.8, 6.4))
        pyplot.title('normalized log10 histogram of positions')
        pyplot.xlabel('hypot(x, y)')
        pyplot.ylabel('z')
        pyplot.yticks(
            [0, -zmin, hist.shape[1] - 1], [zmin, 0, hist.shape[1] + zmin]
        )
        pyplot.imshow(
            hist[time], vmin=-vmax, vmax=vmax, cmap=cmap, origin='lower'
        )
        pyplot.colorbar()
        pyplot.show()

    ipywidgets.interact(
        _plot,
        time=ipywidgets.IntSlider(
            value=hist.shape[0] // 8,
            min=0,
            max=hist.shape[0] - 1,
            continuous_update=False,
        ),
    )


# %time plot_histogram_zr(POSITIONS)


# %% [markdown]
"""
## Multiple particle types

The simulation code is extended to handle multiple types of particles of
different diffusion speeds, diffusion models, and initial positions.
The diffusion of particles of different types is simulated separately using
the existing `simulate_diffusion` function. The positions of different
particle types are then joined. This simulation mode does not allow for
interacting or reacting particles.

Functions are added to initialize particle positions with a uniform
distribution or known positions.

As an example, three diffusion simulations are compared:

0. particles restricted in a box.
1. faster diffusing, unconstrained particles.
2. a mixture of above particles.

"""


# %%
def simulate_diffusion_pt(
    dimensions, duration, sampling_period, particle_types
):
    """Return positions of particles of different types."""
    return positions_join(
        [
            simulate_diffusion(
                dimensions,
                duration,
                sampling_period,
                number_particles=particle_type['number_particles'],
                diffusion_speed=particle_type['diffusion_speed'],
                diffusion_model=particle_type.get(
                    'diffusion_model', diffusion_model_unconstrained
                ),
                diffusion_model_args=particle_type.get(
                    'diffusion_model_args', {}
                ),
                positions_init=particle_type.get(
                    'positions_init', positions_init_origin
                ),
                positions_init_args=particle_type.get(
                    'positions_init_args', {}
                ),
            )
            for particle_type in particle_types
        ]
    )


def positions_join(positions_sequence):
    """Return concatenated positions arrays."""
    return numpy.concatenate(positions_sequence)


def positions_init_uniform(random_moves, init_shape=None, init_position=None):
    """Set initial position of particles to uniform distribution."""
    number_particles, duration, dimensions = random_moves.shape
    if init_position is None:
        init_position = (0,) * dimensions  # center
    if init_shape is None:
        init_shape = (256,) * dimensions  # box
    for dim, (pos, size) in enumerate(
        zip(init_position, init_shape, strict=True)
    ):
        temp = rng.integers(0, size, number_particles)
        temp += pos - size // 2
        random_moves[:, 0, dim] = temp


def positions_init_array(random_moves, init_positions):
    """Set initial position of particles using positions array.

    `init_positions` must be shaped (N, T, D). The first time step
    (index 0) is used as the initial position for each particle.
    Pass `init_positions[:, numpy.newaxis]` for a plain (N, D) array.

    """
    random_moves[:, 0] = init_positions[:, 0]


def example_particle_type_simulations():
    """Compare diffusion of different and combined particle types."""
    particle_type_0 = [
        {
            'number_particles': 1000,
            'diffusion_speed': 10,
            'diffusion_model': diffusion_model_box_cyclic,
            'diffusion_model_args': {'box_shape': (20, 20, 20)},
            'positions_init': positions_init_uniform,
            'positions_init_args': {'init_shape': (20, 20, 20)},
        }
    ]
    particle_type_1 = [
        {
            'number_particles': 1000,
            'diffusion_speed': 20,
            'diffusion_model': diffusion_model_unconstrained,
            'diffusion_model_args': {},
            'positions_init': positions_init_uniform,
            'positions_init_args': {'init_shape': (20, 20, 20)},
        }
    ]

    simulation_args = {
        'dimensions': 3,
        'duration': 2000,
        'sampling_period': 1000,
    }

    particle_counter_args = {
        'counter_position': (0, 0, 0),
        'counter_shape': (10, 10, 10),
    }

    plots = []
    for _ in range(2):
        fig = pyplot.figure()
        plots.append(fig.add_subplot(111))

    for particle_types, label in zip(
        (particle_type_0, particle_type_1, particle_type_0 + particle_type_1),
        ('type 0', 'type 1', 'type 0 and 1'),
        strict=True,
    ):
        positions = simulate_diffusion_pt(
            particle_types=particle_types, **simulation_args
        )

        particle_counts, _particles_counted = particle_counter_box(
            positions, **particle_counter_args
        )

        msd, D = calculate_msd_d(positions)

        plot_particle_counts(particle_counts, ax=plots[0], label=label)
        plot_msd(
            msd,
            D,
            ax=plots[1],
            labels=(
                f'{label} D={D * simulation_args["sampling_period"]:.3f}',
                None,
            ),
        )

    pyplot.show()


# %time example_particle_type_simulations()


# %% [markdown]
"""
## Fluorescence fluctuations

To approximate fluorescence fluctuations experiments, functions handling
excitation of fluorophores and detection of emitted photons are defined.

### Laser Scanning

A confocal laser scanning system is approximated by exciting particles in a
small volume for a certain number of sampling intervals (dwell time) before
moving the volume to another position. The positions of the scanner can be
static (point, 0D), circular, or rasterizing lines (1D), planes (2D), or
volumes (3D).

### Point Spread Function (PSF)

...

"""


# %%
def psf_gaussian(psf_shape, psf_waist, psf_physical_size=1, psf_nphoton=2):
    """Return 3D gaussian approximation of PSF."""

    def f(index):
        s = psf_shape[index] // 2 * psf_physical_size
        c = numpy.linspace(-s, s, psf_shape[index])
        c *= c
        c *= -2.0 / (psf_waist[index] * psf_waist[index])
        return c

    psf = numpy.exp(
        numpy.sum(
            numpy.meshgrid(f(0), f(1), f(2), indexing='ij', sparse=False),
            axis=0,
        )
    )
    if psf_nphoton != 1:
        numpy.power(psf, psf_nphoton, out=psf)
    return psf


def laser_scanning(
    positions,
    particle_brightness,
    scanning_mode='point',
    scanning_position=(0, 0, 0),
    scanning_shape=(32, 32, 32),
    scanning_strides=(1, 1, 1),  # for raster
    scanning_samples=32,  # for circular
    scanning_dwelltime=1,
    scanning_intervals=(0, 0, 0, 0),  # eg. retrace
    scanning_intensity=1.0,
    scanning_psf=None,
    scanning_axes=(-3, -2, -1),
    scanning_offset=0.0,
    scanning_gain=1.0,
    scanning_gamma=1.0,
    scanning_bitdepth=12,
):
    """Return signal, scan times and positions of laser scanning."""
    if scanning_psf is None:
        scanning_psf = numpy.array(scanning_intensity, numpy.float64)
        scanning_psf.shape = 1, 1, 1
    else:
        scanning_psf = numpy.array(scanning_psf, numpy.float64)
        if scanning_intensity != 1.0:
            scanning_psf *= scanning_intensity
        assert scanning_psf.ndim == 3

    dimensions = {'circle': 0, 'point': 0, 'line': 1, 'image': 2, 'volume': 3}[
        scanning_mode
    ]

    if scanning_mode == 'circle':
        scan_positions = _scanning_circular(
            scanning_position, scanning_shape, scanning_samples, scanning_axes
        )
    else:
        scan_positions = _scanning_raster(
            scanning_position,
            scanning_shape,
            scanning_strides,
            scanning_axes,
            dimensions,
        )

    intensities, scan_times = _scanning_integrate(
        positions,
        scan_positions,
        particle_brightness,
        scanning_dwelltime,
        scanning_intervals,
        scanning_psf,
        dimensions,
    )

    intensities = intensities.squeeze()

    signal = _photon_counter(
        intensities,
        scanning_offset,
        scanning_gain,
        scanning_gamma,
        scanning_bitdepth,
    )
    return signal, scan_times, scan_positions


def _scanning_raster(position, shape, strides, axes, dims):
    """Return raster scan positions."""

    def arange(axis):
        s = shape[axis]
        d = strides[axis]
        p = position[axis] + ((s - 1) % (s // d)) // 2
        return numpy.arange(
            p - s // 2, p + s // 2 + s % 2, d, dtype=numpy.int32
        )

    ndims = len(axes)
    coordinates = [None] * ndims
    for i, ax in enumerate(axes):
        if ax % ndims >= ndims - dims:
            c = arange(i)
        else:
            c = numpy.array([position[i]], dtype=numpy.int32)
        coordinates[i] = c
    return numpy.moveaxis(
        numpy.stack(
            [
                numpy.transpose(v, axes)
                for v in numpy.meshgrid(*coordinates, indexing='ij')
            ]
        ),
        0,
        -1,
    )


def _scanning_circular(position, shape, samples, axes):
    """Return circular scan positions."""
    r = numpy.linspace(
        0, 2 * math.pi, samples, endpoint=False, dtype=numpy.float64
    )
    positions = numpy.empty((samples, 3), dtype=numpy.int32)
    s = numpy.sin(r)
    s *= shape[axes[-1]] / 2
    c = numpy.cos(r)
    c *= shape[axes[-2]] / 2
    positions[:, axes[-3]] = position[axes[-3]]
    positions[:, axes[-2]] = numpy.round(position[axes[-2]] + c)
    positions[:, axes[-1]] = numpy.round(position[axes[-1]] + s)
    return positions


@numba.jit(nopython=True)
def _scanning_integrate(
    positions,
    scan_positions,
    particle_brightness,
    dwelltime,
    intervals,
    psf,
    dimensions,
):
    """Return laser scanning intensities and scan times."""
    interval = intervals[-dimensions - 1]
    duration = scan_positions.size // scan_positions.shape[-1] * dwelltime
    for i in range(dimensions):
        duration += intervals[-i - 1] * (scan_positions.shape[-i - 1] - 1)
    periods = (positions.shape[1] + interval) // (duration + interval)
    shape = (periods, *scan_positions.shape[:-1])
    intensities = numpy.empty(shape, dtype=numpy.float64)
    intensities_ = intensities.flat
    scan_times = numpy.empty(intensities.size, dtype=numpy.float64)
    psf_shape = psf.shape
    psf_shape_2 = (psf_shape[0] // 2, psf_shape[1] // 2, psf_shape[2] // 2)
    sample = 0
    time = 0
    for _volume in range(shape[0]):
        for image in range(shape[1]):
            for line in range(shape[2]):
                for pixel in range(shape[3]):
                    scan_times[sample] = time
                    sz, sy, sx = scan_positions[image, line, pixel]
                    sz -= psf_shape_2[0]
                    sy -= psf_shape_2[1]
                    sx -= psf_shape_2[2]
                    si = 0.0
                    for _m in range(dwelltime):
                        for particle in range(positions.shape[0]):
                            pz = positions[particle, time, 0] - sz
                            py = positions[particle, time, 1] - sy
                            px = positions[particle, time, 2] - sx
                            if (
                                pz >= 0
                                and py >= 0
                                and px >= 0
                                and pz < psf_shape[0]
                                and py < psf_shape[1]
                                and px < psf_shape[2]
                            ):
                                pi = psf[pz, py, px]
                                pi *= particle_brightness[particle]
                                si += pi
                        time += 1
                    intensities_[sample] = si
                    sample += 1
                    time += intervals[-1]
                time += intervals[-2]
            time += intervals[-3]
        time += intervals[-4]
    return intensities, scan_times


def _photon_counter(
    intensities, offset=0.0, gain=1.0, gamma=1.0, bitdepth=32, *, poisson=True
):
    """Return Poisson distributed, digitized intensities."""
    max_int = 2**bitdepth - 1
    signal = intensities.astype(numpy.float64)
    signal -= offset
    signal *= gain
    if gamma != 1.0:
        signal /= max_int
        numpy.power(signal, 1 / gamma, out=signal)
        signal *= max_int
    if poisson:
        signal = rng.poisson(signal)
        numpy.clip(signal, 0, max_int, out=signal)
    else:
        eps = numpy.finfo(numpy.float64).eps * 2
        numpy.clip(signal, 0, max_int - eps, out=signal)
        numpy.floor(signal, out=signal)
    if bitdepth > 32:
        dtype = numpy.uint64
    elif bitdepth > 16:
        dtype = numpy.uint32
    elif bitdepth > 8:
        dtype = numpy.uint16
    else:
        dtype = numpy.uint8
    return signal.astype(dtype, copy=False)


def _particle_ids(particle_types):
    """Return IDs of particles in positions axes."""
    number_particles = sum(
        particle_type['number_particles'] for particle_type in particle_types
    )
    ids = numpy.empty(number_particles, dtype=numpy.uint8)
    index = 0
    for i, particle_type in enumerate(particle_types):
        number_particles = particle_type['number_particles']
        ids[index : index + number_particles] = i
        index += number_particles
    return ids


def _particle_brightness(particle_types):
    """Return brightness of particles in positions axes."""
    return numpy.array(
        [pt.get('particle_brightness', 1.0) for pt in particle_types],
        dtype=numpy.float64,
    ).take(_particle_ids(particle_types))


def plot_timeseries(
    timeseries, vmin=0, vmax=None, title=None, label=None, ax=None
):
    """Plot timeseries."""
    ax_ = ax
    if ax is None:
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
    if title is None:
        title = 'timeseries'
    ax.set_title(title)
    ax.set_xlabel('sample')
    ax.set_ylabel('intensity')
    ax.plot(timeseries, label=label)
    ax.set_ylim(bottom=vmin, top=vmax)
    if label:
        ax.legend()
    if ax_ is None:
        pyplot.show()


def plot_images(images, vmin=0, vmax=None, title=None):
    """Interactively plot series of images."""
    count, y, x = images.shape
    if title is None:
        title = 'Images'
    if vmax is None:
        vmax = images.max()

    def _plot(sample):
        pyplot.figure()
        pyplot.title(title)
        pyplot.imshow(images[sample], vmin=vmin, vmax=vmax)
        pyplot.colorbar()
        pyplot.show()

    ipywidgets.interact(
        _plot,
        sample=ipywidgets.IntSlider(
            value=0, min=0, max=count - 1, continuous_update=False
        ),
    )


def example_scan_positions():
    """Plot scan positions of different scanning modes."""
    scanning_position = (2, 1, 0)
    scanning_shape = (32, 32, 32)
    scanning_strides = (4, 4, 4)  # for raster
    scanning_samples = 32  # for circular
    scanning_axes = (-3, -2, -1)

    for scanning_mode in ('point', 'line', 'circle', 'image', 'volume'):
        dimensions = {
            'circle': 0,
            'point': 0,
            'line': 1,
            'image': 2,
            'volume': 3,
        }[scanning_mode]

        if scanning_mode == 'circle':
            scan_positions = _scanning_circular(
                scanning_position,
                scanning_shape,
                scanning_samples,
                scanning_axes,
            )
        else:
            scan_positions = _scanning_raster(
                scanning_position,
                scanning_shape,
                scanning_strides,
                scanning_axes,
                dimensions,
            )
        scan_positions.shape = (-1, 3)

        fig = pyplot.figure()
        ax = fig.add_subplot(111)
        ax.set_title(f'{scanning_mode} scanning')
        ax.set_xlabel('sample index')
        ax.set_ylabel('position')
        ax.plot(scan_positions[:, 2], '.', label='x')
        ax.plot(scan_positions[:, 1], '.', label='y')
        ax.plot(scan_positions[:, 0], '.', label='z')
        ax.legend()
        pyplot.show()


# %time example_scan_positions()


# %% [markdown]
"""
### Point FCS and Photon Counting Histogram (PCH)

...

"""


# %%
def calculate_pch(photon_counts, maxcount=None, bins=10):
    """Return frequency of number of photons from observation volume."""
    if maxcount is None:
        maxcount = numpy.max(photon_counts)
    if maxcount <= bins:
        hist = numpy.bincount(photon_counts, minlength=maxcount)
        bins = numpy.arange(len(hist), dtype=numpy.int32)
    else:
        hist, bins = numpy.histogram(photon_counts, bins, (0, maxcount))
        bins += (bins[1] - bins[0]) / 2
        bins = bins[:-1]
    return hist, bins


def plot_pch(pch, bins, ax=None, label=None):
    """Plot photon counting histogram."""
    ax_ = ax
    if ax is None:
        fig = pyplot.figure()
        ax = fig.add_subplot(111)
    ax.set_title('Photon counting histogram')
    ax.set_xlabel('counts')
    ax.set_ylabel('frequency')
    ax.semilogy(bins, pch, '.-', label=label)
    # ax.set_ylim(bottom=1)
    if label:
        ax.legend()
    if ax_ is None:
        pyplot.show()


def example_point_fcs():
    """Laser scanning example."""
    # physical_size = 0.05  # micrometer per pixel
    # physical_time = 1.0  # microseconds per sampling period

    simulation_args = {
        'dimensions': 3,
        'duration': 2**15,
        'sampling_period': 1000,
    }

    particle_types = [
        {
            'number_particles': 1000,
            'diffusion_speed': 10,
            'diffusion_model': diffusion_model_box_cyclic,
            'diffusion_model_args': {'box_shape': (256, 256, 256)},
            'positions_init': positions_init_uniform,
            'positions_init_args': {'init_shape': (256, 256, 256)},
            'particle_brightness': 1.0,
        },
        {
            'number_particles': 1000,
            'diffusion_speed': 20,
            'diffusion_model': diffusion_model_box_cyclic,
            'diffusion_model_args': {'box_shape': (256, 256, 256)},
            'positions_init': positions_init_uniform,
            'positions_init_args': {'init_shape': (256, 256, 256)},
            'particle_brightness': 2.0,
        },
    ]

    scanning_args = {
        'scanning_mode': 'point',
        'scanning_position': (0, 0, 0),
        'scanning_shape': (256, 256, 256),
        'scanning_strides': (8, 8, 8),  # for raster
        'scanning_samples': 32,  # for circular
        'scanning_dwelltime': 1,
        'scanning_intervals': (0, 0, 0, 0),  # eg. retrace
        'scanning_intensity': 1.0,
        'scanning_axes': (-3, -2, -1),
        'scanning_offset': 0.0,
        'scanning_gain': 1.0,
        'scanning_gamma': 1.0,
        'scanning_bitdepth': 12,
    }

    psf_args = {
        'psf_shape': (63, 31, 31),
        'psf_waist': (0.5, 0.25, 0.25),
        'psf_physical_size': 0.05,
        'psf_nphoton': 2,
    }

    positions = simulate_diffusion_pt(
        particle_types=particle_types, **simulation_args
    )

    scan_signal, scan_times, scan_positions = laser_scanning(
        positions,
        particle_brightness=_particle_brightness(particle_types),
        scanning_psf=psf_gaussian(**psf_args),
        **scanning_args,
    )

    pch, pch_bins = calculate_pch(scan_signal)

    plot_timeseries(scan_signal, title='Point FCS signal')
    plot_pch(pch, pch_bins)


# %time example_point_fcs()


# %% [markdown]
"""
### Camera

A digital widefield camera is approximated by counting photons for a certain
number of sampling intervals along all but two dimensions (usually y and x).
Common properties of digital cameras are detector shape (pixel resolution),
exposure time, binning of pixels, gain, offset, gamma, and bit depth.

### Photomultiplier

A photomultiplier simply integrates all photons for a certain number of
sampling intervals.

"""


# %%
def detector_pmt(
    intensities,
    pmt_exposure=1,
    pmt_offset=0.0,
    pmt_gain=1.0,
    pmt_gamma=1.0,
    pmt_bitdepth=16,
):
    """Return digitized time series of intensities in whole scene."""
    intensities = numpy.sum(intensities, axis=0)
    if pmt_exposure > 1:
        size = intensities.size
        if size % pmt_exposure:
            intensities.resize((size + pmt_exposure - size % pmt_exposure,))
        intensities = numpy.sum(
            intensities.reshape((-1, pmt_exposure)), axis=1
        )
    return _photon_counter(
        intensities, pmt_offset, pmt_gain, pmt_gamma, pmt_bitdepth
    )


def detector_camera(
    positions,
    intensities,
    camera_shape=(256, 256),
    camera_position=(0, 0),
    camera_exposure=1,
    camera_binning=(1, 1),
    camera_axes=(-2, -1),
    camera_offset=0.0,
    camera_gain=1.0,
    camera_gamma=1.0,
    camera_bitdepth=12,
):
    """Return digitized time series of 2D intensity images."""
    images = _detector_camera(
        positions,
        intensities,
        camera_shape,
        camera_position,
        camera_exposure,
        camera_binning,
        camera_axes,
    )
    return _photon_counter(
        images, camera_offset, camera_gain, camera_gamma, camera_bitdepth
    )


@numba.jit(nopython=True)
def _detector_camera(
    positions,
    intensities,
    shape=(256, 256),
    position=(0, 0),
    exposure=1,
    binning=(1, 1),
    axes=(-2, -1),
):
    """Return time series of 2D intensity images."""
    yax, xax = axes
    ymin = position[0] - shape[0] // 2
    ymax = position[0] + shape[0] // 2 + shape[0] % 2
    xmin = position[1] - shape[1] // 2
    xmax = position[1] + shape[1] // 2 + shape[1] % 2
    images = numpy.zeros(
        (
            positions.shape[1] // exposure,
            shape[0] // binning[0],
            shape[1] // binning[1],
        ),
        dtype=numpy.uint32,
    )
    for p in range(positions.shape[0]):
        for t in range(positions.shape[1]):
            y = positions[p, t, yax]
            x = positions[p, t, xax]
            if y >= ymin and y < ymax and x >= xmin and x < xmax:
                images[
                    t // exposure,
                    (y - ymin) // binning[0],
                    (x - xmin) // binning[1],
                ] += intensities[p, t]
    return images


def excitation_ambient(positions, particle_types, ambient_intensity=1.0):
    """Return intensities of fluorophores excited by ambient light."""
    intensities = numpy.empty(positions.shape[:2], dtype=numpy.float64)
    intensities[..., :] = ambient_intensity
    intensities *= _particle_brightness(particle_types)[..., numpy.newaxis]
    return intensities


def example_detector_types():
    """Compare detector types."""
    simulation_args = {
        'dimensions': 3,
        'duration': 2000,
        'sampling_period': 1000,
    }

    particle_types = [
        {
            'number_particles': 1000,
            'diffusion_speed': 10,
            'diffusion_model': diffusion_model_box_cyclic,
            'diffusion_model_args': {'box_shape': (20, 20, 20)},
            'positions_init': positions_init_uniform,
            'positions_init_args': {'init_shape': (20, 20, 20)},
            'particle_brightness': 4.0,
        },
        {
            'number_particles': 1000,
            'diffusion_speed': 20,
            'diffusion_model': diffusion_model_unconstrained,
            'diffusion_model_args': {},
            'positions_init': positions_init_uniform,
            'positions_init_args': {'init_shape': (20, 20, 20)},
            'particle_brightness': 1.0,
        },
    ]

    detector_pmt_args = {
        'pmt_exposure': 1,
        'pmt_offset': 0.0,
        'pmt_gain': 1.0,
        'pmt_gamma': 1.0,
        'pmt_bitdepth': 16,
    }

    detector_camera_args = {
        'camera_shape': (32, 32),
        'camera_position': (0, 0),
        'camera_exposure': 1,
        'camera_binning': (1, 1),
        'camera_axes': (-2, -1),
        'camera_offset': 0.0,
        'camera_gain': 1.0,
        'camera_gamma': 1.0,
        'camera_bitdepth': 12,
    }

    positions = simulate_diffusion_pt(
        particle_types=particle_types, **simulation_args
    )

    intensities = excitation_ambient(positions, particle_types)

    pmt_signal = detector_pmt(intensities, **detector_pmt_args)

    camera_images = detector_camera(
        positions, intensities, **detector_camera_args
    )

    plot_images(camera_images, title='Camera signal')
    plot_timeseries(pmt_signal, title='PMT signal')


# %time example_detector_types()


# %% [markdown]
"""
## To be continued

This notebook is work in progress.

The code is not well tested and documented. Use at your own risk.

"""


# %% [markdown]
"""
## System information

Print information about the software used to generate this document.

"""


# %%
def system_info():
    """Return information about Python and libraries."""
    import datetime
    import sys

    import ipywidgets
    import matplotlib
    import notebook
    import numba
    import numpy
    import widgetsnbextension

    return '\n'.join(
        (
            sys.executable,
            f'Python {sys.version}',
            '',
            f'numpy {numpy.__version__}',
            f'numba {numba.__version__}',
            f'matplotlib {matplotlib.__version__}',
            f'notebook {notebook.__version__}',
            f'ipywidgets {ipywidgets.__version__}',
            f'widgetsnbextension {widgetsnbextension.__version__}',
            '',
            str(datetime.datetime.now()),
        )
    )


print(system_info())
