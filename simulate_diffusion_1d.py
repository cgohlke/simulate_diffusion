# simulate_diffusion_1d.py

# Copyright (c) 2020-2025, Christoph Gohlke
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
"""
Some of the functions used in the first part of the simulation code:

* [`range(stop)`](https://docs.python.org/3/library/stdtypes.html#range)
  returns a sequence of integers from 0 to stop (excluding).

* [`print(objects)`](https://docs.python.org/3/library/functions.html#print)
  prints objects to the text stream file.

* [`numpy.zeros(shape, dtype)`](
  https://numpy.org/doc/stable/reference/generated/numpy.zeros.html)
  returns an array of specified shape and data type, initialized with zeros.

* [`numpy.random.randint(high, size)`](
  https://numpy.org/doc/stable/reference/random/generated/numpy.random.randint.html)
  returns an array of shape "size", initialized with random integers between
  0 and high (excluding).

* [`numpy.take(array, indices, axis)`](
  https://numpy.org/doc/stable/reference/generated/numpy.take.html)
  returns elements at indices from an array along an axis.

* [`numpy.cumsum(array, axis)`](
  https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html)
  returns the cumulative sum of an array along a specified axis.
  For example, `numpy.cumsum([[1, 2, 3], [4, 5, 6]], axis=1)` is
  `[[1, 3, 6], [4, 9, 15]]`.

* [`numpy.mean(array, axis)`](
  https://numpy.org/doc/stable/reference/generated/numpy.mean.html)
  returns the average of the elements along axis.

* [`numpy.arange(stop)`](
  https://numpy.org/doc/stable/reference/generated/numpy.arange.html)
  returns evenly spaced values up to stop.

* [`numpy.linalg.lstsq(a, b)`](
  https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html)
  returns the least-squares solution to a linear matrix equation, i.e. it
  computes the vector `x` that approximately solves the equation `a @ x = b`.

* [`array[..., numpy.newaxis]`](
  https://numpy.org/doc/stable/reference/constants.html#numpy.newaxis)
  appends a new dimension of size 1 to the array, e.g. to turn a vector into
  a matrix.

* [`array[index]`](https://numpy.org/doc/stable/user/basics.indexing.html)
  returns the value(s) of the array at the position(s) specified by index.
"""

# %% [markdown]
"""
## Diffusion on a one-dimensional grid
"""

# %%
# import the numpy array-computing library and initialize the random number
# generator
import numpy

rng = numpy.random.default_rng(12345678)

# Define some variables to control the simulation:
# number of particles
particles = 1000

# duration of the simulation, i.e. the number of sampling periods
duration = 1000

# the sampling period in arbitrary time units
sampling_period = 1000

# probability that a particle moves in a certain direction during the sampling
# period. Given in number of time units of the sampling period
diffusion_speed = 10

# Allocate a two-dimensional array of integers, which can store
# the positions of all particles during the simulation.
positions = numpy.zeros((particles, duration), dtype=numpy.int32)

# Create a look-up-table of directions to move.
# This table will be indexed by random numbers in range 0 to `sampling_period`.
# TODO: the table can be extended by another axis to include movements
# in y and z directions.
directions = numpy.zeros(sampling_period, dtype=numpy.int32)
# move the particle in the positive direction
directions[0:diffusion_speed] = 1
# move the particle in the negative direction
directions[diffusion_speed : diffusion_speed * 2] = -1

# Run the simulation separately for all particles.
# TODO: this loop could be vectorized, i.e. below calculations could be done
# for all particles at once.
for particle in range(particles):
    # Get a random number between 0 and `sampling_period`
    # for all sampling periods in the duration of the simulation.
    random_numbers = rng.integers(sampling_period, size=duration)

    # Index the first axis in the `directions` look-up-table with the random
    # numbers to obtain the relative moves of the particle for all sampling
    # periods.
    moves = numpy.take(directions, random_numbers, axis=0)

    # Set the first position of the particle to the origin
    # TODO: the initial position could be randomized.
    moves[0] = 0

    # Calculate all positions of the particle for the duration of the
    # simulation by cumulatively summing the relative moves.
    # The result is stored in the positions array.
    # TODO: to include obstacles in the simulation, the numpy.cumsum function
    # could be replaced by a custom function restricting movement in and
    # out of obstacles.
    positions[particle] = numpy.cumsum(moves, axis=0)


# Calculate the mean square displacement (MSD) at each sampling period
# by squaring all positions and averaging them over particles.
msd = numpy.mean(numpy.square(positions), axis=0)

# Calculate the diffusion coefficient D from the slope of the MSD values vs
# time. The slope is fitted by solving a linear equation system.
time = numpy.arange(duration)[..., numpy.newaxis]
slope = numpy.linalg.lstsq(time, msd, rcond=None)[0][0]
D = slope / 2

print(D * sampling_period)

# %% [markdown]
"""
### Plot the positions of some particles over time
"""

# %%
from matplotlib import pyplot

pyplot.figure()
pyplot.title('Selected particle positions')
pyplot.xlabel('time')
pyplot.ylabel('position')
for i in range(5):
    pyplot.plot(positions[i])
pyplot.show()

# %% [markdown]
"""
### Plot all particle positions as a color-coded image
"""

# %%
pyplot.figure()
pyplot.title('Particle positions')
pyplot.xlabel('time')
pyplot.ylabel('particle')
minmax = numpy.max(numpy.abs(positions))
pyplot.imshow(positions, cmap='seismic', vmin=-minmax, vmax=minmax)
pyplot.colorbar()
pyplot.show()

# %% [markdown]
"""
### Plot a histogram of particle positions at the end of the simulation
"""

# %%
pyplot.figure()
pyplot.title('Histogram of last particle position')
pyplot.xlabel('position')
pyplot.ylabel('frequency')
minmax = numpy.max(numpy.abs(positions))
pyplot.hist(positions[:, -1], numpy.arange(-minmax - 0.5, minmax + 0.5))
pyplot.show()

# %% [markdown]
"""
### Plot MSD and line fit
"""

# %%
pyplot.figure()
pyplot.title('MSD and linear fit')
pyplot.xlabel('time')
pyplot.ylabel('MSD')
pyplot.plot(time, msd, '.', label='simulation')
pyplot.plot(time, slope * time, '-', lw=3, label='fit')
pyplot.legend()
pyplot.show()

# %%
"""
Remove all names defined above from the global namespace.
Global names might interfere with following code.
"""

# %%
# %reset -f
