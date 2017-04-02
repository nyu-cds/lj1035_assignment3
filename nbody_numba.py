"""
    N-body simulation.

    This version combines all optimizations.
    Original runtime: 128.36904295100067 seconds
    Optimized runtime: 41.05443487100001 seconds
    Relative speedup: 3.13x

    Optimized runtime with Numba: 2.184466565006005 seconds
"""

import timeit
import numpy as np
from numba import jit, int32, float64, void, int64, vectorize
from itertools import combinations

PI = 3.14159265358979323
SOLAR_MASS = 4 * PI * PI
DAYS_PER_YEAR = 365.24

BODIES_array = np.array([
    ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [SOLAR_MASS, 0.0, 0.0]),

    ([4.84143144246472090e+00,
                 -1.16032004402742839e+00,
                 -1.03622044471123109e-01],
                [1.66007664274403694e-03 * DAYS_PER_YEAR,
                 7.69901118419740425e-03 * DAYS_PER_YEAR,
                 -6.90460016972063023e-05 * DAYS_PER_YEAR],
                [9.54791938424326609e-04 * SOLAR_MASS, 0.0, 0.0]),

    ([8.34336671824457987e+00,
                4.12479856412430479e+00,
                -4.03523417114321381e-01],
               [-2.76742510726862411e-03 * DAYS_PER_YEAR,
                4.99852801234917238e-03 * DAYS_PER_YEAR,
                2.30417297573763929e-05 * DAYS_PER_YEAR],
               [2.85885980666130812e-04 * SOLAR_MASS, 0.0, 0.0]),

    ([1.28943695621391310e+01,
                -1.51111514016986312e+01,
                -2.23307578892655734e-01],
               [2.96460137564761618e-03 * DAYS_PER_YEAR,
                2.37847173959480950e-03 * DAYS_PER_YEAR,
                -2.96589568540237556e-05 * DAYS_PER_YEAR],
               [4.36624404335156298e-05 * SOLAR_MASS, 0.0, 0.0]),

    ([1.53796971148509165e+01,
                 -2.59193146099879641e+01,
                 1.79258772950371181e-01],
                [2.68067772490389322e-03 * DAYS_PER_YEAR,
                 1.62824170038242295e-03 * DAYS_PER_YEAR,
                 -9.51592254519715870e-05 * DAYS_PER_YEAR],
                [5.15138902046611451e-05 * SOLAR_MASS, 0.0, 0.0])])

BODIES_keys = np.array([0, 1, 2, 3, 4])
BODIES_name = {'sun': 0, 'jupiter': 1, 'saturn': 2, 'uranus': 3, 'neptune': 4}


@vectorize([float64(float64, float64)])
def vec_deltas(arr1, arr2):
    return arr1 - arr2

@jit('void(float64[:], float64[:], float64, float64, float64, float64, float64, float64)', nopython=True)
def update_vs(v1, v2, dt, dx, dy, dz, m1, m2):
    '''
        update velocity
    '''
    mag = dt * ((dx * dx + dy * dy + dz * dz) ** (-1.5))
    b1 = m1 * mag
    b2 = m2 * mag

    v1[0] -= dx * b2
    v1[1] -= dy * b2
    v1[2] -= dz * b2
    v2[0] += dx * b1
    v2[1] += dy * b1
    v2[2] += dz * b1


@jit('void(float64[:,:,:], int64[:], int64[:,:], int64, float64)', nopython=True)
def advance(BODIES_array, BODIES_keys, bodies_keys_pairs, iterations, dt):
    '''
        advance the system one timestep
    '''
    for i in range(iterations):
        for index in range(len(bodies_keys_pairs)):
            (body1, body2) = bodies_keys_pairs[index]
            p1 = BODIES_array[body1][0]
            v1 = BODIES_array[body1][1]
            m1 = BODIES_array[body1][2]
            p2 = BODIES_array[body2][0]
            v2 = BODIES_array[body2][1]
            m2 = BODIES_array[body2][2]
            (dx, dy, dz) = vec_deltas(p1, p2)
            update_vs(v1, v2, dt, dx, dy, dz, m1[0], m2[0])

        for body in BODIES_keys:
            r = BODIES_array[body][0]
            vx = BODIES_array[body][1][0]
            vy = BODIES_array[body][1][1]
            vz = BODIES_array[body][1][2]
            r[0] += dt * vx
            r[1] += dt * vy
            r[2] += dt * vz


@jit('float64(float64[:,:,:], int64[:], int64[:,:], float64)', nopython=True)
def report_energy(BODIES_array, BODIES_keys, bodies_keys_pairs, e=0.0):
    '''
        compute the energy and return it so that it can be printed
    '''
    for index in range(len(bodies_keys_pairs)):
        (body1, body2) = bodies_keys_pairs[index]
        p1 = BODIES_array[body1][0]
        m1 = BODIES_array[body1][2]
        p2 = BODIES_array[body2][0]
        m2 = BODIES_array[body2][2]
        (dx, dy, dz) = vec_deltas(p1, p2)
        e -= (m1[0] * m2[0]) / ((dx * dx + dy * dy + dz * dz) ** 0.5)

    for body in BODIES_keys:
        vx = BODIES_array[body][1][0]
        vy = BODIES_array[body][1][1]
        vz = BODIES_array[body][1][2]
        m = BODIES_array[body][2]
        e += m[0] * (vx * vx + vy * vy + vz * vz) / 2.

    return e


@jit('void(float64[:,:,:], int32[:], float64[:], float64, float64, float64)')
def offset_momentum(BODIES_array, BODIES_keys, ref, px=0.0, py=0.0, pz=0.0):
    '''
        ref is the body in the center of the system
        offset values from this reference
    '''

    for body in BODIES_keys:
        vx = BODIES_array[body][1][0]
        vy = BODIES_array[body][1][1]
        vz = BODIES_array[body][1][2]
        m = BODIES_array[body][2]
        px -= vx * m[0]
        py -= vy * m[0]
        pz -= vz * m[0]

    v = ref[1]
    m = ref[2]
    v[0] = px / m[0]
    v[1] = py / m[0]
    v[2] = pz / m[0]


@jit('void(int32, int32, int32)')
def nbody(loops, reference, iterations):
    '''
        nbody simulation
        loops - number of loops to run
        reference - body at center of system
        iterations - number of timesteps to advance
    '''
    # Set up global state
    bodies_keys_pairs = np.array(list(combinations(BODIES_keys, 2)))

    offset_momentum(BODIES_array, BODIES_keys, BODIES_array[reference], 0.0, 0.0, 0.0)

    for _ in range(loops):
        advance(BODIES_array, BODIES_keys, bodies_keys_pairs, iterations, 0.01)
        print(report_energy(BODIES_array, BODIES_keys, bodies_keys_pairs, 0.0))


if __name__ == '__main__':
    # Calculate time to run
    print(timeit.timeit("nbody(100, BODIES_name['sun'], 20000)", setup="from __main__ import nbody, BODIES_name", number=1))

