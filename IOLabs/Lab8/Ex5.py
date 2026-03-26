from pyswarms.single import GlobalBestPSO
from pyswarms.utils.functions import single_obj

import numpy as np
import math

options = {"c1": 0.5, "c2": 0.3, "w": 0.9}
x_min = [1, 1]
x_max = [2, 2]
my_bounds = (x_min, x_max)

optimizer = GlobalBestPSO(n_particles=10, dimensions=2, options=options, bounds=my_bounds)
optimizer.optimize(single_obj.sphere, iters=1000)

# b) First results: best cost: 2.0207555088394162, best pos: [1.00168303 1.00865585]

task_x_min = np.zeros(6)
task_x_max = np.ones(6)
task_bounds = (task_x_min, task_x_max)


def task_endurance(params):
    (x, y, z, u, v, w) = params
    return math.exp(-2 * (y - math.sin(x)) ** 2) + math.sin(z * u) + math.cos(v * w)


def f(x):
    # particle =
    j = []
    np.array(j)

