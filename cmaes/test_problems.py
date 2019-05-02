# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group


import numpy as np


class QuadProb(object):
    def __init__(self, x=None):
        self.dim = 4
        self.x = np.array([0.5, 0.7, -0.5, 1.0])
        if x is not None:
            self.dim = len(x)
            self.x = x

    def f(self, x):
        diff = x - self.x
        return 0.5 * np.linalg.norm(diff) ** 2


class Rosen(object):
    def __init__(self):
        self.dim = 10

    def f(self, x):
        import scipy.optimize
        return scipy.optimize.rosen(x)

    def bounds(self):
        return [(-0.5, 0.5)] * self.dim

    def on_eval_f(self, solver):
        counter = solver.eval_counter
        value = solver.last_f
        # if counter % 10 == 0:
        #     solver.plot_convergence()


class ConstrainedQuadProb(object):
    def __init__(self, x=None):
        self.dim = 4
        self.x = x if x is not None else np.array([0.5, 0.7, -0.5, 1.0])

    def f(self, x):
        diff = x - self.x
        return 0.5 * np.linalg.norm(diff) ** 2

    def num_eq_constraints(self):
        return 2

    def c_eq(self, x, index):
        if index == 0:
            return x[0] + x[1] - 1.0
        elif index == 1:
            return x[1] + x[2] - 1.0

    def c_eq_jac(self, x, index):
        """ (optional) """
        if index == 0:
            return np.array([1.0, 1.0, 0.0, 0.0])
        elif index == 1:
            return np.array([0.0, 1.0, 1.0, 0.0])

    def num_ineq_constraints(self):
        return 1

    def c_ineq(self, x, index):
        if index == 0:
            return x[3] - 1.2


if __name__ == '__main__':
    print('test optimization problems')
    prob = QuadProb(np.array([0.5, -0.3, 0.9]))
    print('prob.dim = %d' % (prob.dim))
    from optimization.solver_cma import CMASolver as Solver
    solver = Solver(prob)
    res = solver.solve()
    print(res)
