# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group
#
# adapted by Seungmoon Song <seungmoon.song@gmail.com>

from __future__ import division  # '/' always means non-truncating division
from cmaes.solver import Solver
import numpy as np
import cma
import scipy.optimize
import time
from datetime import datetime
import sys


class CMASolver(Solver):
    def __init__(self, prob):
        Solver.__init__(self, prob)
        opts = cma.CMAOptions()
        # for k, v in opts.iteritems():
        #     print k, v
        # exit(0)
        self.p_dir = 'optim_data/cma/'
        opts.set('verb_disp', 1)
        opts.set('popsize', 8)
        opts.set('verb_filenameprefix', self.p_dir)
        opts.set('maxiter', 2000)
        self.options = opts
        self.cen = None
        self.rng = None

    def set_verbose(self, verbose):
        self.verbose = verbose
        if verbose:
            self.options['verb_disp'] = 1
        else:
            self.options['verb_disp'] = 0

    def create_directory(self):
        verbose = (self.options['verb_disp'] > 0)
        import os
        path = self.p_dir
        if verbose:
            print('cma path = ', path)

        if not os.path.exists(path):
            if verbose:
                print('CMA-ES: create directory [%s]' % path)
            os.makedirs(path)

    def eval_f(self, y):
        x = self.unnormalize(y)
        ret = super(CMASolver, self).eval_f(x)

        # for i in range(self.prob.num_eq_constraints()):
        #     ret_eq_i = self.prob.c_eq(x, i)
        #     # ret += 100.0 * (ret_eq_i ** 2)
        #     ret += 10.0 * (ret_eq_i)  # Assume the quadratic form
        # for i in range(self.prob.num_ineq_constraints()):
        #     ret_ineq_i = self.prob.c_ineq(x, i)
        #     if ret_ineq_i < 0:
        #         ret += 100.0 * (ret_ineq_i ** 2)
        return ret

    def clip(self, x):
        if self.rng is None:
            return x
        return np.clip(x, self.cen-self.rng, self.cen+self.rng)

    # normalize between [-1, 1]
    def normalize(self, x):
        if self.rng is None:
            return x
        return (x - self.cen) / self.rng

    def unnormalize(self, y):
        if self.rng is None:
            return y
        x = self.cen + y * self.rng
        return x

    def solve(self, x0=None, sigma=1.0):
        verbose = (self.options['verb_disp'] > 0)
        begin = time.time()
        if verbose:
            print('Optimization method = CMA-ES')
        if x0 is None:
            if verbose:
                print('Optimization: set x0 as zeros')
            if self.cen is not None:
                x0 = self.cen
            else:
                x0 = np.zeros(self.prob.dim)
        self.create_directory()
        if verbose:
            print('CMA-ES: cen = ', self.cen)
            print('CMA-ES: rng = ', self.rng)
            print('Optimization begins at ', str(datetime.now()))
            #print('normalized_center = ', self.normalize(x0))
            # for k, v in self.options.iteritems():
            #     print(k, '\t', v)

        res = cma.fmin(self.eval_f,
                       self.normalize(x0),
                       sigma,
                       options=self.options)
        if verbose:
            print('Optimization ends at ', str(datetime.now()))
            print('Total times = %.2fs' % (time.time() - begin))

        ret = scipy.optimize.OptimizeResult()
        ret['y'] = res[0]
        ret['x'] = self.unnormalize(res[0])
        ret['fun'] = res[1]
        # assert(np.allclose(res[1], self.prob.f(ret['x'])))
        ret['nfev'] = self.eval_counter
        # ret['jac'] = self.eval_g(ret['x'])
        ret['message'] = 'Optimization terminated successfully.'
        ret['status'] = 0
        ret['success'] = True
        return ret


class CMASolverPar(CMASolver):
    def solve(self, x0=None, sigma=1.0):
        verbose = (self.options['verb_disp'] > 0)
        begin = time.time()
        if verbose:
            print('Optimization method = CMA-ES')
        if x0 is None:
            if verbose:
                print('Optimization: set x0 as zeros')
            if self.cen is not None:
                x0 = self.cen
            else:
                x0 = np.zeros(self.prob.dim)
        self.create_directory()
        if verbose:
            print('CMA-ES: cen = ', self.cen)
            print('CMA-ES: rng = ', self.rng)
            print('Optimization begins at ', str(datetime.now()))
            #print('normalized_center = ', self.normalize(x0))
            # for k, v in self.options.iteritems():
            #     print(k, '\t', v)

        res = cma.fmin(None,
                       self.normalize(x0),
                       sigma,
                       parallel_objective=self.eval_f,
                       options=self.options)
        if verbose:
            print('Optimization ends at ', str(datetime.now()))
            print('Total times = %.2fs' % (time.time() - begin))

        ret = scipy.optimize.OptimizeResult()
        ret['y'] = res[0]
        ret['x'] = self.unnormalize(res[0])
        ret['fun'] = res[1]
        # assert(np.allclose(res[1], self.prob.f(ret['x'])))
        ret['nfev'] = self.eval_counter
        # ret['jac'] = self.eval_g(ret['x'])
        ret['message'] = 'Optimization terminated successfully.'
        ret['status'] = 0
        ret['success'] = True
        return ret


if __name__ == '__main__':
    import optimization.test_problems
    import numpy as np
    # prob = test_problems.QuadProb()
    prob = optimization.test_problems.Rosen()
    x0 = np.random.rand(prob.dim) - 0.5

    solver = CMASolver(prob)
    res = solver.solve(x0)
    print(res)
