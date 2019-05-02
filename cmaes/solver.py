# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group


"""
Example Usage:

# prob has f() and g() as member functions

solver = Solver(prob)
res = solver.solve()
x = res['x']
f = res['f']
"""
import cmaes.utils
import numpy as np


class Solver(object):
    def __init__(self, prob):
        self.prob = prob
        self.eval_counter = 0
        self.numerical_diff_step = 1e-4
        self.iter_values = list()
        self.verbose = True
        self.check_gradient = False

    def get_check_gradient(self, ):
        return self.check_gradient

    def set_check_gradient(self, check_gradient=True):
        self.check_gradient = check_gradient

    def eval_f(self, x):
        ret = self.prob.f(x)

        self.eval_counter += 1
        self.last_x = x
        self.last_f = ret
        self.iter_values.append(ret)
        if hasattr(self.prob, 'on_eval_f'):
            self.prob.on_eval_f(self)

        return ret

    def eval_g(self, x):
        if hasattr(self.prob, 'g'):
            ret = self.prob.g(x)
            if self.get_check_gradient():
                h = self.numerical_diff_step
                ret2 = utils.grad(self.prob.f, x, h)
                isgood = np.allclose(ret, ret2, atol=1e-05)
                if not isgood:
                    print(ret)
                    print(ret2)
                    print("diff = %.12f" % np.linalg.norm(ret - ret2))
                print('check_gradient... %s' % isgood)
        else:
            h = self.numerical_diff_step
            ret = utils.grad(self.prob.f, x, h)
        return ret

    def eval_c_eq_jac(self, x, i):
        if hasattr(self.prob, 'c_eq_jac'):
            ret = self.prob.c_eq_jac(x, i)
        else:
            h = self.numerical_diff_step

            def c_eq_f(x):
                return self.prob.c_eq(x, i)
            ret = utils.grad(c_eq_f, x, h)
        return ret

    def eval_c_ineq_jac(self, x, i):
        if hasattr(self.prob, 'c_ineq_jac'):
            ret1 = self.prob.c_ineq_jac(x, i)
            return ret1
        else:
            h = self.numerical_diff_step

            def c_ineq_f(x):
                return self.prob.c_ineq(x, i)
            ret2 = utils.grad(c_ineq_f, x, h)
            return ret2

    def collect_constraints(self):
        constraints = list()
        if hasattr(self.prob, 'num_eq_constraints'):
            num = self.prob.num_eq_constraints()
            if self.verbose:
                print('  [Solver]: num_eq_constraints = %d' % num)
            for i in range(num):
                c = dict()
                c['type'] = 'eq'
                assert(hasattr(self.prob, 'c_eq'))
                c['fun'] = self.prob.c_eq
                c['jac'] = self.eval_c_eq_jac
                c['args'] = [i]
                constraints.append(c)
        if hasattr(self.prob, 'num_ineq_constraints'):
            num = self.prob.num_ineq_constraints()
            if self.verbose:
                print('  [Solver]: num_ineq_constraints = %d' % num)
            for i in range(num):
                c = dict()
                c['type'] = 'ineq'
                assert(hasattr(self.prob, 'c_ineq'))
                c['fun'] = self.prob.c_ineq
                c['jac'] = self.eval_c_ineq_jac
                c['args'] = [i]
                constraints.append(c)
        if self.verbose:
            print('  [Solver]: num_constraints = %d' % len(constraints))
        return constraints

    def bounds(self):
        if hasattr(self.prob, 'bounds'):
            return self.prob.bounds()
        else:
            return None

    def solve(self, x0=None):
        pass

    def set_verbose(self, verbose):
        self.verbose = verbose

    def plot_convergence(self, filename=None):
        yy = self.iter_values
        xx = range(len(yy))
        import matplotlib.pyplot as plt
        # Plot
        plt.ioff()
        fig = plt.figure()
        fig.set_size_inches(18.5, 10.5)
        font = {'size': 28}
        plt.title('Value over # evaluations')
        plt.xlabel('X', fontdict=font)
        plt.ylabel('Y', fontdict=font)
        plt.plot(xx, yy)
        plt.axes().set_yscale('log')
        if filename is None:
            filename = 'plots/iter.png'
        plt.savefig(filename, bbox_inches='tight')
        plt.close(fig)
        print('plotting convergence OK.. ' + filename)

    def save_result(self, res, filename):
        with open(filename, 'w+') as fin:
            fin.write(str(res))
        print('writing result OK.. ' + filename)
