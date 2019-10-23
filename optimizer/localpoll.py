"""
minimize stepwise function
written by Hongsheng Liu
UNC Chapel Hill and IBM Watson Research center
"""


import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize.optimize import _status_message

_MACHEPS = np.finfo(np.float64).eps


def local_search_step_function(func, bounds, x0, args=(), delta=None, theta=0.01, phi=1.5,
                               maxiter=100, patience=20, disp=False, seed=2019):
    """
    :param func: callable
        The objective function to be minimized.  Must be in the form
        ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
        and ``args`` is a  tuple of any additional fixed parameters needed to
        completely specify the function.

    :param bounds: sequence
        Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
        defining the lower and upper bounds for the optimizing argument of
        `func`. It is required to have ``len(bounds) == len(x)``.
        ``len(bounds)`` is used to determine the number of parameters in ``x``.

    :param x0: np.array,
        initialization derived by importance sampling

    :param args: tuple, optional
        Any additional fixed parameters needed to
        completely specify the objective function.

    :param delta: np.array,
        step length in each direction

    :param theta: float,
        shrink control, theta < 1.0

    :param phi: float,
        expansion control, phi >= 1.0

    :param maxiter: int,
        maximum number of iterations

    :param patience: int,
        Stop algorithm when a monitored quantity has stopped improving in the past patience iterations.

    :param disp: bool, optional
        Display status messages and current minimum

    :param  seed: int,
        seed for random number generator
    :return:
    """

    solver = LocalGPSSolver(func, bounds, x0=x0, args=args, delta=delta, theta=theta, phi=phi,
                            maxiter=maxiter, patience=patience, disp=disp, seed=seed)

    return solver.solve()


class LocalGPSSolver(object):
    def __init__(self, func, bounds, x0, args=(), delta=None, theta=0.01, phi=1.5,
                 maxiter=100, patience=30, disp=False, seed=2019):

        """
        :param func: callable
            The objective function to be minimized.  Must be in the form
            ``f(x, *args)``, where ``x`` is the argument in the form of a 1-D array
            and ``args`` is a  tuple of any additional fixed parameters needed to
            completely specify the function.

        :param bounds: sequence
            Bounds for variables.  ``(min, max)`` pairs for each element in ``x``,
            defining the lower and upper bounds for the optimizing argument of
            `func`. It is required to have ``len(bounds) == len(x)``.
            ``len(bounds)`` is used to determine the number of parameters in ``x``.

        :param x0: np.array,
            initialization derived by global search techniques or random sampling

        :param args: tuple, optional
            Any additional fixed parameters needed to
            completely specify the objective function.

        :param delta: np.array,
            The step length in each direction

        :param theta: float,
            The shrink parameter, theta < 1.0

        :param phi: float,
            The expansion parameter, phi > 1.0

        :param maxiter: int, optional
            The maximum number of iterations

        :param disp: bool, optional
            Display status messages and current minimum

        :param seed: int or `np.random.RandomState`, optional

        """

        self.func = func
        self.args = args

        # convert tuple of lower and upper bounds to limits
        # [(low_0, high_0), ..., (low_n, high_n]
        #     -> [[low_0, ..., low_n], [high_0, ..., high_n]]
        self.limits = np.array(bounds, dtype='float').T
        self.parameter_count = np.size(self.limits, 1)

        if (np.size(self.limits, 0) != 2
                or not np.all(np.isfinite(self.limits))):
            raise ValueError('bounds should be a sequence containing '
                             'real valued (min, max) pairs for each value'
                             ' in x')

        self.__scale_arg1 = 0.5 * (self.limits[0] + self.limits[1])
        self.__scale_arg2 = np.fabs(self.limits[0] - self.limits[1])

        self.population = self._unscale_parameters(x0)
        self.min_val = self.func(x0, *self.args)

        if delta is None:
            self.delta_lb, self.delta_ub = 0.01, 0.1
        else:
            self.delta_lb, self.delta_ub = delta

        self.theta = theta

        self.phi = phi

        self.patience = patience

        self.maxiter = maxiter

        self.rng = np.random.RandomState(seed)

        self.disp = disp

    def solve(self):
        status_message = _status_message['success']
        nfev, nit, lag, warning_flag = 0, 0, 0, False

        delta = self.delta_lb

        for nit in range(1, self.maxiter + 1):
            trial = np.copy(self.population)
            prev_min_val = self.min_val
            better_candidate = None  # store better candidate
            equal_candidates = []  # store candidates with equal function values
            # print("Current value: ", self.min_val)

            for idx in range(self.parameter_count):

                trial[idx] = np.min([self.population[idx] + delta, 1])
                func_eval_right = self.func(self._scale_parameters(trial), *self.args)
                trial[idx] = np.max([self.population[idx] - delta, 0])
                func_eval_left = self.func(self._scale_parameters(trial), *self.args)
                nfev += 2

                if better_candidate is None:
                    if func_eval_right == prev_min_val:
                        equal_candidates.append((idx, 1))
                    if func_eval_left == prev_min_val:
                        equal_candidates.append((idx, -1))

                if func_eval_right < self.min_val:
                    self.min_val = func_eval_right
                    better_candidate = (idx, 1)
                if func_eval_left < self.min_val:
                    self.min_val = func_eval_left
                    better_candidate = (idx, -1)
                trial[idx] = self.population[idx]

            if better_candidate is None:
                len_equal_candidates = len(equal_candidates)
                if len_equal_candidates > 0:
                    idx_equal_candidate = np.random.randint(0, len_equal_candidates)
                    idx, sign = equal_candidates[idx_equal_candidate]
                    self.population[idx] = np.clip(self.population[idx] + sign * delta, 0.0, 1.0)

                delta = np.min([self.phi * delta, self.delta_ub])
                lag += 1

                if lag >= self.patience:
                    if self.disp:
                        print("No improvement in the past {} iterations.".format(self.patience))

                    final_result = OptimizeResult(
                        x=self._scale_parameters(self.population),
                        fun=self.min_val,
                        nfev=nfev,
                        nit=nit,
                        message=status_message,
                        success=(warning_flag is True))
                    return final_result
            else:
                idx, sign = better_candidate
                self.population[idx] = np.clip(self.population[idx] + sign * delta, 0.0, 1.0)
                delta = np.max([self.theta * delta,  self.delta_lb])
                lag = 0

        if nit == self.maxiter:
            status_message = _status_message['maxiter']
            warning_flag = True

        final_result = OptimizeResult(
            x=self._scale_parameters(self.population),
            fun=self.min_val,
            nfev=nfev,
            nit=nit,
            message=status_message,
            success=(warning_flag is True))
        return final_result

    def _scale_parameters(self, parameters, feat=None):
        """
        scale from a number between 0 and 1 to parameters
        """

        if feat is None:
            return self.__scale_arg1 + (parameters - 0.5) * self.__scale_arg2
        else:
            return self.__scale_arg1[feat] + (parameters - 0.5) * self.__scale_arg2[feat]

    def _unscale_parameters(self, parameters):
        """
        scale from parameters to a number between 0 and 1.
        """
        return (parameters - self.__scale_arg1) / self.__scale_arg2 + 0.5

