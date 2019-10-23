"""
inheritance-diagram:: dfo.optimizer.direct
    :parts: 1
"""


from misc.debug import DbgMsgOut, DbgMsg
from .base import BoxConstrainedOptimizer
from numpy import max, min, abs, array
import numpy as np
import heapq

__all__ = ['Cube', 'DIRECT']


class Cube(object):
    def __init__(self, x, f, depth):
        self.x = array(x)
        self.f = f
        self.ndim = self.x.shape[0]
        self.depth = depth

    def increase_depth(self, i=None):
        if i is not None:
            self.depth[i] += 1
        else:
            for i in range(self.ndim):
                self.depth[i] += 1


class DIRECT(BoxConstrainedOptimizer):
    def __init__(self, function, xlo=None, xhi=None, debug=0, fstop=None, maxiter=None):

        BoxConstrainedOptimizer.__init__(self, function, xlo, xhi, debug, fstop, maxiter, cache=True)

        self.pq_cache = None
        self.eps = 1e-2
        self.K = 0
        self.max_depth = 5
        self.visited = []

    def check(self):
        """
        Checks the optimization algorithm's settings and raises an exception if
        something is wrong.
        """
        BoxConstrainedOptimizer.check(self)

        # if self.samplesize is None:
        #     raise Exception(DbgMsg("DIRECT", "The sample size should not be None."))

    def reset(self, x0):
        """
        Puts the optimizer in its initial state and sets the initial point to
        be the 1-dimensional array *x0*. The length of the array becomes the
        dimension of the optimization problem (:attr:`ndim` member). The shape
        of *x* must match that of *xlo* and *xhi*.
        """
        BoxConstrainedOptimizer.reset(self, x0)
        # Debug message
        if self.debug:
            DbgMsgOut("DIRECT", "Resetting DIRECT")

    def run(self):
        """
        Run the DIRECT algorithm.
        """
        # Debug message
        if self.debug:
            DbgMsgOut("CSOPT", "Starting a coordinate search run at i=" + str(self.niter))

        # Reset stop flag
        self.stop = False

        # Check
        self.check()

        self.x = 0.5 * np.ones(shape=(self.ndim,))
        self.f = self.fun(self.denormalize(self.x))

        # pq: potentiality, f(center), depth
        self.pq_cache = [(self.f - 1.0 / 3.0 * self.K, 1, Cube(self.x, self.f, np.ones(shape=(self.ndim,))))]

        while not self.stop and self.pq_cache:
            val, it, cube = heapq.heappop(self.pq_cache)
            self.update_cube(cube)
            x, depth = cube.x, cube.depth
            minimum_depth = min(depth)
            # print("depth: ", depth)
            if self.debug:
                DbgMsgOut("DIRECT", "Cube.f =" + str(cube.f))

            inc_index, better_index, same_index, worse_index = [], [], [], []
            for i in range(self.ndim):
                # try points with length of the maximum side of hyper-rectangle
                if depth[i] == minimum_depth:
                    x[i] -= (1 / 3)**depth[i]
                    improved = self.update_potential_rectangle(x, depth, i)
                    if improved == 0:
                        same_index.append(i)
                    elif improved > 0:
                        better_index.append(i)
                    else:
                        worse_index.append(i)
                    x[i] += 2 * (1 / 3)**depth[i]
                    improved = self.update_potential_rectangle(x, depth, i)
                    if improved == 0:
                        same_index.append(i)
                    elif improved > 0:
                        better_index.append(i)
                    else:
                        worse_index.append(i)
                    x[i] -= (1 / 3) ** depth[i]
                    inc_index.append(i)

            if better_index != [] and worse_index != []:
                # Decrease the size of the cube and save it in the cache
                for idx in inc_index:
                    cube.increase_depth(idx)
                self.niter += 1

                # Push the smaller cube into the cache centering at self.x
                heapq.heappush(self.pq_cache, (cube.f - 0.5**depth[0] * self.K, self.niter, cube))

            if self.debug:
                DbgMsgOut("DIRECT", "Iteration i=" + str(self.niter) + " fbest=" + str(self.f))

    def update_cube(self, cube):
        '''
        :param cube: class Cube object
        :return: None
        '''
        # print("update cube")
        x = cube.x
        depth = cube.depth
        for i in range(self.ndim):
            if self.cache.isVisited(x + (1.0 / 3.0)**depth[i]) or self.cache.isVisited(x - (1.0 / 3.0)**depth[i]):
                cube.increase_depth(i)
        # print("cube's depth: ", cube.depth)

    def update_potential_rectangle(self, x, depth, i):
        '''
        Check potentially Potential Hyper-rectangles.
        :param x:
        :param depth:
        :param i:
        :return: updated or not
        '''
        # print("x::::::::::", x)
        f = self.fun(self.denormalize(x))
        if depth[i] <= self.max_depth and f <= self.f:

            # build new cube with x_new, depth_new
            x_new = x.copy()
            depth_new = depth.copy()
            depth_new[i] += 1
            cube = Cube(x_new, f, depth_new)
            heapq.heappush(self.pq_cache, (f - self.K * 0.5**depth_new[i], self.niter, cube))

        if f < self.f:
            self.f = f
            self.x = x.copy()
            if self.debug:
                DbgMsgOut("DIRECT", "Better centers found in iteration i=" + str(self.niter) + " fbest=" + str(self.f))
            return 1
        elif f == self.f:
            return 0
        else:
            return -1
