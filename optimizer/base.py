"""
.. inheritance-diagram:: dfo.optimize.base
    :parts: 1

**Base classes for optimization algorithms and plugins
(dfo subsystem name: OPT)**

Every optimization algorthm should be used in the following way.

1. Create the optimizer object.
2. Call the :meth:`reset` method of the object to set the initial point.
3. Optionally call the :meth:`check` method to check the consistency of
   settings.
4. Run the algorithm by calling the :meth:`run` method.

The same object can be reused by calling the :meth:`reset` method followed by
an optional call to :meth:`check` and a call to :meth:`run`.
"""

from misc.debug import DbgMsg, DbgMsgOut
from .cache import Cache

import numpy as np
from numpy import sqrt, isinf, Inf, where, zeros, array, concatenate
from numpy import random
from time import sleep
import sys

__all__ = ['Plugin', 'Reporter', 'Stopper', 'Annotator', 'AnnotatorGroup', 'Optimizer',
        'BoxConstrainedOptimizer', 'UCEvaluator', 'BCEvaluator', 'CEvaluator', 'normalizer', 'denormalizer']


class Plugin(object):
    """
    Base class for optimization algorithm plugins.
    """
    def __init__(self, quiet=False):
        # Stop flag
        self.stop = False

        # Should the reporter be silent
        self.quiet = quiet

        pass

    def reset(self):
        """
        Resets the plugin to its initial state
        """
        pass

    def setQuiet(self, quiet):
        # reset quiet
        self.quiet = quiet

    def __call__(self, x, ft, opt):
        pass


class Reporter(Plugin):
    def __init__(self, onImprovement=True, onIterStep=1):
        Plugin.__init__(self)
        self.onImprovement = onImprovement
        self.onIterStep = onIterStep

    def __call__(self, x, ft, opt):
        # We have improvement at first iteration or whenever optimizer says there is improvement
        if opt.f is None or opt.niter == opt.bestIter:
            improved = True
        else:
            improved = False

        # Report
        if not self.quiet:
            if (self.onImprovement and improved) or (self.onIterStep != 0 and opt.niter % self.onIterStep == 0):
                if type(ft) is tuple:
                    print("iter=" + str(opt.niter) + " f=" + str(ft[0]) + " h=" + str((ft[2]**2).sum()) + " fbest=" + str(opt.f))
                else:
                    print("iter=" + str(opt.niter) + " f=" + str(ft) + " fbest=" + str(opt.f))
                sys.stdout.flush()


class Stopper(Plugin):
    """
    Stopper plugins are used for stopping the optimization algorithm when a particular condition is satisfied
    """
    def __call__(self, x, ft, opt):
        pass


class Annotator(object):
    """
    Annotators produce annotations of the function value.
    """
    def produce(self):
        # Produce an annotation
        return None

    def consume(self, annotation):
        # Consume an annotation
        return


class AnnotatorGroup(object):
    """
    This object is a container holding annotators.
    """
    def __init__(self):
        self.annotators = []

    def add(self, annotator, index=-1):
        """
        Adds an annotator at position *index*.
        """
        if index < 0:
            self.annotators.append(annotator)
            return len(self.annotators) - 1
        else:
            self.annotators[index] = annotator
            return index

    def produce(self):
        """
        Produces a list of annotations corresponding to annotators.
        """
        annot = []
        for a in self.annotators:
            annot.append(a.produce())
        return annot

    def consume(self, annotations):
        """
        Consumes a list of annotations.
        """
        for ii in range(len(annotations)):
            self.annotators[ii].consume(annotations[ii])


def UCEvaluator(x, f, annGrp, nanBarrier):
    """
    Evaluator for unconstrained optimization.
    Returns the function value and the annotations.
    """
    fval = f(x)
    if nanBarrier and np.isnan(fval):
        fval = np.Inf

    return np.array(fval), annGrp.produce()


def BCEvaluator(x, xlo, xhi, f, annGrp, extremeBarrierBox, nanBarrier):
    """
    Evaluator for box-constrained optimization.
    Returns the function value and the annotations.
    """
    # Check box constraint violations
    violatedLo = (x < xlo).any()
    violatedHi = (x > xhi).any()

    # Enforce box constraints if extreme barrier approach is used
    if extremeBarrierBox:
        if violatedLo or violatedHi:
            return Inf
    else:
        if violatedLo:
            # print("x, xlo", x, xlo)
            raise Exception(DbgMsg("BCOPT", "Point violates lower bound."))
        if violatedHi:
            raise Exception(DbgMsg("BCOPT", "Point violates upper bound."))

    fval = f(x)

    if nanBarrier and np.isnan(fval):
        fval = np.Inf

    return np.array(fval), annGrp.produce()


def CEvaluator(x, xlo, xhi, f, fc, c, annGrp, extremeBarrierBox, nanBarrier):
    """
    Evaluator for constrained optimization.
    Returns the function value, the constraints values, and the annotations.
    """
    # Check box constraint violations
    violatedLo = (x < xlo).any()
    violatedHi = (x > xhi).any()

    # No function value
    # Enforce box constraints if extreme barrier approach is used
    if extremeBarrierBox:
        if violatedLo or violatedHi:
            fval = Inf
            cval = np.array([])
            return fval, cval, None
    else:
        if violatedLo:
            raise Exception(DbgMsg("BCOPT", "Point violates lower bound."))
        if violatedHi:
            raise Exception(DbgMsg("BCOPT", "Point violates upper bound."))

    # Do we have fc
    if fc is not None:
        # yes
        fval, cval = fc(x)
    else:
        # no
        if c is not None:
            cval = c(x)
        else:
            cval = np.array([])

            fval = f(x)

            if nanBarrier and np.isnan(fval):
                fval = np.Inf

    return np.array(fval), np.array(cval), annGrp.produce()


class Optimizer(object):
    """
    Base class for unconstrained optimization algorithms.
    """
    def __init__(self, function, debug=0, fstop=None, maxiter=None, nanBarrier=False, cache=False):
        # Function subject to optimization, must be picklable for parallel optimization methods.
        self.function = function
        # Debug mode flag
        self.debug = debug

        # Problem dimension
        self.ndim = None

        # Stopping conditions
        self.fstop = fstop
        self.maxiter = maxiter

        # NaN barrier
        self.nanBarrier = nanBarrier

        # Cache
        if cache:
            self.cache = Cache()
        else:
            self.cache = None

        # Iteration counter
        self.niter = 0

        # Best-yet point
        self.x = None
        self.f = None
        self.bestIter = None
        self.bestAnnotations = None

        # Plugins
        self.plugins = []

        # Annotator group
        self.annGrp = AnnotatorGroup()

        # Stop flag
        self.stop = False

        # Annotations produced at last cost function evaluation
        self.annotations = None

    def check(self):
        """
        Checks the optimization algorithm's settings and raises an exception
        if something is wrong.
        """
        if self.fun is None:
            raise Exception(DbgMsg("OPT", "Cost function not defined."))

        if self.maxiter is not None and self.maxiter < 1:
            raise Exception(DbgMsg("OPT", "Maximum number of iterations must be at least 1."))

    def installPlugin(self, plugin):
        """
        Installs a plugin object or an annotator in the plugins list and/or annotators list.
        """
        i1 = None
        if issubclass(type(plugin), Annotator):
            i1 = self.annGrp.add(plugin)
        i2 = None
        if issubclass(type(plugin), Plugin):
            self.plugins.append(plugin)
            i2 = len(self.plugins) - 1

        return (i1, i2)

    def getEvaluator(self, x):
        """
        Returns a tuple holding the evaluator function and its positional
        arguments that evaluate the problem at *x*.
        """
        return UCEvaluator, [x, self.function, self.annGrp, self.nanBarrier]

    def fun(self, x, count=True):
        """
        Evaluates the cost function at *x* (array).
        """
        data = None
        if self.cache is not None:
            data = self.cache.lookup(x)

        if data is not None:
            f, annot = data[0], data[1]
        else:
            # Evaluate
            evf, args = self.getEvaluator(x)
            f, annot = evf(*args)

        if count:
            self.newResult(x, f, annot)

        return np.array(f)

    def updateBest(self, x, f):
        """
        Updates best yet function value.
        """
        if self.f is None or self.f > f:
            self.f = f
            self.x = x
            self.bestIter = self.niter
            return True

        return False

    def newResult(self, x, f, annotations=None):
        """
        Registers the cost function value *f* obtained at point *x* with
        annotations list given by *annotations*.
        """
        self.niter += 1

        if self.cache and self.cache.lookup(x) is None:
            self.cache.insert(x, (f, annotations, self.niter))

        updated = self.updateBest(x, f)

        if annotations is not None:
            self.annotations = annotations
            self.annGrp.consume(annotations)

        if updated:
            self.bestAnnotations = self.annotations

        nplugins = len(self.plugins)
        for index in range(nplugins):
            plugin = self.plugins[index]
            if plugin is not None:
                stopBefore = self.stop
                plugin(x, f, self)
                if self.debug and self.stop and not stopBefore:
                    DbgMsgOut("OPT", "Run stopped by plugin object.")

        # Force stop condition on f<=fstop
        if self.fstop is not None and self.f <= self.fstop:
            self.stop = True
            if self.debug:
                DbgMsgOut("OPT", "Function fell below desired value. Stopping.")



        # Force stop condition on niter > maxiter
        if self.maxiter is not None and self.niter >= self.maxiter:
            self.stop = True
            if self.debug:
                DbgMsgOut("OPT", "Maximal number of iterations exceeded. Stopping.")

    def reset(self, x0):
        """
        Puts the optimizer in its initial state and sets the initial point to be the 1-dimensional array or list *x0*.
        """
        # Debug message
        if self.debug:
            DbgMsgOut("OPT", "Resetting.")

        # Determine dimension of the problem from initial point
        x0 = array(x0)
        self.ndim = x0.shape[0]

        if x0.ndim != 1:
            raise Exception(DbgMsg("OPT", "Initial point must be a vector."))

        # Store initial point
        self.x = x0.copy()
        self.f = None

        # Reset iteration counter
        self.niter = 0

        # Reset plugins
        for plugin in self.plugins:
            if plugin is not None:
                plugin.reset()

    def run(self):
        """
        Runs the optimization algorithm.
        """
        pass


def normalizer(x, origin, scale):
    return (x - origin) / scale


def denormalizer(y, origin, scale):
    return y * scale + origin


class BoxConstrainedOptimizer(Optimizer):
    """
    Box-constrained optimizer class
    """
    def __init__(self, function, xlo=None, xhi=None, debug=0, fstop=None, maxiter=None, nanBarrier=False, extremeBarrierBox=False, cache=False):
        Optimizer.__init__(self, function, debug, fstop, maxiter, nanBarrier, cache)
        self.xlo = xlo
        self.xhi = xhi

        self.extremeBarrierBox = False

    def check(self):
        """
        Checks the optimization algorithm's settings and raises an exception
        if something is wrong.
        """
        if self.xlo is not None:
            if (self.xlo.ndim != 1 or self.xhi.ndim != 1):
                raise Exception(DbgMsg("OPT", "Bounds must be one-dimensional vectors."))

        if self.xhi is not None:
            if (self.xlo.shape[0] != self.xhi.shape[0]):
                raise Exception(DbgMsg("OPT", "Bounds must match in length."))

        if (self.xlo is not None) and (self.xhi is not None):
            if (self.xlo >= self.xhi).any():
                raise Exception(DbgMsg("OPT", "Lower bound must be below upper bound."))

        Optimizer.check(self)

    def reset(self, x0):

        x0 = array(x0)
        Optimizer.reset(self, x0)

        if self.debug:
            DbgMsgOut("BCOPT", "Resetting.")

        # Set default bounds to Inf, -Inf
        if self.xlo is None:
            self.xlo = zeros(self.ndim)
            self.xlo.fill(-Inf)
        else:
            self.xlo = array(self.xlo)

        if self.xhi is None:
            self.xhi = zeros(self.ndim)
            self.xhi.fill(Inf)
        else:
            self.xhi = array(self.xhi)

        self.hasBounds = False
        if np.isfinite(self.xlo).any() or np.isfinite(self.xhi).any():
            self.hasBounds = True

        self.allBounds = False
        if np.isfinite(self.xlo).all() or np.isfinite(self.xhi).all():
            self.allBounds = True

        # Check initial point against bounds
        if (x0 < self.xlo).any():
            raise Exception(DbgMsg("BCOPT", "Initial point violates lower bound."))

        if (x0 > self.xhi).any():
            raise Exception(DbgMsg("BCOPT", "Initial point violates upper bound."))

        # Normalization (defaults to low-high range)
        self.normScale = self.xhi - self.xlo

        self.normOrigin = self.xlo.copy()

        ndx = where(isinf(self.xlo) & ~isinf(self.xhi))
        if len(ndx[0]) > 0:
            self.normScale[ndx] = 2 * (self.xhi[ndx] - x0[ndx])
            self.normOrigin[ndx] = x0[ndx] - self.normScale[ndx] / 2.0

        ndx = where(~isinf(self.xlo) & isinf(self.xhi))
        if len(ndx[0]) > 0:
            self.normScale[ndx] = 2 * (x0[ndx] - self.xlo[ndx])
            self.normOrigin[ndx] = self.xlo[ndx]

        ndx = where(isinf(self.xlo) & isinf(self.xhi))
        if len(ndx[0]) > 0:
            self.normScale[ndx] = 2.0
            self.normOrigin[ndx] = x0[ndx] - 1.0

    def bound(self, x):
        """
        Fixes components of *x* so that the bounds are enforced.
        """
        pos = where(x < self.xlo)[0]
        x[pos] = self.xlo[pos]

        pos = where(x > self.xhi)[0]
        x[pos] = self.xhi[pos]

    def normalize(self, x):
        """
        Returns a normalized point *y* corresponding to *x*.
        Components of *y* are

        .. math:: y^i = \\frac{x^i - n_o^i}{n_s^i}
        If both bounds are finite, the result is within the :math:`[0,1]`
        interval.
        """
        return normalizer(x, self.normOrigin, self.normScale)

    def denormalize(self, y):
        return denormalizer(y, self.normOrigin, self.normScale)

    def getEvaluator(self, x):
        return BCEvaluator, [x, self.xlo, self.xhi, self.function, self.annGrp, self.extremeBarrierBox, self.nanBarrier]

    def fun(self, x, count=True):
        data = None
        if self.cache is not None:
            data = self.cache.lookup(x)

        if data is not None:
            f = data[0]
        else:
            evf, args = self.getEvaluator(x)
            f = evf(*args)[0]
        if count:
            self.newResult(self.normalize(x), f)

        return np.array(f)
