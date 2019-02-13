#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import division


import numpy as np
from numpy import sqrt, pi, exp
from scipy.integrate import quad


def gyration(t, t0, D_local):
    """Calculate the gyration (radius) of the Gaussian concentration plume.

    Args:
        t (float): time
        t0 (float or array): initial dilation time
        D_local (float or array): local dispersion coefficient

    Returns:
        gyration (float or array)

    Examples:
        >>> print(np.around(gyration(100, 10, 0.01), 4))
        1.4832
        >>> print(np.around(gyration(100, 10, (0.01, 0.01)), 4))
        [1.4832 1.4832]
        >>> print(np.around(gyration(100, (10, 100), (0.01, 0.01)), 4))
        [1.4832 2.    ]
        >>> print(np.around(gyration(100, 10, (0.01, 0.01, 0.01)), 4))
        [1.4832 1.4832 1.4832]
        >>> print(np.around(gyration(100, 0, 0.01), 4))
        1.4142
        >>> print(np.around(gyration(100, 10, 0), 4))
        0.0
    """
    t = np.array(t)
    t0 = np.array(t0)
    D_local = np.array(D_local)
    return np.sqrt(2.*D_local*(t+t0))

def initial_dilation_time(R0, D_local):
    """Calculate the initial dilation time from a given gyration.

    Args:
        R0 (float or array): gyration
        D_local (float or array): local dispersion coefficient

    Returns:
        initial dilation time (float or array)

    Examples:
        >>> print(np.around(initial_dilation_time(gyration(0, 10, 0.01),
        ... 0.01), 4))
        10.0
        >>> print(np.around(initial_dilation_time(
        ... gyration(0, 10, (0.01, 0.01)), 0.01), 4))
        [10. 10.]
    """
    R0 = np.array(R0)
    D_local = np.array(D_local)
    return R0**2 / (2.*D_local)


class Concentration(object):
    """Calculate analytical solutions for mean concentration and
       mean concentration variance.

       See Schüler et al. 2016, AWR for details, derivations, and assumptions
       https://doi.org/10.1016/j.advwatres.2016.06.012
    """
    def __init__(self, dimension, u_mean, D_local, lambda_Y, sigma_2_Y, t0,
                 trans=True):
        """Initialiser

        Args:
            dimension (int): dimension
            u_mean (float): mean velocity in x-direction
            D_local (float or array): local dispersion coefficient
            lambda_Y (float or array): correlation length of log K field
            sigma_2_Y (float): variance of log K field
            t0 (float or array): initial spreading time to avoid unnatural
                                 delta distributions at t=0
                                 (t0 > 0 needed for concentration variance)
            trans (bool): shall the transversal disp. coeff. be upscaled or
                          not, the effect is very small
        Examples:
            >>> C1d = Concentration(1, 1, 0.01, 1, 0.1, 10)
            >>> C2d = Concentration(2, [1, 0], [0.01, 0.01], [1, 1], 0.1, \
                                    [10, 10])
            >>> C3d = Concentration(3, [1, 0, 0], [0.01, 0.01, 0.01], \
                                    [1, 1, 1], 0.1, [10, 10, 10])
            >>> #mean concentration at a single point in space and time
            >>> t = 1000
            >>> x = 1000
            >>> y = 0
            >>> z = 0
            >>> print(np.around(C1d.mean(t, x), 4))
            0.0245
            >>> print(np.around(C2d.mean(t, x, y), 4))
            0.002
            >>> print(np.around(C3d.mean(t, x, y, z), 5))
            0.00018
            >>> #mean concentration at a single point in time over space axis
            >>> t = 1000
            >>> x = np.linspace(900, 1100, 100)
            >>> y = np.linspace(-40, 40, 20)
            >>> z = np.linspace(-40, 40, 20)
            >>> print(np.around(C1d.mean(t, x)[50], 4))
            0.0245
            >>> print(np.around(C2d.mean(t, x, y)[50,10], 4))
            0.0018
            >>> print(np.around(C3d.mean(t, x, y, z)[50,10,10], 5))
            0.00015
            >>> #mean concentration over space and time axis
            >>> t = np.linspace(0, 1000, 10)
            >>> x = np.linspace(900, 1100, 100)
            >>> y = np.linspace(-40, 40, 20)
            >>> z = np.linspace(-40, 40, 20)
            >>> print(np.around(C1d.mean(t, x)[-1,50], 4))
            0.0245
            >>> print(np.around(C2d.mean(t, x, y)[-1,50,10], 4))
            0.0018

        """
        self.dimension = dimension
        self.u_mean = np.atleast_1d(u_mean)
        self.sigma_2_Y = sigma_2_Y
        self.gamma_2 = (1 + sigma_2_Y / 6)**2
        self.trans = trans
        if dimension == 1:
            self.D_local = np.atleast_1d(D_local)
            self.lambda_Y = lambda_Y
            self.t0 = np.atleast_1d(t0)
            self.u_mean_0 = self.u_mean
        else:
            #make sure that these parameters are arrays
            self.u_mean = np.array(self.u_mean)
            self._check_len(self.u_mean, 'u_mean')
            self.u_mean_0 = self.u_mean[0]
            self.D_local = np.array(D_local)
            self._check_len(self.D_local, 'D_local')
            self.lambda_Y = np.array(lambda_Y)
            self._check_len(self.lambda_Y, 'lambda_Y')
            self.t0 = np.array(t0)
            self._check_len(self.t0, 't0')

        self._set_D_ens()

        #dispersive time scale
        self.tau_D = self.lambda_Y**2 / self.D_local

        #the different mixing models in a look up dict
        self.model_fct = {'IEM': self.Chi_IEM,
                          'TDIEM': self.Chi_TDIEM,
                          'local': self.Chi_local}
        #standard mixing model
        self.model = 'TDIEM'

    def _set_D_ens(self):
        if self.dimension == 1:
            self.dD_ens = np.atleast_1d(self.sigma_2_Y * self.u_mean_0 *
                                        self.lambda_Y * sqrt(pi/2) /
                                        self.gamma_2)
        else:
            self.dD_ens = np.zeros(self.dimension)
            #Gelhar and Axness 1983, eq. (33), (36), and (37)
            #with additional factor sqrt(pi/2) for correct corr. fct.
            self.dD_ens[0] = (self.D_local[0] + self.sigma_2_Y * self.u_mean_0 *
                             self.lambda_Y[0] * sqrt(pi/2) / self.gamma_2)

            if self.trans:
                if self.dimension == 2:
                    self.dD_ens[1] = (self.sigma_2_Y * self.u_mean_0 *
                                      self.D_local[0] / (8 * self.gamma_2) *
                                     (1 + 3 * self.D_local[1] / self.D_local[0]))
                if self.dimension == 3:
                    self.dD_ens[1] = (self.sigma_2_Y * self.u_mean_0 *
                                      self.D_local[0] / (15 * self.gamma_2) *
                                     (1 + 4 * self.D_local[1] / self.D_local[0]))
                    self.dD_ens[2] = (self.sigma_2_Y * self.u_mean_0 *
                                      self.D_local[0] / (15 * self.gamma_2) *
                                     (1 + 4 * self.D_local[1] / self.D_local[0]))
        self.D_ens = np.empty_like(self.D_local)
        self.D_ens = self.D_local + self.dD_ens

    def _check_len(self, array, name):
        if array.size != self.dimension:
            raise ValueError('Error: len({0}) != dimension of {1}'.
                             format(name, self.dimension))

    @property
    def model(self):
        return self.__model
    @model.setter
    def model(self, model):
        if not model in self.model_fct.keys():
            raise ValueError('Model {0} is unknown'.format(model))
        self.__model = model

    def Chi_IEM(self, t):
        """"The IEM mixing model from turbulence theory.

        For details see Colucci et al. 1998, Phys. Fluids.
        """
        return np.sum(self.D_ens / self.lambda_Y**2)

    def Chi_local(self, t):
        """"The IEM mixing model from hydrogeology.

        For details see Kapoor et al. 1998, WRR.
        """
        return np.sum(self.D_local / self.lambda_Y**2)

    def Chi_TDIEM(self, t):
        """The time dependent mixing model.

        For details see Schüler et al. 2016, AWR.
        """
        d = self.dimension
        return np.sum(self.D_eff(t, d) / (self.lambda_Y)**2 *
                      self.tau_D[0]/t)

    def D_eff(self, t, dimension):
        """Calculate effective dispersion coefficient for large Pe.

        Args:
            t (float, array): time
            dimension (int): space dimension
        """
        if dimension < 2:
            raise ValueError('Effective dispersion coefficient not defined '
                             'for spatial dimensions less than 2.')
        return (self.D_ens-self.D_local) * (1 - (1 + 4 * t / self.tau_D)**
                                           (-(dimension-1)/2.))

    def __reshape_axis(self, d, i, array):
        """Reshape the axis to a given dimension in order to span a grid.

        Args:
            d (int): dimension
            i (int): dimension index, meaning at which dimension will this
                     array appear in the resulting grid
            array (1d np.array) : array which will be reshaped
        """
        if hasattr(array, '__getitem__'):
            shape = [1] * d
            shape[i] = len(np.atleast_1d(array))
            array = np.reshape(array, shape)
        return array

    def mean(self, t, x, y=None, z=None):
        """Calculates the mean concentration."""
        #filter out all None values in positions
        pos = [p for p in (x,y,z) if p is not None]

        return_shape = []

        t_dim = 0
        #if time is a vector, reshape accordingly
        if hasattr(t,'__getitem__') and len(np.atleast_1d(t)) > 1:
            return_shape.append(len(t))
            t_dim = 1
            t = self.__reshape_axis(len(pos)+t_dim, 0, t)

        #for more than 1 spatial dimension reshape accordingly
        for i in range(len(pos)):
            return_shape.insert(i+1, len(np.atleast_1d(pos[i])))
            pos[i] = self.__reshape_axis(len(pos)+t_dim, i+t_dim, pos[i])

        #TODO find a better way for this!
        #without it, t *= foat(x) returns an int, if t scalar!
        try:
            t = float(t)
        except:
            pass

        pre_fact = np.ones_like(t)
        exp_arg = np.zeros(return_shape)
        for d in range(self.dimension):
            pre_fact *= sqrt(4*pi*self.D_ens[d] * (t+self.t0[d]))
            exp_arg -= ((pos[d]-self.u_mean[d]*t)**2 /
                        (4*self.D_ens[d]*(t+self.t0[d])))
        r = exp(exp_arg) / pre_fact
        return np.squeeze(r)

    def mean_cm(self, t):
        """Calculates the mean concentration at the centre of mass."""
        fact = np.ones_like(t)
        for d in range(self.dimension):
            fact *= sqrt(4*pi*self.D_ens[d] * (t+self.t0[d]))
        return np.squeeze(1. / fact)

    def var_integrand(self, t_prime, t, pos):
        """Calculate integrand of time integral of the mean conc. variance.

        This time integral cannot be evaluated analytically without
        approximations.

        Args:
            t_prime (float): variable of integration
            t (float): time
            pos (1d array): position vector (1-, 2-, or 3-dimensional),
                            single vector
        """
        if t == 0:
            return 0
        #get member variables into local scope and get rid of self reference
        #spatial dimension
        t_p = t_prime
        u_mean = self.u_mean
        D_ens = self.D_ens
        t0 = self.t0

        Chi = self.model_fct[self.model]

        tt0tp = 2*t + t0 - t_p
        tpt0 = t_p + t0
        disp_area = 2. * D_ens * tt0tp
        pos_prop_2 = (pos - u_mean*t)**2

        prod = np.prod(exp(-pos_prop_2 / disp_area) /
                       np.sqrt(4.*pi*D_ens**2*tt0tp*tpt0))

        pre_fact = np.sum(2.*D_ens * prod * ((t-t_p) / (disp_area*tpt0) +
                     pos_prop_2/disp_area**2))

        exp_arg = -2. * quad(Chi, t_p, t)[0]

        return pre_fact * exp(exp_arg)

    def var_cm_integrand(self, t_prime, t):
        """Calculate integrand of time integral of the variance at centre of m.

        This time integral cannot be evaluated analytically without
        approximations.

        Args:
            t_prime (float): variable of integration
            t (float): time
        """
        if t == 0:
            return 0
        #get member variables into local scope and get rid of self reference
        #spatial dimension
        t_p = t_prime
        D_ens = self.D_ens
        t0 = self.t0

        Chi = self.model_fct[self.model]

        tt0tp = 2*t + t0 - t_p
        tpt0 = t_p + t0

        prod = 1. / np.prod(np.sqrt(4.*pi*D_ens**2*tt0tp*tpt0))
        pre_fact = np.sum(2.*D_ens * prod * (t-t_p) / (2.*D_ens*tt0tp*tpt0))
        exp_arg = -2. * quad(Chi, t_p, t)[0]

        return pre_fact * exp(exp_arg)

    def integrate_1d(self, t_array, pos):
        c_var = np.empty((len(t_array), len(pos[0])))
        for i, t in enumerate(t_array):
            for j, x in enumerate(pos[0]):
                c_var[i,j] = quad(self.var_integrand, 0, t, args=(t, x))[0]
        return c_var

    def integrate_2d(self, t_array, pos):
        pos_arg = np.empty(self.dimension)
        c_var = np.empty((len(t_array), len(pos[0]), len(pos[1])))
        for i, t in enumerate(t_array):
            for j, x in enumerate(pos[0]):
                pos_arg[0] = x
                for k, y in enumerate(pos[1]):
                    pos_arg[1] = y
                    c_var[i,j,k] = quad(self.var_integrand, 0, t,
                                        args=(t, pos_arg))[0]
        return c_var

    def integrate_3d(self, t_array, pos):
        pos_arg = np.empty(self.dimension)
        c_var = np.empty((len(t_array), len(pos[0]), len(pos[1]),
                          len(pos[2])))
        for i, t in enumerate(t_array):
            for j, x in enumerate(pos[0]):
                pos_arg[0] = x
                for k, y in enumerate(pos[1]):
                    pos_arg[1] = y
                    for l, z in enumerate(pos[2]):
                        pos_arg[2] = z
                        c_var[i,j,k,l] = quad(self.var_integrand, 0, t,
                                              args=(t, pos_arg))[0]
        return c_var

    def variance(self, t, x_array, y_array=None, z_array=None):
        """Calculate the concentration variance.

        Numerically integrates a semi-analytical solution for the
        concentration variance.

        Args:
            t (float, array): time
            x (float, array): longitudinal distance from injection point
            y (float, array, optional): variable which is the transversal
                                 distance from injection point, if y is None:
                                 dimension is assumed to be 1
            z (float, array, optional): variable which is the transversal
                                 distance from injection point and
                                 perpendicular to y,
                                 if z is None:
                                 dimension is assumed to be 1 or 2

        Examples:
            >>> C1d = Concentration(1, 1, 0.01, 1, 0.1, 10)
            >>> C2d = Concentration(2, [1, 0], [0.01, 0.01], [1, 1], 0.1, \
                                    [10, 10])
            >>> C3d = Concentration(3, [1, 0, 0], [0.01, 0.01, 0.01], \
                                    [1, 1, 1], 0.1, [10, 10, 10])
            >>> #mean concentration at a single point in space and time
            >>> t = 10
            >>> x = 10
            >>> y = 0
            >>> z = 0
            >>> C1d.model = 'IEM'
            >>> print(np.around(C1d.variance(t, x), 4))
            0.0033
            >>> print(np.around(C2d.variance(t, x, y), 4))
            0.0037
            >>> print(np.around(C3d.variance(t, x, y, z), 5))
            0.00233
            >>> #mean concentration at a single point in time over space axis
            >>> t = 10
            >>> x = np.linspace(9, 11, 100)
            >>> y = np.linspace(-4, 4, 20)
            >>> z = np.linspace(-4, 4, 20)
            >>> print(np.around(C1d.variance(t, x)[50], 4))
            0.0033
            >>> print(np.around(C2d.mean(t, x, y)[50,10], 4))
            0.0981
            >>> print(np.around(C3d.mean(t, x, y, z)[50,10,10], 5))
            0.05811
            >>> #mean concentration over space and time axis
            >>> t = np.linspace(0, 10, 10)
            >>> x = np.linspace(9, 11, 100)
            >>> y = np.linspace(-4, 4, 20)
            >>> z = np.linspace(-4, 4, 20)
            >>> print(np.around(C1d.mean(t, x)[-1,50], 4))
            0.1741
        """
        pos = [np.atleast_1d(p) for p in (x_array,y_array,z_array)
                if p is not None]
        t_array = np.atleast_1d(t)
        if self.dimension == 1:
            c_var = self.integrate_1d(t_array, pos)
        elif self.dimension == 2:
            c_var = self.integrate_2d(t_array, pos)
        else:
            c_var = self.integrate_3d(t_array, pos)
        return np.squeeze(c_var)

    def variance_cm(self, t):
        """Calculate the concentration variance at the centre of mass.

        Numerically integrates a semi-analytical solution for the
        concentration variance at the centre of mass.

        Args:
            t (float, array): time
        """
        t_array = np.atleast_1d(t)
        c_var_cm = np.empty_like(t_array)
        for i, t in enumerate(t):
            c_var_cm[i] = quad(self.var_cm_integrand, 0, t, args=(t,))[0]
        return np.squeeze(c_var_cm)

    def variance_upper_limit(self, t, x, y=None, z=None):
        """Calculate an upper limit for the concentration variance.

           See Andricevic 1998 eq. (25) for details.

       Args:
            t (float): time
            x (float): longitudinal distance from injection point
            y (float, optional): variable which is the transversal distance
                                 from injection point, if y is None:
                                 dimension is assumed to be 1
            z (float, optional): variable which is the transversal distance
                                 from injection point and perpendicular to y,
                                 if z is None:
                                 dimension is assumed to be 1 or 2
        """
        #c_0 assumed to be 1
        return 0.25 - (self.mean(t, x, y, z) - 0.5)**2


if __name__ == '__main__':
    import doctest
    doctest.testmod()
