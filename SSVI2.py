# -*- coding: utf-8 -*-

import numpy as np
import datetime

from miraepy.derivatives.params.impliedvol_all import ImpliedVolSurface #, ImpliedVolTS
from miraepy.derivatives.params.dividend_all import DiscreteDividend, ContinuousDividend
from miraepy.derivatives.params.interestrate import SpotRateCurve
from miraepy.common.code import ir_id #ticker # , get_all_from_db, get_code_from_db , get_ticker_from_db, ticker_to_name_from_db
from scipy.optimize import minimize
from scipy.interpolate import interp1d

from miraepy.derivatives.blackscholes import bs_formula

# Original code: 장광성 차장님
# written @ 16/4/21
# TD
#  (1) SSVI fitting: following SVI fitting process in Gatheral and Jacquier (2013)
#      returns initial value of SVI-JW parameter (psi, p, c, vt)

# SVI class: store whole vol surf.
class SSVI(object):
    # modified ver (4/20, 황보람)
    # to try vega-weighted minimization, add rate term structure member
    def __init__(self, K, T, F, iv_mat, ir, type = 'Heston', wgttype = 'none'):
        self._K = K                 # vector
        self._T = T                 # vector
        self._F = F                 # vector
        self._iv = iv_mat           # matrix

        # added feature (4/20, 황보람)
        self.ir = ir
        self._k = []
        self._theta = []
        for i in range(len(self._T)):
            k_vec = np.log(K / F[i])
            iv_interp = interp1d(k_vec, self._iv[i][:], kind='cubic')
            iv_ATM = iv_interp(0)
            self._k.append(np.array(k_vec))  # vector
            self._theta.append((iv_ATM ** 2.) * self._T[i])   # vector, ATM total variance proxy

        self._type = type
        self._wgttype = wgttype

        # for vega weight (possibly use)
        self.vega = []
        for i in range(len(self._T)):
            for j in range(len(self._K)):
                if j < 10:
                    self.vega.append(bs_formula('p', self._F[i], self._K[j], self._T[i], self.ir(self._T[i]), 0, self._iv[i][j], 'v'))
                elif j > 10:
                    self.vega.append(bs_formula('c', self._F[i], self._K[j], self._T[i], self.ir(self._T[i]), 0, self._iv[i][j], 'v'))
                else:
                    self.vega.append(bs_formula('c', self._F[i], self._K[j], self._T[i], self.ir(self._T[i]), 0, self._iv[i][j], 'v') + \
                                     bs_formula('p', self._F[i], self._K[j], self._T[i], self.ir(self._T[i]), 0, self._iv[i][j], 'v'))
        self.vega = np.reshape(self.vega, (len(self._T), len(self._K)))

        # constraints for SSVI, Heston-like param.
        self.cons_SSVI_Heston = (
            # constraints of parameter itself
            { 'type': 'ineq',
              'fun' : lambda x: np.array([1 - np.abs(x[0])]),
              'jac' : lambda x: np.array([-np.sign(x[0]), 0.])
            }, # for rho
            {'type': 'ineq',
             'fun': lambda x: np.array([x[1]]),
             'jac': lambda x: np.array([0., 1.])
             }, # for lambda

            # constraints of parameter combination
            {'type': 'ineq',
             'fun': lambda x: np.array([4. * x[1] - (1. + np.abs(x[0]))]),
             'jac': lambda x: np.array([-np.sign(x[0]), 4.])
             } # free of static arbitrage condition
            )
        self._x_Heston = np.array([0.] * 2)          # Coefficients

        # constraints for SSVI, Power-law param.
        self.cons_SSVI_Power = (
            # constraints of parameter itself
            {'type': 'ineq',
             'fun': lambda x: np.array([1 - np.abs(x[0])]),
             'jac': lambda x: np.array([-np.sign(x[0]), 0., 0.])
             }, # for rho
            {'type': 'ineq',
             'fun': lambda x: np.array([x[1]]),
             'jac': lambda x: np.array([0., 1., 0.])
             }, # for eta
            {'type': 'ineq',
             'fun': lambda x: np.array([0.5 - np.abs(x[2] - 0.5)]),
             'jac': lambda x: np.array([0., 0., -np.sign(x[2])])
             }, # for gamma

            # constraints of parameter combination
            {'type': 'ineq',
             'fun': lambda x: np.array(2 - x[1] * (1 + np.abs(x[0]))),
             'jac': lambda x: np.array([-np.sign(x[0]), (1 + np.abs(x[0])), 0.])
             } # free of static arbitrage condition
            )
        self._x_Power = np.array([0.] * 3)  # Coefficients

    def SSVI_Heston(self, l, theta):
        return 1. / (l * theta) * (1. - (1. - np.exp(-l*theta)) / (l * theta))

    def SSVI_Power(self, eta, gamma, theta):
        return eta * (theta ** -gamma)

    def SSVI_totalvar(self, x, kind, tind):
        k = self._k[tind][kind]
        theta = self._theta[tind]
        phi = 0
        if self._type == 'Heston':
            phi = self.SSVI_Heston(x[1], theta)
        elif self._type == 'Power':
            phi = self.SSVI_Power(x[1], x[2], theta)

        return theta / 2. * (1. + x[0] * phi * k + np.sqrt( (phi * k + x[0]) ** 2. + (1. - x[0] ** 2.) ))

    # Heston-like formula Jacobian: correct? check!
    def SSVI_totalvar_deriv_Heston(self, x, kind, tind):
        k = self._k[tind][kind]
        theta = self._theta[tind]
        phi = self.SSVI_Heston(x[1], theta)

        # check this calculation is correct
        phi0 = np.exp(-x[1] * theta) * (x[1] * theta - np.exp(-x[1] * theta) + 1.) / ((x[1] ** 2) * theta)

        d0 = theta / 2. * (phi * k + phi * k / (np.sqrt((phi * k + x[0]) ** 2 + (1 - x[0] ** 2))))
        d1 = phi0 * theta / 2. * (x[0] * k + ((phi * k + x[0]) * k) / (np.sqrt((phi * k + x[0]) ** 2 + (1 - x[0] ** 2))))

        return np.array([d0, d1])

    # Power law formula Jacobian: correct? check!
    def SSVI_totalvar_deriv_Power(self, x, kind, tind):
        k = self._k[tind][kind]
        theta = self._theta[tind]
        phi = self.SSVI_Power(x[1], x[2], theta)

        # check this calculation is correct
        phi0 = theta ** (-x[2])
        phi1 = -x[1] * (theta ** (-x[2])) * np.log(theta)

        d0 = theta / 2. * (phi * k + phi * k / (np.sqrt((phi * k + x[0]) ** 2 + (1 - x[0] ** 2))))
        d1 = phi0 * theta / 2. * (x[0] * k + ((phi * k + x[0]) * k) / (np.sqrt((phi * k + x[0]) ** 2 + (1 - x[0] ** 2))))
        d2 = phi1 * theta / 2. * (x[0] * k + ((phi * k + x[0]) * k) / (np.sqrt((phi * k + x[0]) ** 2 + (1 - x[0] ** 2))))

        return np.array([d0, d1, d2])

    # L = 0.5 Sum_i (SVI_var - IV^2*t)^2
    def SSVI_L(self, x):
        sm = 0
        for i in range(len(self._T)):
            for j in range(len(self._K)):
                # original: minimize value diff
                if self._wgttype == 'none':
                    sm += (( self.SSVI_totalvar(x, j, i) - (self._iv[i][j]**2. * self._T[i]) )**2.)
                # alter. 1: vega-weighted
                elif self._wgttype == 'vega':
                    sm += ((self.SSVI_totalvar(x, j, i) - (self._iv[i][j] ** 2. * self._T[i])) ** 2.) * self.vega[i][j]
        sm *= 0.5

        return sm

    # DL = Sum_i (SVI_var - IV^2*t) * DSVI_var
    def SSVI_L_deriv_Heston(self, x):
        d = [0.] * 2
        for i in range(len(self._T)):
            for j in range(len(self._K)):
                var_k = self.SSVI_totalvar(x, j, i)
                var_k_deriv = self.SSVI_totalvar_deriv_Heston(x, j, i)

                # original: minimize value diff
                if self._wgttype == 'none':
                    d += (var_k - (self._iv[i][j] ** 2. * self._T[i])) * var_k_deriv
                # alter. 1: vega-weighted
                elif self._wgttype == 'vega':
                    d += (var_k - (self._iv[i][j] ** 2. * self._T[i])) * var_k_deriv * self.vega[i][j]

        return d

    # DL = Sum_i (SVI_var - IV^2) * DSVI_var
    def SSVI_L_deriv_Power(self, x):
        d = [0.] * 3
        for i in range(len(self._T)):
            for j in range(len(self._K)):
                var_k = self.SSVI_totalvar(x, j, i)
                var_k_deriv = self.SSVI_totalvar_deriv_Power(x, j, i)

                # original: minimize value diff
                if self._wgttype == 'none':
                    d += (var_k - (self._iv[i][j] ** 2. * self._T[i])) * var_k_deriv
                # alter. 1: vega-weighted
                elif self._wgttype == 'vega':
                    d += (var_k - (self._iv[i][j] ** 2. * self._T[i])) * var_k_deriv * self.vega[i][j]

        return d

    # Calibration (Heston-like phi)
    def SSVI_calib_Heston(self):
        # initial value check
        init_val = [0., 1.]
        res = minimize(
            self.SSVI_L, init_val,
            #jac=self.SSVI_L_deriv_Heston,
            constraints=self.cons_SSVI_Heston,
            method='SLSQP',
            options={'disp': True, 'maxiter': 10000},
            tol=1e-8
        )
        self._x = np.array(res.x)

        # initial value of SVI-JW parameter
        init_val = []
        for i in range(len(self._T)):
            v = self._theta[i] / self._T[i]
            psi = 0.5 * self._x[0] * np.sqrt(self._theta[i]) * self.SSVI_Heston(self._x[1], self._theta[i])
            p = 0.5 * np.sqrt(self._theta[i]) * self.SSVI_Heston(self._x[1], self._theta[i]) * (1. - self._x[0])
            c = 0.5 * np.sqrt(self._theta[i]) * self.SSVI_Heston(self._x[1], self._theta[i]) * (1. + self._x[0])
            vt = self._theta[i] / self._T[i] * (1. - self._x[0] ** 2.)

            init_val.append(np.array([psi, p, c, vt]))
        return init_val

    # Calibration (power law phi)
    def SSVI_calib_Power(self):
        # initial value check
        init_val = [0., 1., 0.5]
        res = minimize(
            self.SSVI_L, init_val,
            #jac=self.SSVI_L_deriv_Power,
            constraints=self.cons_SSVI_Power,
            method='SLSQP',
            options={'disp': True, 'maxiter': 10000},
            tol=1e-8
        )
        self._x = np.array(res.x)

        # initial value of SVI-JW parameter
        init_val = []
        for i in range(len(self._T)):
            v = self._theta[i] / self._T[i]
            psi = 0.5 * self._x[0] * np.sqrt(self._theta[i]) * self.SSVI_Power(self._x[1], self._x[2], self._theta[i])
            p = 0.5 * np.sqrt(self._theta[i]) * self.SSVI_Power(self._x[1], self._x[2], self._theta[i]) * (1. - self._x[0])
            c = 0.5 * np.sqrt(self._theta[i]) * self.SSVI_Power(self._x[1], self._x[2], self._theta[i]) * (1. + self._x[0])
            vt = self._theta[i] / self._T[i] * (1. - self._x[0] ** 2.)

            init_val.append(np.array([psi, p, c, vt]))
        return init_val

    # Calibration part
    def SSVI_calib(self):
        init_val = []
        if self._type == 'Heston':
            init_val = self.SSVI_calib_Heston()
        elif self._type == 'Power':
            init_val = self.SSVI_calib_Power()
        return init_val