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
#  (1) SVI-JW Projection of each slice: Fit SVI-JW format for each tenor with v fixed
#      (v is fixed to (ATMF vol) ** 2, which is interpolated)

# SVI Slice: in SVI-JW format
class SVI_S(object):

    # modified ver (4/20, 황보람)
    # to try vega-weighted minimization, add rate term structure member
    def __init__(self, K, T, F, iv_slice, ir, init_val, wgttype = 'none'):
        self._K = K                 # vector
        self._T = T                 # scalar
        self._F = F                 # scalar
        self._iv = iv_slice         # vector

        self._ir = ir
        self._wgttype = wgttype

        self._k = np.log(K / F)     # vector
        iv_interp = interp1d(self._k, self._iv, kind='cubic')   # ATMF vol interp (k = 0)
        self.iv_ATM = iv_interp(0)

        # for vega weight (possibly using)
        self.vega = []
        for i in range(len(self._k)):
            if i < 10:
                self.vega.append(bs_formula('p', self._F, self._K[i], self._T, self._ir(self._T), 0, self._iv[i], 'v'))
            elif i > 10:
                self.vega.append(bs_formula('c', self._F, self._K[i], self._T, self._ir(self._T), 0, self._iv[i], 'v'))
            else:
                self.vega.append(bs_formula('c', self._F, self._K[i], self._T, self._ir(self._T), 0, self._iv[i], 'v') +
                                 bs_formula('p', self._F, self._K[i], self._T, self._ir(self._T), 0, self._iv[i], 'v'))

        # constraints for SVI-JW (>0): JW form should be optimized in raw-converted version
        self.cons = (
            # constraints of parameter itself

            # constraints of parameter combination
            {'type': 'ineq',
             'fun': lambda x: np.array([2. * x[0] + x[1]]),
            }, # beta assumption ( -1 <= beta <= 1 )

            {'type': 'ineq',
             'fun': lambda x: np.array ([-2. * x[0] + x[2]]),
            }, # beta assumption ( -1 <= beta <= 1 )

            {'type': 'ineq',
             'fun': lambda x: np.array([x[3]]),
             } # vt assumption ( in SVI-raw format, -1 < rho < 1 )

           # Others (combination)?
            )
        self._init_val = init_val
        self._x = np.array([0.] * 4)          # Coefficients

    # 0: v     <= ATM variance
    # 1: psi   <= ATM skew (d (BSvol) / dk) @ k = 0
    # 2: p     <= the slope of the left (put OTM) wing
    # 3: c     <= the slope of the right (call OTM) wing ( p + 2 * psi)
    # 4: vt    <= the minimum implied variance ( v * 4 * p * c / (p + c)^2 )
    # additional factor: w = v * t, parameters are constant scaled by 1 / sqrt(w)

    # SVI-raw form
    # i: k index
    # var_svi^2 = a + b(rho(k-m)+sqrt((k-m)^2 + sigma^2))

    # from JW form parameter to raw form parameter
    def SVI_paramconv(self, x):
        # JW form param extraction: v is the only external factor
        v = (self.iv_ATM) ** 2.
        psi = x[0]
        p = x[1]
        c = x[2]
        vt = x[3]
        w = v * self._T

        # calc to raw form
        b = np.sqrt(w) / 2. * (c + p)
        rho = 1. - p * np.sqrt(w) / b
        beta = rho - 2. * psi * np.sqrt(w) / b
        alpha = np.sign(beta) * np.sqrt(1. / beta ** 2. - 1)
        m = (v - vt) * self._T / (b * (-rho + np.sign(alpha) * np.sqrt(1. + alpha ** 2.) - alpha * np.sqrt(1. - rho ** 2.)))
        if m == 0:
            sigma = (vt * self._T - w) / b / (np.sqrt(1. - rho ** 2.) - 1)
        else:
            sigma = alpha * m
        a = vt * self._T - b * sigma * np.sqrt(1. - rho ** 2.)

        x_new = [a, b, rho, m, sigma]
        return x_new

    # SVI-raw formula
    def SVI_var_k(self, x, k):
        return ( x[0] + x[1]*(x[2]*(k-x[3]) + np.sqrt((k-x[3])**2 + x[4]**2)) )

    # SVI-raw formula derivatives
    def SVI_var_k_deriv(self, x, k):
        dtran = k - x[3]
        sqrt = np.sqrt(dtran**2 + x[4]**2)

        d0 = 1.0
        d1 = x[2]*dtran + sqrt
        d2 = x[1]*dtran
        d3 = -x[1]*(x[2] + dtran/sqrt)
        #d4 = x[4]/sqrt : original...is x[1]*x[4]/sqrt correct? (4/19, 황보람)
        d4 = x[1]*x[4]/sqrt
        return np.array([ d0, d1, d2, d3, d4 ])

    # L = 0.5 Sum_i (SVI_var - IV^2*t)^2
    def SVI_L(self, x):
        sm = 0
        x_new = self.SVI_paramconv(x)
        for i in range(len(self._k)):
            if self._wgttype == 'none':
            # original: minimize value diff
                sm += (( self.SVI_var_k(x_new, self._k[i]) - (self._iv[i]**2 * self._T) )**2)
            # alter. 1: minimize value diff w/ vega weight (use OTM / straddle at ATM)
            elif self._wgttype == 'vega':
                sm += (( self.SVI_var_k(x_new, self._k[i]) - ((self._iv[i]**2) * self._T) )**2) * self.vega[i]
        if np.isnan(sm):
            sm = 1e10
        else:
            sm *= 0.5

        return sm

    # DL = Sum_i (SVI_var - IV^2) * DSVI_var
    # Make it when the explicit derivatives are calculated
    def SVI_L_deriv(self, x):
        d = [0.] * 5
        x_new = self.SVI_paramconv(x)
        for i in range(len(self._k)):
            var_k = self.SVI_var_k(x_new, self._k[i])
            var_k_deriv = self.SVI_var_k_deriv(x_new, self._k[i])

            # original: minimize value diff
            # d += (var_k - (self._iv[i]**2 * self._T)) * var_k_deriv

            # ver 1b: minimize value diff w/ vega weight (use OTM / straddle at ATM)
            # d += (var_k - (self._iv[i]**2 * self._T)) * var_k_deriv * self.vega[i]
        return d

    # Calibration part
    def calibration(self):
        res = minimize(
            self.SVI_L, self._init_val,
            #jac=self.SVI_L_deriv,
            constraints=self.cons,
            method='SLSQP',
            options={'disp': True, 'maxiter': 10000},
            tol=1e-8
        )
        self._x = np.array(res.x)
        print 'solution'
        print self._x

    # Return SVI-calculated vol
    # x: vector; x-coordinate
    def get_SVI_Vol(self, s_vec):
        y_vec = []
        x_new = self.SVI_paramconv(self._x)
        for s in s_vec:
            y_vec.append(self.SVI_var_k(x_new, s) / self._T)
        # add: return optimized value (4/19, 황보람)
        return np.sqrt(np.array(y_vec))
