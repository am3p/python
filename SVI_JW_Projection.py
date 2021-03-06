# -*- coding: utf-8 -*-

import numpy as np
import datetime

from miraepy.derivatives.params.impliedvol_all import ImpliedVolSurface #, ImpliedVolTS
from miraepy.derivatives.params.dividend_all import DiscreteDividend, ContinuousDividend
from miraepy.derivatives.params.interestrate import SpotRateCurve
from miraepy.common.code import ir_id #ticker # , get_all_from_db, get_code_from_db , get_ticker_from_db, ticker_to_name_from_db
from scipy.optimize import minimize

from miraepy.derivatives.blackscholes import bs_formula

# Original code: 장광성 차장님
# written @ 16/4/21
# TD
#  (1) SVI-JW Projection of each slice: Fit SVI-JW format for each tenor
# Check pt
#  (1) Whether the opt. func. satisfy no-arbitrage condition (theoretical)
#  (2) Close ATM vol

# SVI Slice: in raw-SVI format
class SVI_S(object):

    # original initialize
    # def __init__(self, K, T, F, iv_slice):

    # modified ver (4/20, 황보람)
    # to try option-value minimization, add rate term structure member
    def __init__(self, K, T, F, iv_slice, ir, init_val):
        self._K = K                 # vector
        self._T = T                 # scalar
        self._F = F                 # scalar
        self._iv = iv_slice         # vector

        # added pt (4/20, 황보람)
        self.ir = ir

        self._k = np.log(K / F)     # vector

        # for vega weight (possibly use)
        self.vega = []
        for i in range(len(self._k)):
            if i < 10:
                self.vega.append(bs_formula('p', self._F, self._K[i], self._T, self.ir(self._T), 0, self._iv[i], 'v'))
            elif i > 10:
                self.vega.append(bs_formula('c', self._F, self._K[i], self._T, self.ir(self._T), 0, self._iv[i], 'v'))
            else:
                self.vega.append(bs_formula('c', self._F, self._K[i], self._T, self.ir(self._T), 0, self._iv[i], 'v') +
                                 bs_formula('p', self._F, self._K[i], self._T, self.ir(self._T), 0, self._iv[i], 'v'))

        # constraints for SVI-JW (>0): JW form should be optimized in raw-converted version
        self.cons = (
            # constraints of parameter itself
            { 'type': 'ineq',
              'fun' : lambda x: np.array([x[0]]),
              #'jac' : lambda x: np.array([1., 0., 0.])
            }, # for v

            # constraints of parameter combination
            {'type': 'ineq',
             'fun': lambda x: np.array([2. * x[1] + x[2]]),
             #'jac': lambda x: np.array([0., 2., 1.])
            } # combined constraints 1. preventing vertical arbitrage (Gatheral, 2004)

           # Others (combination)?
            )
        self._init_val = init_val
        self._x = np.array([0.] * 3)          # Coefficients

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
        # JW form param extraction: reduced param conversion
        v = x[0]
        psi = x[1]
        p = x[2]
        c = p + 2. * psi
        vt = v * 4. * p * c / ((p + c) ** 2.)
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

    def SVI_var_k(self, x, k):
        return ( x[0] + x[1]*(x[2]*(k-x[3]) + np.sqrt((k-x[3])**2 + x[4]**2)) )

    # dtran = _k[i] - m
    # sqrt = np.sqrt(dtran**2 + sigma**2)
    #
    # d0 = 1
    # d1 = rho*dtran + sqrt
    # d2 = b*dtran
    # d3 = -b*rho - dtran / sqrt
    # d4 = sigma / sqrt

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

    # L = 0.5 Sum_i (SVI_var - IV^2)^2 -> L = 0.5 Sum_i (SVI_var - IV^2*t)^2
    def SVI_L(self, x):
        sm = 0
        x_new = self.SVI_paramconv(x)
        for i in range(len(self._k)):
            #original: minimize value diff
            #implied variance form
            sm += (( self.SVI_var_k(x_new, self._k[i]) - (self._iv[i]**2 * self._T) )**2)

            # ver 1b: minimize value diff w/ vega weight (use OTM / straddle at ATM)
            #sm += (( self.SVI_var_k(x_new, self._k[i]) - ((self._iv[i]**2) * self._T) )**2) * self.vega[i]
        if np.isnan(sm):
            sm = 1e10
        else:
            sm *= 0.5
        return sm

    # DL = Sum_i (SVI_var - IV^2) * DSVI_var
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

    def calibration(self):
        res = minimize(
            self.SVI_L, self._init_val,
            #jac=self.SVI_L_deriv,
            constraints=self.cons,
            method='SLSQP',
            options={'disp': True, 'maxiter': 10000},
            tol=1e-6
        )
        self._x = np.array(res.x)
        print 'solution'
        print self._x

    # x: vector; x-coordinate
    def get_SVI_Vol(self, s_vec):
        y_vec = []
        x_new = self.SVI_paramconv(self._x)
        for s in s_vec:
            y_vec.append(self.SVI_var_k(x_new, s) / self._T)
        # add: return optimized value (4/19, 황보람)
        # print np.sqrt(np.array(y_vec))
        return np.sqrt(np.array(y_vec))
