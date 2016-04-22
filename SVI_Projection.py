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
# deployed @ 16/4/19
# TD
#  (1) SVI Projection of each slice: Fit (raw) SVI format for each tenor
# Check pt
#  (1) Whether the opt. func. satisfy no-arbitrage condition (theoretical)
#  (2) Close ATM vol

# SVI Slice: in raw-SVI format
class SVI_S(object):

    # modified ver (4/20, 황보람)
    # to try option-value minimization, add rate term structure member
    def __init__(self, K, T, F, iv_slice, ir, wgttype = 'none'):
        self._K = K                 # vector
        self._T = T                 # scalar
        self._F = F                 # scalar
        self._iv = iv_slice         # vector

        # added (4/20, 황보람)
        self._ir = ir
        self._wgttype = wgttype

        self._k = np.log(K / F)     # vector

        # for vega weight
        self._vega = []
        for i in range(len(self._k)):
            if i < 10:
                self._vega.append(bs_formula('p', self._F, self._K[i], self._T, self._ir(self._T), 0, self._iv[i], 'v'))
            elif i > 10:
                self._vega.append(bs_formula('c', self._F, self._K[i], self._T, self._ir(self._T), 0, self._iv[i], 'v'))
            else:
                self._vega.append(bs_formula('c', self._F, self._K[i], self._T, self._ir(self._T), 0, self._iv[i], 'v') +
                                  bs_formula('p', self._F, self._K[i], self._T, self._ir(self._T), 0, self._iv[i], 'v'))

        print 'K: ', K
        print 'F: ', F
        print 'self._k: ', self._k

        # constraints
        self.cons = (
            # constraints for parameter itself
            { 'type': 'ineq',
              'fun' : lambda x: np.array([x[0]]),
              'jac' : lambda x: np.array([1., 0., 0., 0., 0.])
            }, # for a
            { 'type': 'ineq',
              'fun' : lambda x: np.array([x[1]]),
              'jac' : lambda x: np.array([0., 1., 0., 0., 0.])
            }, # for b
            { 'type': 'ineq',
              # original constraints
              'fun' : lambda x: np.array([1-abs(x[2])]),
              'jac' : lambda x: np.array([0., 0., -x[2]/abs(x[2]), 0., 0.])
            }, # for rho
            {'type': 'ineq',
             'fun': lambda x: np.array([x[4]]),
             'jac': lambda x: np.array([0., 0., 0., 0., 1.])
             }, # for sigma

            # constraints for parameter combination
            { 'type': 'ineq',
              'fun' : lambda x: np.array([4 - x[1]*self._T*(1+abs(x[2]))]),
              'jac' : lambda x: np.array([0., -self._T*(1+abs(x[2])), -x[1]*self._T*x[2]/abs(x[2]), 0., 0.])
            } # combined constraints 1. preventing vertical arbitrage (Gatheral, 2004)
            )
        self._x = np.array([0.] * 5)          # Coefficients

    # 0: a     <= the overall level of variance, a vertical traslation of the smile
    # 1: b     <= the angle between the left and right asymptotes, where the asymptotes are
    #               var_l(k) = a - b(1-rho)(k-m)
    #               var_r(k) = a + b(1-rho)(k-m)
    # 2: rho   <= the counter-clockwise rotation of the smile
    # 3: m     <= translating the graph from left to right
    # 4: sigma <= the smoothness of the vertex
    # i: k index
    # var_svi^2 = a + b(rho(k-m)+sqrt((k-m)^2 + sigma^2))
    # raw-SVI formula
    def SVI_var_k(self, x, k):
        return ( x[0] + x[1]*(x[2]*(k-x[3]) + np.sqrt((k-x[3])**2 + x[4]**2)) )

    # derivative vector of raw-SVI formula
    def SVI_var_k_deriv(self, x, k):
        dtran = k - x[3]
        sqrt = np.sqrt(dtran**2 + x[4]**2)

        d0 = 1.0
        d1 = x[2]*dtran + sqrt
        d2 = x[1]*dtran
        d3 = -x[1]*(x[2] + dtran/sqrt)
        d4 = x[1]*x[4]/sqrt                     #d4 = x[4]/sqrt : original...is x[1]*x[4]/sqrt correct? (4/19, 황보람)
        return np.array([ d0, d1, d2, d3, d4 ])

    # objective function: L = 0.5 Sum_i (SVI_var - IV^2)^2 * weight
    def SVI_L(self, x):
        sm = 0
        for i in range(len(self._k)):
            # original: minimize value diff
            if self._wgttype == 'none':
                sm += (( self.SVI_var_k(x, self._k[i]) - (self._iv[i]**2) )**2)
            # alter. 1: minimize value diff w/ vega weight (use OTM / straddle at ATM)
            elif self._wgttype == 'vega':
                sm += (( self.SVI_var_k(x, self._k[i]) - (self._iv[i]**2) )**2) * self._vega[i]
        sm *= 0.5
        return sm

    # Jacobian: DL = Sum_i (SVI_var - IV^2) * DSVI_var * weight
    def SVI_L_deriv(self, x):
        d = [0.] * 5

        for i in range(len(self._k)):
            var_k = self.SVI_var_k(x, self._k[i])
            var_k_deriv = self.SVI_var_k_deriv(x, self._k[i])
            # original: minimize value diff
            if self._wgttype == 'none':
                d += (var_k - (self._iv[i]**2)) * var_k_deriv
            # alter. 1: minimize value diff w/ vega weight (use OTM / straddle at ATM)
            elif self._wgttype == 'vega':
                d += (var_k - (self._iv[i] ** 2)) * var_k_deriv * self._vega[i]
        return d

    # Calibration: Declare minimization
    def calibration(self):
        # original initial value: (0.1, 0.1, 0.1, 0.1, 0.1), which alternatives are possibly efficient?
        init_val = [0.1] * 5
        res = minimize(
            self.SVI_L, init_val,
            #jac=self.SVI_L_deriv,
            constraints=self.cons,
            method='COBYLA',
            options={'disp': True, 'maxiter': 10000},
            tol=1e-8
        )
        self._x = np.array(res.x)
        print 'solution'
        print self._x

    # x: vector; x-coordinate
    def get_SVI_Vol(self, s_vec):
        y_vec = []

        for s in s_vec:
            y_vec.append(self.SVI_var_k(self._x, s))

        # add: return optimized value (4/19, 황보람)
        # print np.sqrt(np.array(y_vec))
        return np.sqrt(np.array(y_vec))

    def get_SVI_Param(self):
        return self._x

