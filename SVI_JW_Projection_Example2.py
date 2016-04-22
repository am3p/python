# -*- coding: utf-8 -*-
import datetime
import numpy as np
from scipy.interpolate import interp1d

from miraepy.derivatives.params.impliedvol_all import ImpliedVolSurface #, ImpliedVolTS
from miraepy.derivatives.params.dividend_all import DiscreteDividend, ContinuousDividend
from miraepy.derivatives.params.interestrate import SpotRateCurve
from miraepy.common.code import ir_id #ticker # , get_all_from_db, get_code_from_db , get_ticker_from_db, ticker_to_name_from_db

from SVI_JW_ATMFixed2 import SVI_S
from SSVI2 import SSVI

instrument_ids = ['SX5E',       'SPX'      ,'KOSPI200'    , 'HSCEI'         ]
tickers        = ['sx5e index', 'spx index','kospi2 index', 'hscei index'   ]
ir_ids         = ['EURIRS',     'USDIRS'   ,'KRWIRS'      , 'HKDIRS'        ]

base_date_lvs = [datetime.date(2016, 3, 20), datetime.date(2016, 2, 25), datetime.date(2016, 3, 24)]
base_date_ivs = base_date_lvs
for m in range(3):
    base_date_lv = base_date_iv = base_date_ivs[m]
    for n in range(4):
        inst_id     = instrument_ids[n]

        #Day Convention
        day_flag    = 'calendar'
        divd_flag   = 'Discrete'
        ivs_tag     = 'sd_ftp'
        div_tag     = 'sd_raw'
        ir_tag      = 'sd_raw'
        db_tag      = 'sd_svi'

        #내재변동성 읽어오기
        ivs = ImpliedVolSurface.from_db(inst_id, base_date_iv, ivs_tag)
        ti = ivs.get_ti(base_date_lv)   # 연단위 tenor
        di = ivs.get_di(base_date_lv)   # 잔존일자
        vi = ivs.get_vi(base_date_lv)   # 내재변동성 matrix

        #금리정보
        ir  = SpotRateCurve.from_db(ir_id(inst_id), base_date_iv, ir_tag)

        #배당정보
        if divd_flag == "Continuous":
            div = ContinuousDividend.from_db(inst_id, base_date_iv, div_tag)
        elif divd_flag == "Discrete":
            div = DiscreteDividend  .from_db(inst_id, base_date_iv, div_tag)

        #========================SVI Projection==========================
        S = ivs.strikes[10]
        fi = ivs.get_forward(base_date_lv, S, ti, di, ir, div)      # Forward Value F

        # Step 1. Initial fit using SSVI (Heston: Problematic so far...)
        ssvi = SSVI(ivs.strikes, ti, fi, vi, ir, 'Power')
        init_val = ssvi.SSVI_calib()

        f1 = open('SVI_result_' + inst_id + base_date_iv.strftime('%Y%m%d') + '.txt', 'w')
        #tmp_str1 = "SVI"
        #for j in range(len(ivs.strikes)):
            #tmp_str1 += ' ' + str(ivs.strikes[j])
        #f1.write(tmp_str1 + '\n')

        for j in range(len(ti)):
            svi = SVI_S(ivs.strikes, ti[j], fi[j], vi[j], ir, init_val[j])
            svi.calibration()
            S = np.linspace(ivs.strikes[0], ivs.strikes[20], 21)
            k = np.log(S / fi[j])

            iv_interp = interp1d(k, vi[j], 'cubic')

            tmp = svi.get_SVI_Vol([0])
            tmp_str1 = str(ti[j])
            #for ind_strk in range(len(ivs.strikes)):
            #    tmp_str1 += ' ' + str(tmp[ind_strk])
            tmp_str1 += ' ' + str(iv_interp(0)) + ' ' + str(tmp[0])
            f1.write(tmp_str1 + '\n')
        f1.close()