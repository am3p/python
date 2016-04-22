# -*- coding: utf-8 -*-
import datetime
import numpy as np

from miraepy.derivatives.params.impliedvol_all import ImpliedVolSurface #, ImpliedVolTS
from miraepy.derivatives.params.dividend_all import DiscreteDividend, ContinuousDividend
from miraepy.derivatives.params.interestrate import SpotRateCurve
from miraepy.common.code import ir_id #ticker # , get_all_from_db, get_code_from_db , get_ticker_from_db, ticker_to_name_from_db

from SVI_Projection import SVI_S

instrument_ids = ['SX5E',       'SPX'      ,'KOSPI200'    , 'HSCEI'         ]
tickers        = ['sx5e index', 'spx index','kospi2 index', 'hscei index'   ]
ir_ids         = ['EURIRS',     'USDIRS'   ,'KRWIRS'      , 'HKDIRS'        ]

base_date_lv = base_date_iv = datetime.date(2016, 2, 11)
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

    var_svi = np.zeros((len(ti),len(ivs.strikes)))

    f1 = open('SVI_result_' + inst_id + base_date_iv.strftime('%Y%m%d') + '.txt','w')
    f2 = open('SVI_parameter_' + inst_id + base_date_iv.strftime('%Y%m%d') + '.txt','w' )
    tmp_str1 = "SVI"
    tmp_str2 = ""
    for i in range(len(ivs.strikes)):
        tmp_str1 += ' ' + str(ivs.strikes[i])
    f1.write(tmp_str1 + '\n')

    # SVI calibration for each tenor
    for i in range(len(ti)):
        svi = SVI_S(ivs.strikes, ti[i], fi[i], vi[i], ir, 'vega')
        svi.calibration()

        S = np.linspace(ivs.strikes[0], ivs.strikes[20], 21)
        k = np.log(S/fi[i])
        tmp = svi.get_SVI_Vol(k)
        tmp_str1 = tmp_str2 = str(ti[i])
        for j in range(len(ivs.strikes)):
            tmp_str1 += ' ' + str(tmp[j])
        opt_param = svi.get_SVI_Param()
        for j in range(len(opt_param)):
            tmp_str2 += ' ' + str(opt_param[j])
        f1.write(tmp_str1 + '\n')
        f2.write(tmp_str2 + '\n')

    f1.close()
    f2.close()
