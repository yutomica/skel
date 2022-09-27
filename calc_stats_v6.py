import numpy as np
import pandas as pd
import math
from scipy.stats import f_oneway

# DFのサマリ
def calc_summary(df):
    print("calculating statistic values ..")
    res = pd.DataFrame()
    for col in df.columns:
        print(" - "+col)
        if any([df[col].dtype == 'object',df[col].dtype == 'bool',df[col].dtype == 'datetime64[ns]',df[col].dtype == '<M8[ns]']):
            _tmp = pd.DataFrame({
                u'データ型':df[col].dtype,
                u'件数':len(df),
                u'欠損率':len(df[df[col].isna()])/len(df),
                u'水準数':len(df[col].value_counts(dropna=False)),
                u'最小値':None,
                u'5%':None,
                u'25%':None,
                u'50%':None,
                u'75%':None,
                u'95%':None,
                u'最大値':None,
                u'平均値':None,
                u'平均値(0除き)':None,                
                u'標準偏差':None
            },columns=[u'データ型',u'件数',u'欠損率',u'水準数',u'最小値',u'5%',u'25%',u'50%',u'75%',u'95%',u'最大値',u'平均値',u'平均値(0除き)',u'標準偏差'],index=[col])
            res = pd.concat([res,_tmp])
        else:
            _tmp = pd.DataFrame({
                u'データ型':df[col].dtype,
                u'件数':len(df),
                u'欠損率':len(df[df[col].isna()])/len(df),
                u'水準数':None,
                u'最小値':df[col].min(),
                u'5%':df[col].quantile(0.05),
                u'25%':df[col].quantile(0.25),
                u'50%':df[col].quantile(0.5),
                u'75%':df[col].quantile(0.75),
                u'95%':df[col].quantile(0.95),
                u'最大値':df[col].max(),
                u'平均値':sum(df[col].dropna())/len(df[col].dropna()),                
                u'平均値(0除き)':sum(df[col].dropna())/len(df[df[col]!=0][col].dropna()),
                u'標準偏差':np.std(list(df[col].dropna()))
            },columns=[u'データ型',u'件数',u'欠損率',u'水準数',u'最小値',u'5%',u'25%',u'50%',u'75%',u'95%',u'最大値',u'平均値',u'平均値(0除き)',u'標準偏差'],index=[col])
            res = pd.concat([res,_tmp])
    return res


# 相関比
def calc_ESQ(df,var_num,var_char,dropna=True):

    _df = df[[var_num,var_char]].copy()
    if not dropna:
        _df[var_char] = _df[var_char].fillna("NA")
    _df = _df.dropna()
    print("Calculating ESQ -- "+var_num+"*"+var_char)

    if len(_df) > 0 :
        _df = pd.merge(_df,_df.groupby(var_char).mean().rename(columns={var_num:'AVE'}),how='inner',left_on=var_char,right_index=True)
        _df['SS'] = (_df[var_num]-_df['AVE'])*(_df[var_num]-_df['AVE'])
        SW = _df['SS'].sum()
        AVE_ALL = _df[var_num].sum()/len(_df)
        _df2 = pd.merge(
                _df[[var_char,var_num]].groupby(var_char).count().rename(columns={var_num:'CNT'}),
                _df[[var_char,var_num]].groupby(var_char).mean().rename(columns={var_num:'AVE'}),
                how = 'inner',left_index=True,right_index=True
        )
        _df2['SS'] = (_df2['AVE'] - AVE_ALL)*(_df2['AVE']-AVE_ALL)*_df2['CNT']
        SB = _df2['SS'].sum()   
        ESQ = math.sqrt(SB/(SW+SB)) if any([SW!=0,SB!=0]) else 0.0
        return ESQ,len(_df)/len(df)
    else: return 0.0,0.0


# 連関係数
def calc_CV(df,var_a,var_b,dropna=True):

    def cramersV(x, y):
        table = np.array(pd.crosstab(x, y)).astype(np.float32)
        n = table.sum()
        colsum = table.sum(axis=0)
        rowsum = table.sum(axis=1)
        expect = np.outer(rowsum, colsum) / n
        chisq = np.sum((table - expect) ** 2 / expect)
        return np.sqrt(chisq / (n * (np.min(table.shape) - 1)))

    _df = df[[var_a,var_b]].copy()
    if not dropna:
        _df[var_a] = _df[var_a].fillna("NA_A")
        _df[var_b] = _df[var_b].fillna("NA_B")
    _df = _df.dropna()
    print("Calculating CramersV -- "+var_a+"*"+var_b)
    cv = cramersV(_df[var_a],_df[var_b])

    return cv,len(_df)/len(df)


# ANOVA_1D
def anova_1d(df,var_num,var_char,dropna=True):

    _df = df[[var_num,var_char]].copy()
    if not dropna:
        _df[var_char] = _df[var_char].fillna("NA")
    _df = _df.dropna()
    print("Executing 1D Anova -- "+var_a+"*"+var_b)
    _input = list()
    for level in list(_df[var_char].value_counts(dropna=dropna).index):
        _input.append(list(_df[_df[var_char] == level][var_num].values))
    anova = f_oneway(*_input)
    
    return anova[0],anova[1],len(_df)/len(df)


