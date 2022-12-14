
"""
【説明変数カテゴリ化関数】

 - 入力：
  df (pd.DataFrame)	 :分析用データフレーム
  target (str)		 :目的変数（数値）の列名称、「0」:good、「1」:badの2値
  var (str)		 :カテゴリ化する説明変数（数値）の列名称
  maxbins (int)	         :最大分割数（default = 20）
  stop_Pval (float)	 :分割ストップ時P値（default = 0.05）
  verbose (bool)         :計算過程を出力（default = True）

 - 出力：
  bins (list)		 :ビン（分割の閾値リスト）（※）
  divtbl (pd.DataFrame)  :カテゴリ化の過程を示したテーブル

 - 注意事項（※）：
  本関数は、pandas.cut関数での数値ビニングに使用する、ビン(分割の閾値リスト)を返すもの
  説明変数のカテゴリ化は、出力される分割の閾値リストとpandas.cut関数を利用して、以下のように行うこと

    bins,tbl = Categorize(input_df,'def_flg','varname')
    input_df['C_varname'] = pd.cut(input_df['varname'],bins=bins,include_lowest=True)

"""

# -*- coding: utf-8 -*-

import pandas as pd
from scipy import stats

def Categorize(df,target,var,maxbins=20,stop_Pval=0.05,verbose=True):

    inputdata = df[[target,var]].copy()

    if(verbose): 
        print("\n\nCategorizing "+var)
        print(' - length of input data = '+str(len(inputdata)))
        print(' - maxbins = '+str(maxbins))

    # エラーチェック
    if target == var:
            print(' .. Input error !!')
            return [],pd.DataFrame()
    if inputdata[var].dtype.kind not in ('i','f','u'):
    #if not any([inputdata[var].dtype == 'int',inputdata[var].dtype == 'float',inputdata[var].dtype == 'int64',inputdata[var].dtype == 'float64']):
            print(' .. Type error !!')
            return [],pd.DataFrame()
    if inputdata.isnull().sum()[target] > 0:
            print(' .. Target data contains missing value !!')
            return [],pd.DataFrame()

    # 前処理
    leaves = int(maxbins*(1 - inputdata[var].isnull().sum()/len(inputdata)))
    inputdata.loc[:,'range'] = pd.qcut(inputdata.loc[:,var],leaves,duplicates='drop')
    if(verbose):
        print(' - length of preprocessed data = '+str(len(inputdata.dropna())))
        print(' - bins = '+str(leaves))


    # 分割表作成
    divtbl = pd.merge(
            inputdata.groupby('range').mean()[[target]].rename(columns={target:'pct_bad'}),
            inputdata.groupby('range').count()[[target]].rename(columns={target:'obs'}),
            how='inner',left_index=True,right_index=True
    )
    divtbl = divtbl[divtbl['obs']>0]
    divtbl.loc[:,'obs_bad'] = divtbl.loc[:,'obs'] * divtbl.loc[:,'pct_bad']
    divtbl.loc[:,'obs_good'] = divtbl.loc[:,'obs'] - divtbl.loc[:,'obs_bad']
    divtbl.loc[:,'grid_left'] = [x.left for x in divtbl.index]
    divtbl.loc[:,'grid_right'] = [x.right for x in divtbl.index]
    
    # 2分割関数
    def divide(_divtbl):
            cntr = 1
            res = pd.DataFrame()
            while cntr < len(_divtbl):
                    tbl_0 = _divtbl.iloc[:cntr,:].copy()
                    tbl_0['label'] = 0
                    tbl_1 = _divtbl.iloc[cntr:,:].copy()
                    tbl_1['label'] = 1
                    tbl = pd.concat([tbl_0,tbl_1])
                    tbl2 = tbl.groupby('label').sum()[['obs_good','obs_bad']]
                    x2, p, dof, expected = stats.chi2_contingency(tbl2)
                    _res = pd.DataFrame({'x2':x2,'P':p},index=[_divtbl.index[cntr].left])
                    res = pd.concat([res,_res])
                    cntr += 1
            res = res[res['P'] < stop_Pval]
            if len(res) > 0:
                    res = res.sort_values('x2',ascending=False)
                    return res.index[0]
            else: return False
    
    # 分割実行
    current_node = [divtbl]
    next_node = []
    grid = []
    node_cntr = 1
    branch_cntr = 1
    cntr = 1
    if(verbose): print(' - Splitting ..')
    while True:
            if(verbose): print("    / Node = "+str(node_cntr))
            for tbl in current_node:
                    if len(tbl) > 2:
                            thresh = divide(tbl)
                    else:
                            break
                    if all([thresh,len(tbl[tbl['grid_right'] <= thresh])>0,len(tbl[tbl['grid_right'] > thresh])>0]):
                            grid.append(thresh)
                            divtbl.loc[tbl[tbl['grid_right'] <= thresh].index,'N_'+str(node_cntr)] = cntr
                            divtbl.loc[tbl[tbl['grid_right'] > thresh].index,'N_'+str(node_cntr)] = cntr + 1
                            cntr += 2
                            if len(tbl[tbl['grid_right'] <= thresh]) > 1:
                                next_node.append(tbl[tbl['grid_right'] <= thresh])
                            if len(tbl[tbl['grid_right'] > thresh]) > 1:
                                next_node.append(tbl[tbl['grid_right'] > thresh])
                    branch_cntr += 1
            if len(next_node) == 0:break
            current_node = next_node
            next_node = []
            node_cntr += 1
            branch_cntr = 1

    bins = [inputdata[var].min()] + grid + [inputdata[var].max()]
    if(verbose): print(' .. Mission complete !!')

    return sorted(bins),divtbl

if __name__ == '__main__':

    # タイタニック号データでテスト
    df = pd.read_csv('titanic.csv')

    for col in df.columns:
        bins,tbl = Categorize(df,'Survived',col)
        if len(bins)>0:
            df['C_'+col] = pd.cut(df[col],bins=bins,include_lowest=True)
            tbl.to_excel(col+'.xlsx')

