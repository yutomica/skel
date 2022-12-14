#encoding utf-8

"""

入力：
　- df(pandas.DataFrame) : 分析に使用するデータフレーム
　- target(str)：目的変数名
　- col(str)：説明変数（AR値算出対象変数）名
  - CapGrid(int):CAP曲線のグリッド数　

出力：
　AR,OBS,OBS_1,CAP

  AR(float) : AR値（計算結果）
  OBS(float)：AR値を計算するのに使用したデータ行数
  OBS_1(float)：うち、不良件数
  CAP(pandas.DataFrame):CAP曲線のデータフレーム

"""

import numpy as np
import pandas as pd
import os

def CalcAR(df, target, col, CapGrid=200):
    df_calcar01 = df.copy()
    
    df_calcar01 = df_calcar01.rename(columns={target:'target'})
    df_calcar01 = df_calcar01.rename(columns={col:'col'})
    df_calcar01 = df_calcar01.dropna(subset=['target'])
    df_calcar01 = df_calcar01[['target','col']]
    

    if df[col].dtype.name == 'object' or df[col].dtype.name == 'category':
    #if df[col].dtype.name == 'category':
        GoodObs = sum(df_calcar01['target'] == 0)
        BadObs = sum(df_calcar01['target'] == 1)
        TotObs = GoodObs + BadObs
        df_calcar01.loc[:,'var_char'] = [str(x) for x in df_calcar01['col']]
        _char = df_calcar01.groupby('var_char').mean()
        df_calcar01 = pd.merge(df_calcar01.drop(['target'],axis=1),_char,left_on='var_char',right_index=True)
        df_calcar01 = df_calcar01.sort_values(by='target',ascending=False)
    else:
        df_calcar01 = df_calcar01.dropna(subset=['col'])
        df_calcar01 = df_calcar01.sort_values(by='col')
        df_calcar01['order'] = [int(x*2/len(df_calcar01)) for x in range(len(df_calcar01))]
        order = df_calcar01.groupby('order').mean()[['target']].idxmax().values[0]
        #条件によってソート順を変える
        if order == 1:
            df_calcar01 = df_calcar01.sort_values('col', ascending=False)
        else:
            df_calcar01 = df_calcar01.sort_values('col')
        GoodObs = sum(df_calcar01['target'] == 0)
        BadObs = sum(df_calcar01['target'] == 1)
        TotObs = GoodObs + BadObs

    #インデックスのリセット
    df_calcar01.reset_index(inplace=True, drop=True)

    #件数
    df_calcar03 = df_calcar01
    df_calcar03['No'] = pd.RangeIndex(start=1, stop=len(df_calcar03.index) + 1, step=1)
    df_calcar03.loc[df_calcar03['No'] <= BadObs, 'Cnt_I'] = df_calcar03['No']
    df_calcar03.loc[~(df_calcar03['No'] <= BadObs), 'Cnt_I'] = BadObs
    df_calcar03['Cnt_A'] = df_calcar03['target'].cumsum()
    df_calcar03['Cnt_R'] = df_calcar03['No'] * (BadObs / TotObs)

    #CAP曲線の縦軸に変換
    df_calcar03['y_I'] = df_calcar03['Cnt_I']/BadObs
    df_calcar03['y_A'] = df_calcar03['Cnt_A']/BadObs
    df_calcar03['y_R'] = df_calcar03['Cnt_R']/BadObs

    #微小面積の計算
    df_calcar03['ydx_I'] = (1/TotObs)*df_calcar03['y_I']
    df_calcar03['ydx_A'] = (1/TotObs)*df_calcar03['y_A']
    #df_calcar03['ydx_R'] = (1/TotObs)*df_calcar03['y_R']

    S_I = df_calcar03['ydx_I'].sum()
    S_A = df_calcar03['ydx_A'].sum()
    AR = (S_A - 0.5)/(S_I - 0.5)
    #df_calcar05 = pd.DataFrame({'AR':AR,'OBS':TotObs,'OBS_1':BadObs},index=[col])

    df_calcar03['grid'] = [int((x-1) * CapGrid / TotObs) for x in df_calcar03['No']]
    df_calcar03 = df_calcar03[['grid', 'y_I', 'y_A', 'y_R']]
    df_calcar03 = df_calcar03.groupby('grid').max()
    df_calcar03.columns = ['CAP_Ideal', 'CAP_Actual', 'CAP_Random']
   
    return AR,TotObs,BadObs,df_calcar03

if __name__ == '__main__':

    df = pd.read_csv('titanic.csv')
    output = pd.DataFrame()
    for col in df.columns.drop(['Survived']):
        print(col)
        AR,OBS,OBS_1,captable = CalcAR(df,'Survived',col)
        _output = pd.DataFrame({'OBS':OBS,'OBS_1':OBS_1,'AR':AR},index=[col])
        output = pd.concat([output,_output])
        captable.to_csv('cap_'+col+'.csv')
    output.to_csv('AR_list.csv')

