"""
【説明変数間相関分析による変数選択プログラム】

・処理概要：
　説明変数間の相関係数を算出し、全変数間の相関係数が閾値以下となる変数を特定し選択
　相関係数が閾値以上になる変数ペアについては、判別力（AUC）が高い方を優先的に選択

・入力：
　- df (pandas.DataFrame)   :分析対象データフレーム
  - target (str)            :目的変数の名称、上記データフレームに含まれること、目的変数自体は0(non event)or1(event)の数値変数
  - thresh (float)          :相関係数の閾値（デフォルト0.3）
  - MODE_DEBUG (True/False) :デバッグ実行有無（デフォルトFalse）、デバッグモードにすると出力内容が変わる

・出力：
  - 選択結果となる変数名のリスト
  - (デバッグモードの場合) リスト①,リスト②,分析用テーブルのデータフレーム
  　リスト①　全変数間の相関係数が閾値以下の変数名リスト
  　リスト②　相関係数が閾値以上になる変数ペアのうち、選択された変数名リスト

"""

import pandas as pd
from itertools import product
from math import fabs
from sklearn.metrics import roc_auc_score

def SelectVars(df,target,thresh=0.3,MODE_DEBUG=False):

    print('')
    print('executing SelectVars ..')
    print('')

    print(' - shape of input dataframe :')
    print('   '+str(df.shape[0])+'obs,'+str(df.shape[1])+'cols')
    print('')
    
    # 文字変数と欠損レコードの除去
    cols_char = [x for x in df.columns if df[x].dtype == 'object']
    df = df.drop(cols_char,axis=1)
    df = df.dropna()
    print(' - shape of preprocessed dataframe :')
    print('   '+str(df.shape[0])+'obs,'+str(df.shape[1])+'cols')
    print('')

    # 相関係数の算出
    corrs = df.drop([target],axis=1).corr()
    corrs_ = corrs[(corrs < thresh)&(corrs > -1*thresh)]
    print(' - selelct variables from '+str(corrs.shape[0])+' candidates ..')
    print('')

    # 全変数との相関係数が閾値以下である変数の特定 -> cols_independent
    cols_independent = list(corrs_[corrs_.isnull().sum()==1].index)
    df = df.drop(cols_independent,axis=1)
    print(' - cols independent : '+str(len(cols_independent)))
    for c in cols_independent:
        print('   '+c)
    print('')

    # 相関係数が閾値以上の変数ペアについて、AUCが高い方を優先的に残す選択処理を実施 -> cols_dependent
    cols_dependent = list()
    output = pd.DataFrame()
    if len(cols_independent) < corrs.shape[0]:
        # 判別力分析
        tbl_auc = pd.DataFrame()
        for col in df.drop([target],axis=1).columns:
            _tbl_auc = pd.DataFrame({
                'AUC':max(roc_auc_score(df[target],df[col]),1-roc_auc_score(df[target],df[col]))
            },index=[col])
            tbl_auc = pd.concat([tbl_auc,_tbl_auc])

        # 分析用テーブル(selectvars)の作成
        selectvars = pd.DataFrame()
        for v1,v2 in product(list(tbl_auc.index),list(tbl_auc.index)):
            if v1!=v2:
                _selectvars = pd.DataFrame({'var_keep':v1,'var_delete':v2,'auc_keep':tbl_auc.loc[v1,'AUC'],'auc_delete':tbl_auc.loc[v2,'AUC'],'corr':fabs(corrs.loc[v1,v2])},index=[0])
                selectvars = pd.concat([selectvars,_selectvars])
            else:
                pass

        selectvars = selectvars.sort_values(by='auc_keep',ascending=False)
        if(MODE_DEBUG): output = selectvars.copy()
        while(len(selectvars)>0):
            _keep = selectvars.iloc[0]['var_keep']
            _delete = list(selectvars[(selectvars['var_keep']==_keep)&(selectvars['corr']>thresh)]['var_delete'].values)
            # _keep変数を削除
            selectvars = selectvars[selectvars['var_keep']!=_keep]
            # _keep変数と相関が高い変数（_delete）を削除
            selectvars = selectvars[~selectvars['var_keep'].isin(_delete)]
            cols_dependent += [_keep]
    else:
        pass

    print(' - cols dependent : '+str(len(cols_dependent)))
    for c in cols_dependent:
        print('   '+c)
    print('')

    # 出力
    print(' .. Mission complete !!')
    if(MODE_DEBUG):
        return cols_independent,cols_dependent,output
    else:
        return cols_independent + cols_dependent


if __name__ == '__main__':
    # タイタニック号のデータで検証
    df = pd.read_csv('titanic.csv')
    cols_selected = SelectVars(df,'Survived')
    a,b,c = SelectVars(df,'Survived',MODE_DEBUG=True)

