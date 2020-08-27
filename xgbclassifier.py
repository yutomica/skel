
from xgboost import XGBClassifier

xgb_params = {
    'eta': 0.05,
    'max_depth': 4,
    'min_child_weight':50,
    'subsample': 0.7,
    'colsample_bytree': 0.7,
    'lambda':1,
    'alpha':0,
    'objective':'binary:logistic',
    'eval_metric': 'auc',
    'silent': True,
    'n_estimators':10000,
    'random_state':2019
}

model = XGBClassifier(**xgb_params)
model = model.fit(trn_data,trn_target,eval_set=[(val_data,val_target)],early_stopping_rounds=200,verbose=100)
oof_xgb[val_idx] = clf_xgb.predict_proba(train.iloc[val_idx][features])[:,1]
predictions_xgb += clf_xgb.predict_proba(test[features])[:,1] / folds.n_splits

print("oof score _ cat : {:<8.5f}".format(roc_auc_score(target,oof_cat)))
print("oof score _ xgb : {:<8.5f}".format(roc_auc_score(target,oof_xgb)))
print("oof score _ lgbm : {:<8.5f}".format(roc_auc_score(target,oof_lgbm)))
