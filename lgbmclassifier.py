
from lightgbm import LGBMClassifier

lgbm_params = {
    'n_estimators':10000,
    'bagging_freq': 5,
    'bagging_fraction': 0.4,
    'boost_from_average':'false',
    'boost': 'gbdt',
    'feature_fraction': 0.05,
    'learning_rate': 0.01,
    'max_depth': -1,  
    'metric':'auc',
    'min_data_in_leaf': 80,
    'min_sum_hessian_in_leaf': 10.0,
    'num_leaves': 13,
    'num_threads': 8,
    'tree_learner': 'serial',
    'objective': 'binary', 
    'verbosity': 1,
    'random_state':2019
}

clf_lgbm = LGBMClassifier(**lgbm_params)
clf_lgbm = clf_lgbm.fit(trn_data,trn_target,eval_set = (val_data,val_target),verbose=100,early_stopping_rounds=200)
