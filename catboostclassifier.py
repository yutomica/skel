
from catboost import CatBoostClassifier

clf = CatBoostClassifier(random_seed=2019,num_boost_round=10000,eval_metric='AUC')
clf.fit(trn_data,trn_target,eval_set=(val_data,val_target), verbose_eval=100, early_stopping_rounds=200, use_best_model=True)
