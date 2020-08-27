
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import BaggingClassifier
from sklearn.model_selection import StratifiedKFold


clf = LogisticRegressionCV(cv=5,random_state=2019)
oof_blr = np.zeros(len(target))
predictions_blr = np.zeros(len(test))

folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=2019)
for fold_, (trn_idx, val_idx) in enumerate(folds.split(train,target)):
    print("fold : "+str(fold_))
    trn_X = train.iloc[trn_idx,:]
    trn_y = target.iloc[trn_idx]
    val_X = train.iloc[val_idx,:]
    val_y = target.iloc[val_idx]
    model = BaggingClassifier(base_estimator=clf,n_estimators=100,random_state=2019)
    model = model.fit(trn_X,trn_y)
    oof_blr[val_idx] = model.predict_proba(val_X)[:,1]
    predictions_blr += model.predict_proba(test)[:,1]/folds.n_splits
    print(" - CV score: {:<8.5f}".format(roc_auc_score(val_y,oof_blr[val_idx])))

print("CV score: {:<8.5f}".format(roc_auc_score(target,oof_blr)))
