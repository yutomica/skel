

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV

model = RandomForestClassifier()
model = model.fit(train,target)
print("score: {:<8.5f}".format(roc_auc_score(target,model.predict_proba(train)[:,1])))
print("score/val: {:<8.5f}".format(roc_auc_score(val,model.predict_proba(validation)[:,1])))

model = RandomForestClassifier()
params = {
    "n_estimators":[i for i in range(10,100,10)],
    "criterion":["gini","entropy"],
    "max_depth":[i for i in range(1,6,1)],
    'min_samples_split': [2, 4, 10,12,16],
    "random_state":[3]
}
gs = GridSearchCV(model,params,scoring='roc_auc',cv=3,verbose=1)
gs = gs.fit(train,target)
