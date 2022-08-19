from sklearn import manifold
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from dataload import get_data
import numpy as np
from sklearn.metrics import recall_score,roc_auc_score,f1_score

def get_metrics(y_true,y_pred):
    y_pred = np.array(y_pred)
    recall_s = recall_score(y_true=y_true,y_pred=y_pred)
    f1_s = f1_score(y_true=y_true,y_pred=y_pred)
    auc = roc_auc_score(y_true=y_true,y_score=y_pred)
    fpr = get_fpr(y_true=y_true, y_pred=y_pred)
    return (recall_s,f1_s,auc,fpr)

def get_fpr(y_true,y_pred):
    y_pred_int = np.round(y_pred)
    y_pred_int = y_pred_int.reshape(-1,)
    fp = np.sum(np.multiply((1-y_true),y_pred_int))
    tn = np.sum(np.multiply((1-y_true),(1-y_pred_int)))
    return fp/(fp+tn)

if __name__ == "__main__":
    data, labels = get_data()
    # X, color = datasets.make_swiss_roll(n_samples=1500)
    print("Computing LLE embedding")
    X_r, err = manifold.locally_linear_embedding(data, n_neighbors=80, n_components=50)
    train_x,test_x,train_y,test_y = train_test_split(
        X_r,labels,test_size=0.3,random_state=111,stratify=labels
    )
    print("model start ... ")
    scaler = MinMaxScaler()
    train_x_scal = scaler.fit_transform(X=train_x)
    test_x_scal = scaler.transform(X=test_x)
    params = {
        "C":[1.0,10.0,100.0,1000.0],
        "gamma":[1.0,2.0,5.0,10.0,100.0]
    }
    # C=10.0,gamma=2.0
    svm = SVC(random_state=111,kernel='rbf',class_weight={1:8})
    clf = GridSearchCV(svm,param_grid=params,cv=5)
    clf.fit(X=train_x_scal,y=train_y)
    print(clf.best_params_)
    svm_ = clf.best_estimator_
    pred_res = svm_.predict(test_x_scal)
    recall_s, f1_s, auc,fpr = get_metrics(y_true=test_y,y_pred=pred_res)
    print(recall_s)
    print(f1_s)
    print(auc)
    print(fpr)
    '''
    svm = SVC(random_state=111)
    clf = Pipeline([('scaler',scaler),('svm',svm)])
    params = {
        'svm_gamma':[1e-4,1e-3,1e-2,1e-1,1,10],
        'svm_C':[1e-3,1e-2,1e-1,1,10,100],
        'svm_kernel':['rbf'],
    }
    search = GridSearchCV(clf,params,refit=True,cv=5,n_jobs=-1)
    search.fit(train_x,train_y)
    print(search.best_params_, search.best_score_)
    model = search.best_estimator_
    print(search.score(test_x, test_y))
    '''


'''
X, color = datasets.make_swiss_roll(n_samples=1500)

print("Computing LLE embedding")
X_r, err = manifold.locally_linear_embedding(X, n_neighbors=12, n_components=2)
print("Done. Reconstruction error: %g" % err)

# ----------------------------------------------------------------------
# Plot result

fig = plt.figure(figsize=(8,16))

ax = fig.add_subplot(211, projection="3d")
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
'''
''''
ax.set_title("Original data")
ax = fig.add_subplot(212)
ax.scatter(X_r[:, 0], X_r[:, 1], c=color, cmap=plt.cm.Spectral)
plt.axis("tight")
plt.xticks([]), plt.yticks([])
plt.title("Projected data")
plt.show()
'''