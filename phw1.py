import pandas as pd
import numpy as np

# Load Data
df = pd.read_csv('D:/study/3-2/머러/lab1/data.csv', header=None) #, names=cols
# -----------------Replace \'?\' to np.NaN --------------------
df.replace({'?':np.nan}, inplace=True)

print('-----------------(Before delete) Find Missing Values--------------------')
print(df.isna().sum())
# -----------------Delete Missing Values--------------------
df.dropna(how='any', inplace=True)
print('-----------------(After delete) Find Missing Values--------------------')
print(df.isna().sum())
print(df)

X = df.drop(0, axis=1)
X = X.drop(10, axis=1)
y = df[10]
print('-----------------Split Predictor & Target Feature --------------------')
print('[Predictor Columns]')
print(X)
print('[Target Column]')
print(y)



from sklearn.model_selection import train_test_split

# Split train data and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size= 1/8, shuffle=True, random_state=42)

from sklearn.preprocessing import StandardScaler, MaxAbsScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

# Declare what i will use scalers, models with various parameters.
scalers = [StandardScaler(), MaxAbsScaler(), MinMaxScaler(), RobustScaler()]
models = {  DecisionTreeClassifier() : {
        'criterion' : ['entropy'],
        'max_depth' : [1,3,5,7,9],
        'min_samples_leaf' : [2,5,7,9]
        },
        DecisionTreeClassifier() : {
        'criterion' : ['gini'],
        'max_depth' : [1,3,5,7,9],
        'min_samples_leaf' : [2,5,7,9]
        },
    LogisticRegression() : {
        'solver' : ['newton-cg','lbfgs', 'liblinear'],
        'penalty' : ['l2'],
        'C' : [100, 10, 1.0, 0.1, 0.01]
        },
    SVC() : {
        'kernel' :['poly', 'rbf', 'sigmoid'],
        'C':[50, 10, 1.0, 0.1, 0.01],
        'gamma':['scale']
        }
}

# To save accuracy for each models for each scalers.
train_best_scores=[[0] * 4 for _ in range(4)]
test_best_scores=[[0] * 4 for _ in range(4)]

# Get an accuracy for each models for each dataset using each scaling method.
for i in scalers:
    i.fit(X_train)
    X_train = i.transform(X_train)
    X_test = i.transform(X_test)
    # print('###################{}###################'.format(i))
    # print('######## Train #########')
    # print(X_train)
    # print('######## Test #########')
    # print(X_test)
    cnt = 0 # dict type has not sequence.
    for j in models: # j is model,  models[j] is parameter
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
        grid_search = GridSearchCV(estimator=j, param_grid=dict(models[j].items()), n_jobs=-1, cv=cv, scoring='accuracy',error_score=0)
        grid_result = grid_search.fit(X_train, y_train)
        estimator = grid_result.best_estimator_
        y_pred = estimator.predict(X_test)

        # summarize results
        print("%s %s Best: %f using %s" % (i, j, grid_result.best_score_, grid_result.best_params_))

        train_best_scores[scalers.index(i)][cnt] = grid_result.best_score_
        test_best_scores[scalers.index(i)][cnt] = accuracy_score(y_test, y_pred)
        cnt +=1

print('======== Train ========')
print('[Best Score] :', np.max(train_best_scores))
print('[Worst Score] :', np.min(train_best_scores))
print('======== Test ========')
print('[Best Score] :', np.max(test_best_scores))
print('[Worst Score] :', np.min(test_best_scores))

# Visualization for result
import matplotlib.pyplot as plt

plt.figure(figsize=(10,4))
x=['Decision(Entorpy)','Decision(Gini)','Logistic','SVM']
plt.subplot(2, 1, 1)
plt.plot(x, train_best_scores[0], marker='o', lw=1.5, label='Standard Scaler')
plt.plot(x, train_best_scores[1], marker='o', lw=1.5, label='MaxAbs Scaler')
plt.plot(x, train_best_scores[2], marker='o',lw=1.5, label='MinMiax Scaler')
plt.plot(x, train_best_scores[3], marker='o', lw=1.5, label='Robust')
# plt.plot(best_scores,'ro')
plt.legend(loc=0)
plt.grid(True)


plt.xlabel('Model')
plt.ylabel('Acc')
plt.title('Train Model Acc')

plt.subplot(2, 1, 2)
plt.plot(x, test_best_scores[0], marker='o', lw=1.5, label='Standard Scaler')
plt.plot(x, test_best_scores[1], marker='o', lw=1.5, label='MaxAbs Scaler')
plt.plot(x, test_best_scores[2], marker='o',lw=1.5, label='MinMiax Scaler')
plt.plot(x, test_best_scores[3], marker='o', lw=1.5, label='Robust')

plt.grid(True)

plt.xlabel('Model')
plt.ylabel('Acc')
plt.title('Test Acc')
plt.show()
