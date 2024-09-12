# -*- coding: utf-8 -*-
"""

@author: sarvio valente

"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import func
from skopt import gp_minimize
from sklearn.ensemble import RandomForestClassifier

#%% importar o dataset

df = pd.read_csv('./semana2/dataset_problema2.csv')

X = df.drop(['id', 'Severidade'], axis = 1)
y = df['Severidade']

#%% separar dados de treinamento e dados de teste

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#%% padronizar os dados

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() #media 0 e desvio padrão 1

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)

X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_train.columns


#%% testar nos dados de teste

def avaliar_modelo(modelo, X_train_sc, X_test_sc):
    from sklearn.metrics import mean_squared_error
    from sklearn.metrics import mean_absolute_error
    
    X_train_sel = X_train_sc[sel_features]
    X_test_sel = X_test_sc[sel_features]

    X_train_sel = X_train_sc[sel_features]
    modelo.fit(X_train_sel, y_train)
    
    y_pred = modelo.predict(X_test_sel)
    
    r2 = modelo.score(X_test_sel, y_test)
    
    rmse = (mean_squared_error(y_test, y_pred)**0.5)
    
    mae = mean_absolute_error(y_test, y_pred)
    
    print('r2', r2)
    print('rmse', rmse)
    print('mae', mae)



#%% seleção de features
from sklearn.feature_selection import RFE
from sklearn.model_selection import cross_val_score

#%% modelo final - Regressão linear multipla
#r2 0.9059954611372723
#rmse 6.8960222700020335
#mae 5.306167646435706

from sklearn.linear_model import LinearRegression

modelo_linear = LinearRegression()

lista, sel_features = func.select_features(model=modelo_linear
                                           , X_train=X_train_sc
                                           , y_train=y_train, max_f=18
                                           , importance_getter='auto')


avaliar_modelo(modelo_linear, X_train_sc, X_test_sc)


#%% SVR
#r2 0.7616870144813807
#rmse 10.979892733740364
#mae 8.54888888888889

from sklearn.tree import DecisionTreeRegressor

model_svr = DecisionTreeRegressor(random_state=0)


lista, sel_features = func.select_features(model=model_svr, X_train=X_train_sc, y_train=y_train, max_f=12, importance_getter='auto')

avaliar_modelo(model_svr,  X_train_sc, X_test_sc)

#%% Score:  0.8757009686752486
#r2 0.8757009686752486
#rmse 7.929724805232801
#mae 6.633255684256221

from sklearn.svm import SVR
model_svr = SVR(kernel="linear")
lista, sel_features = func.select_features(model=model_svr, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')


avaliar_modelo(model_svr,  X_train_sc, X_test_sc)


#%%
#r2 0.8413838500696946
#rmse 8.957729540375649
#mae 7.1140740740740736
from sklearn.tree import ExtraTreeRegressor
model_extra_tree = ExtraTreeRegressor(random_state=0)

lista, sel_features = func.select_features(model=model_extra_tree, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')


avaliar_modelo(model_extra_tree,  X_train_sc, X_test_sc)

#%%
#r2 0.8757783081486816
#rmse 7.927257464334654
#mae 6.540065719838723
from sklearn.linear_model import SGDRegressor
model_clf = SGDRegressor(max_iter=1000, tol=1e-3)

lista, sel_features = func.select_features(model=model_clf, X_train=X_train_sc, y_train=y_train, max_f=14, importance_getter='auto')


avaliar_modelo(model_clf,  X_train_sc, X_test_sc)

#%%
# r2 0.8827456534898017
# rmse 7.701737616411041
# mae 6.64822970603713
from sklearn.linear_model import ElasticNet

model_en = ElasticNet(random_state=0)

lista, sel_features = func.select_features(model=model_en, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')


avaliar_modelo(model_en,  X_train_sc, X_test_sc)

#%%
#r2 0.8923065620559525
#rmse 7.381062186505593
#mae 6.295207686188805

#Cross-validated Least Angle Regression model.
from sklearn.linear_model import LarsCV
model_reg = LarsCV(cv=5)

lista, sel_features = func.select_features(model=model_reg, X_train=X_train_sc, y_train=y_train, max_f=9, importance_getter='auto')

avaliar_modelo(model_reg,  X_train_sc, X_test_sc)

#%%
#r2 0.8957464122482819
#rmse 7.262225806826695
#mae 6.013775692742941

from sklearn import linear_model
model_lasso = linear_model.Lasso(alpha=0.1)

lista, sel_features = func.select_features(model=model_lasso, X_train=X_train_sc, y_train=y_train, max_f=12, importance_getter='auto')


avaliar_modelo(model_lasso,  X_train_sc, X_test_sc)

#%%
#r2 0.8954021580674149
#rmse 7.274206167432006
#mae 5.873569807645327
from sklearn.linear_model import LassoCV
model_lassoCV = LassoCV(cv=5, random_state=0)
lista, sel_features = func.select_features(model=model_lassoCV, X_train=X_train_sc, y_train=y_train, max_f=12, importance_getter='auto')

avaliar_modelo(model_lassoCV,  X_train_sc, X_test_sc)

#%%
#r2 0.8959969120274949
#rmse 7.253495747281158
#mae 5.7751780736091565
from sklearn import linear_model
model_lassoLars = linear_model.LassoLars(alpha=0.01)
lista, sel_features = func.select_features(model=model_lassoLars, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')


avaliar_modelo(model_lassoLars,  X_train_sc, X_test_sc)


#%%

#r2 0.895984753493283
#rmse 7.253919721706053
#mae 5.8728709437268956
from sklearn.linear_model import LassoLarsCV

model = LassoLarsCV(cv=5)

lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=12, importance_getter='auto')


avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.9026334691168716
#rmse 7.018254163210872
#mae 5.782227855022274
model = linear_model.LassoLarsIC(criterion='bic')
lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=8, importance_getter='auto')

avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.8487273338694804
#rmse 8.747913383944146
#mae 6.605836465566188
from sklearn.linear_model import OrthogonalMatchingPursuit
model = OrthogonalMatchingPursuit()

lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')

avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.880945013483763
#rmse 7.760648950210742
#mae 6.219919144850497
from sklearn.linear_model import OrthogonalMatchingPursuitCV
model = OrthogonalMatchingPursuitCV(cv=5)

lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto', start=2)

avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.9024334903565776
#rmse 7.025457777424136
#mae 5.825651057668385
from sklearn import linear_model
model = linear_model.ARDRegression()

lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')

avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.8949596294240643
#rmse 7.28957764621814
#mae 5.758890697898146
from sklearn import linear_model
model = linear_model.BayesianRidge()
lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=11, importance_getter='auto')

avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.910325579792539
#rmse 6.735324624173553
#mae 5.38651986024368
from sklearn.linear_model import HuberRegressor
model = HuberRegressor(max_iter=1000)
lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=14, importance_getter='auto')

avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.781850911974344
#rmse 12.772525540364795
#mae 9.993444669470378
model = linear_model.PoissonRegressor()
lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')

avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.8741868076639172
#rmse 7.977876972169887
#mae 6.82601898066702
model = linear_model.TweedieRegressor()
lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')

avaliar_modelo(model,  X_train_sc, X_test_sc)

#%%
#r2 0.842760441018416
#rmse 8.9187738575472
#mae 6.976543952504597
model = linear_model.PassiveAggressiveRegressor(max_iter=100, random_state=0,tol=1e-3)
lista, sel_features = func.select_features(model=model, X_train=X_train_sc, y_train=y_train, max_f=20, importance_getter='auto')

avaliar_modelo(model,  X_train_sc, X_test_sc)



















































