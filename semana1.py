# -*- coding: utf-8 -*-
"""
Created on Tue Mar 22 15:28:33 2022

@author: sarvi0
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import StandardScaler
import seaborn as sns

from skopt import gp_minimize

#baseline 76.555%
#modelo 3 76.794%


#%% abrir o datase de treino e teste

train = pd.read_csv('./semana1/train.csv')
test = pd.read_csv('./semana1/test.csv')


#%% pre-processamento dos dados

#descrição estátistica das features núméricas
est = train.describe()

print(train.info())

#verificar valores nulos ou NAN
print(train.isnull().sum())

print(test.isnull().sum())

#mapear as colunas
col = pd.Series(list(train.columns))

X_train = train.drop(['PassengerId', 'Survived'], axis = 1)

X_test = test.drop(['PassengerId'], axis = 1)


#%% Criar novas features segunda versão
def extract_title(name):
    if 'Mr.' in name:
        return 'Mr'
    elif 'Mrs.' in name:
        return 'Mrs'
    elif 'Miss.' in name:
        return 'Miss'
    elif 'Master.' in name:
        return 'Master'
    else:
        return 'Other'

def family_size(sibsp, parch):
    return sibsp + parch + 1

def fare_per_person(fare, family_size):
    return fare / family_size if family_size > 0 else fare

def extract_deck(cabin):
    if pd.isna(cabin):
        return 'Unknown'
    else:
        return cabin[0]

def age_bin(age):
    if pd.isna(age):
        return 'Unknown'
    elif age <= 12:
        return 'Child'
    elif age <= 18:
        return 'Teenager'
    elif age <= 35:
        return 'Young Adult'
    elif age <= 50:
        return 'Adult'
    else:
        return 'Senior'

def fare_bin(fare):
    if fare <= 10:
        return 'Low'
    elif fare <= 50:
        return 'Medium'
    elif fare <= 100:
        return 'High'
    else:
        return 'Very High'

def is_alone(family_size):
    return 1 if family_size == 1 else 0

def criar_features2(df):
    subs = {'female':1, 'male':0}
    df['mulher'] = df['Sex'].replace(subs)
    
    df['Fare'] = df['Fare'].fillna(df['Fare'].mean())
    
    df['Age'] = df['Age'].fillna(df['Age'].mean())
    
    df['Embarked'] = df['Embarked'].fillna('S')
    
    subs = {'S':1, 'C':2, 'Q':3}
    df['porto'] = df['Embarked'].replace(subs)
    df['Title'] = df['Name'].apply(extract_title)
    df['FamilySize'] = df.apply(lambda row: family_size(row['SibSp'], row['Parch']), axis=1)
    df['FarePerPerson'] = df.apply(lambda row: fare_per_person(row['Fare'], row['FamilySize']), axis=1)
    df['Deck'] = df['Cabin'].apply(extract_deck)
    df['AgeBin'] = df['Age'].apply(age_bin)
    df['FareBin'] = df['Fare'].apply(fare_bin)
    df['IsAlone'] = df['FamilySize'].apply(is_alone)
    
    # Codificar variáveis categóricas usando One-Hot Encoding
    df = pd.get_dummies(df, columns=['Title', 'Deck', 'AgeBin', 'FareBin', 'Embarked'], drop_first=True)
    
    # Remover colunas que não são úteis para o modelo
    df.drop(['Name', 'Ticket', 'Cabin', 'Sex'], axis=1, inplace=True)
    
    return df

X_train = criar_features2(X_train)
X_test = criar_features2(X_test)
# Adicionando uma feature faltante no conjunto de teste
X_test['Deck_T'] = False

# Ordenando as colunas do conjunto de teste para ficar na mesma ordem do conjunto de treino
X_test = X_test[X_train.columns]

#%% Selecionar Features
def select_features(model, X_tain, y_train, importance_getter='auto', max_f = 20, step = 1):
    from sklearn.feature_selection import RFE
    
    lista_score = list()
    for i in range(1, max_f +1):
    
        selector = RFE(model, n_features_to_select = i, step = step)
        selector = selector.fit(X_train, y_train)
        mask = selector.support_
    
        features = X_train_sc.columns
    
        sel_features = features[mask]
    
        X_sel = X_train_sc[sel_features]
        score = cross_val_score(model, X_sel, y_train, cv = 10)
        
        print("Iteração ",i, ". Score: ", np.mean(score))
        
        lista_score.append(np.mean(score))
    return lista_score, sel_features

#%% Selecionar as features
"""
features = ['Title_Mr', 'Title_Mrs', 'Title_Miss', 'mulher', 'Fare', 'FarePerPerson', 'IsAlone', 'Deck_Unknown'
            , 'Deck_B', 'Pclass', 'FareBin_Low', 'Deck_C', 'AgeBin_Child', 'FareBin_Very High', 'Embarked_S'
            , 'Deck_E']

X_train = X_train[features]
X_test = X_test[features]

y_train = train['Survived']
"""


#%% Coeficiente de correlação
def analisar_correlacao(df):
    """
    Calcula o coeficiente de correlação entre as features do dataset
    e plota um heatmap das correlações.
    
    Parameters:
    df (pd.DataFrame): DataFrame com as features do dataset.
    
    Returns:
    corr_matrix (pd.DataFrame): Matriz de correlação das features.
    """
    # Calcula a matriz de correlação
    corr_matrix = df.corr()
    corr_matrix2 = corr_matrix['Survived']

    # Exibe a matriz de correlação
    print(corr_matrix2)
    #return

    # Plotar um heatmap para visualizar as correlações
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Heatmap de Correlação das Features")
    plt.show()

    return corr_matrix
#%% Visualização

import matplotlib.pyplot as plt

for i in X_train:
    
    train.columns
    plt.hist(X_train[i])
    plt.title(i)
    plt.show()
  


#%% pivot_table

table = pd.pivot_table(train, index = ['Survived'], columns = ['Pclass'], values = 'PassengerId', aggfunc = 'count')



#%% Função para treinar modelo usando gp_optimize
def treinar_modelo(algoritmo, parametros):
    """
    Função genérica para treinar modelos com cross-validation.
    
    Parâmetros:
    algoritmo: Classe do modelo (ex: RandomForestClassifier, SVC)
    parametros: Dicionário contendo os parâmetros do modelo a serem otimizados
    
    Retorna:
    mean_score: Média dos scores de validação cruzada (inverso para minimização)
    """
    
    # Instancia o modelo com os parâmetros fornecidos
    model = algoritmo(**parametros)
    
    # Realiza a validação cruzada
    score = cross_val_score(model, X_train_sc, y_train, cv=10)
    
    # Calcula a média dos scores
    mean_score = np.mean(score)
    
    # Exibe a média do score
    print(f'Média do score: {mean_score}')
    
    # Retorna o valor negativo da média para fins de otimização (minimização)
    return -mean_score


        
    


#%% Padronização das variáveis

scaler = StandardScaler() #media 0 e desvio padrão 1

X_train_sc = scaler.fit_transform(X_train)

X_test_sc = scaler.transform(X_test)

X_train_sc = pd.DataFrame(X_train_sc)
X_train_sc.columns = X_train.columns

X_test_sc = pd.DataFrame(X_test_sc)
X_test_sc.columns = X_train.columns

y_train = train['Survived']

#%% modelo e validação cruzada

#Logistic Regression
model_lr = LogisticRegression(random_state= 0, max_iter=10000)

lista, sel_features = select_features(model_lr, X_train_sc, y_train, max_f=30, importance_getter='auto')

X_train_sel = X_train_sc[sel_features]

score = cross_val_score(model_lr, X_train_sel, y_train, cv = 10)

print(np.mean(score))




#%% Naive Bayes para Classificação

otimos_gnb = gp_minimize(
    lambda x: treinar_modelo(GaussianNB, {
        'var_smoothing': x[0]    # Ajuste do parâmetro de estabilidade var_smoothing
    }),
    dimensions=[
        (1e-11, 1e-7)   # var_smoothing: valor entre 1e-11 e 1e-7
    ],
    random_state=0,
    verbose=1,
    n_calls=30,
    n_random_starts=10
)

# Exibe o melhor valor da função (fun) e os parâmetros otimizados (x)
print(f'Melhor valor da função: {otimos_gnb.fun}, Parâmetros otimizados: {otimos_gnb.x}')

# Treinando o modelo GaussianNB com os parâmetros otimizados
model_gnb_otimizado = GaussianNB(
    var_smoothing=otimos_gnb.x[0]
)

# Ajustar o modelo no conjunto de treino
model_gnb_otimizado.fit(X_train_sc, y_train)

# Previsão no conjunto de treino
y_pred_gnb = model_gnb_otimizado.predict(X_train_sc)

# Matriz de confusão
mc_gnb = confusion_matrix(y_train, y_pred_gnb)
print("Matriz de Confusão:")
print(mc_gnb)

# Acurácia do modelo
score_gnb_otimizado = model_gnb_otimizado.score(X_train_sc, y_train)
print(f'Acurácia do modelo otimizado: {score_gnb_otimizado}')


#%% KNN para classificação
from sklearn.neighbors import KNeighborsClassifier

otimos_knn = gp_minimize(
    lambda x: treinar_modelo(KNeighborsClassifier, {
        'n_neighbors': x[0], 
        'weights': x[1], 
        'p': x[2]
    }),
    dimensions=[
        (3, 20),               # n_neighbors: entre 3 e 20
        ('uniform', 'distance'),  # weights: 'uniform' ou 'distance'
        (1, 5)                 # p: 1 (distância Manhattan) ou 2 (distância Euclidiana)
    ],  
    random_state=0, 
    verbose=1, 
    n_calls=30, 
    n_random_starts=10
)

# Exibe o melhor valor da função (fun) e os parâmetros otimizados (x)
print(f'Melhor valor da função: {otimos_knn.fun}, Parâmetros: {otimos_knn.x}')

# Treinando o modelo KNeighborsClassifier com os parâmetros otimizados
model_knn = KNeighborsClassifier(
    n_neighbors=otimos_knn.x[0], 
    weights=otimos_knn.x[1], 
    p=otimos_knn.x[2]
)


# Ajuste do modelo no conjunto de treino
model_knn.fit(X_train_sel, y_train)

# Previsão no conjunto de treino
y_pred_knn = model_knn.predict(X_train_sc)

# Matriz de confusão
mc_knn = confusion_matrix(y_train, y_pred_knn)
print("Matriz de Confusão:")
print(mc_knn)

# Acurácia do modelo
score_knn = model_knn.score(X_train_sc, y_train)
print(f'Acurácia: {score_knn}')


#%% SVM para classificação
from sklearn.svm import SVC

otimos_svc = gp_minimize(
    lambda x: treinar_modelo(SVC, {
        'C': x[0],            # Parâmetro de regularização
        'kernel': x[1],       # Tipo de kernel
        'gamma': x[2],        # Coeficiente do kernel para 'rbf', 'poly' e 'sigmoid'
        'degree': x[3]        # Grau do polinômio (se kernel='poly')
    }),
    dimensions=[
        (0.1, 100.0),         # C: valor entre 0.1 e 100
        ('linear', 'poly', 'rbf', 'sigmoid'),  # kernel: tipos de kernel a serem testados
        ('scale', 'auto'),    # gamma: 'scale' ou 'auto'
        (2, 5)                # degree: grau do polinômio, aplicável apenas se kernel='poly'
    ],
    random_state=0,
    verbose=1,
    n_calls=30,
    n_random_starts=10
)

# Exibe o melhor valor da função (fun) e os parâmetros otimizados (x)
print(f'Melhor valor da função: {otimos_svc.fun}, Parâmetros otimizados: {otimos_svc.x}')

# Treinando o modelo SVC com os parâmetros otimizados
model_svc_otimizado = SVC(
    C=otimos_svc.x[0],
    kernel=otimos_svc.x[1],
    gamma=otimos_svc.x[2],
    degree=otimos_svc.x[3] if otimos_svc.x[1] == 'poly' else 3  # Apenas aplicável se o kernel for 'poly'
)

# Ajustar o modelo no conjunto de treino
model_svc_otimizado.fit(X_train_sc, y_train)

# Previsão no conjunto de treino
y_pred_svc = model_svc_otimizado.predict(X_train_sc)

# Matriz de confusão
mc_svc = confusion_matrix(y_train, y_pred_svc)
print("Matriz de Confusão:")
print(mc_svc)

# Acurácia do modelo
score_svc_otimizado = model_svc_otimizado.score(X_train_sc, y_train)
print(f'Acurácia do modelo otimizado: {score_svc_otimizado}')


#%% Decision Tree

from sklearn.tree import DecisionTreeClassifier

model_dt = DecisionTreeClassifier(criterion = 'entropy', max_depth = 3, min_samples_split = 2, min_samples_leaf = 1, random_state = 0)

score = cross_val_score(model_dt, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% Random Forest

from sklearn.ensemble import RandomForestClassifier

model_rf = RandomForestClassifier(criterion = 'entropy', n_estimators = 100, max_depth = 5, min_samples_split = 2, min_samples_leaf = 1, random_state = 0)

score = cross_val_score(model_rf, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% Otimização de hiperparametros do modelo RandomForestClassifier

optimos = gp_minimize(
    lambda x: treinar_modelo(RandomForestClassifier, {
        'criterion': x[0], 
        'n_estimators': x[1], 
        'max_depth': x[2], 
        'min_samples_split': x[3], 
        'min_samples_leaf': x[4], 
        'random_state': 0,
        'n_jobs': -1
    }),
    dimensions=[
        ('entropy', 'gini'), 
        (100, 150), 
        (3, 20),
        (2, 10),
        (1, 10)
    ],
    random_state=0, 
    verbose=1, 
    n_calls=30, 
    n_random_starts=10
)


print(optimos.fun, optimos.x)


model_rf = RandomForestClassifier(criterion = optimos.x[0], n_estimators = optimos.x[1], max_depth = optimos.x[2], 
                                    min_samples_split = optimos.x[3], min_samples_leaf = optimos.x[4], random_state = 0, n_jobs = -1 )


lista, sel_features = select_features(model_rf, X_train_sc, y_train, max_f=5, importance_getter='auto')

X_train_sel = X_train_sc[sel_features]
#%%
  
model_rf.fit(X_train_sel, y_train)

y_pred = model_rf.predict(X_train_sel)

mc = confusion_matrix(y_train, y_pred) 
print(mc)

score = model_rf.score(X_train_sel, y_train)
print(score)

#%% Ensanble model (Voting)
from sklearn.ensemble import VotingClassifier

model_voting = VotingClassifier(estimators = [('LR', model_lr), ('KNN', model_knn), ('SVC', model_svc_otimizado), ('RF', model_rf), ('NB', model_gnb_otimizado)], voting = 'hard')

model_voting.fit(X_train_sc, y_train)

score = cross_val_score(model_voting, X_train_sc, y_train, cv = 10)

print(np.mean(score))


#%% predição nos dados de teste

X_test_sel = X_test_sc[sel_features]

y_pred = model_rf.predict(X_test_sel)

submission = pd.DataFrame(test['PassengerId'])

submission['Survived'] = y_pred

submission.to_csv('submission8.csv', index = False)

























































