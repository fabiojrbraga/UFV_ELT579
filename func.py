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

def select_features(model, X_train, y_train, importance_getter='auto'
                    , start=1, max_f = 20, step = 1):
    from sklearn.feature_selection import RFE
    
    lista_score = list()
    for i in range(start, max_f +1):
    
        selector = RFE(model, n_features_to_select = i, step = step)
        selector = selector.fit(X_train, y_train)
        mask = selector.support_
    
        features = X_train.columns
    
        sel_features = features[mask]
    
        X_sel = X_train[sel_features]
        score = cross_val_score(model, X_sel, y_train, cv = 10)
        
        print("Iteração ",i, ". Score: ", np.mean(score))
        
        lista_score.append(np.mean(score))
    return lista_score, sel_features


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


def treinar_modelo(algoritmo, X_train_sc, y_train, parametros):
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


        
    



















































