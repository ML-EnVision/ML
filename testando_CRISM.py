# -*- coding: utf-8 -*-
"""
Created on Sat Dec 20 03:41:59 2025

@author: Roberto
"""

import scipy.io
import matplotlib.pyplot as plt
import numpy as np
from imblearn.over_sampling import SMOTE
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score

#Nome (caminho completo) do arquivo contendo os espectros das regiões observadas
# tabela=r"D:/RSI/IDL61/teste_CRISM/CRISM_labeled_pixels_ratioed.mat"
import os
tabela = os.path.join(os.path.dirname(__file__), "CRISM_labeled_pixels_ratioed.mat")

#Número do elemento a ser identificado
elemento=1


#Início do programa

#Lendo a tabela
data = scipy.io.loadmat(tabela)

#Acessando os espectros e os rótulos
spectra = data['pixspec']
labels = data['pixlabs']

print(f"Shape dos espectros: {spectra.shape}")  # Deveria ser (592413, 350)
print(f"Shape dos rótulos: {labels.shape}")    # Deveria ser (592413,)

labels = labels.flatten()

print(f"Novo shape dos rótulos: {labels.shape}")


spectra_elemento = spectra[labels == elemento]

#print(f"Shape dos espectros de CO2 ice: {spectra_co2ice.shape}")

spectra_non_elemento = spectra[labels != elemento]

#print(f"Shape dos espectros sem CO2 ice: {spectra_non_co2ice.shape}")

X = np.concatenate([spectra_elemento, spectra_non_elemento])

Y = np.concatenate([np.ones(spectra_elemento.shape[0]), np.zeros(spectra_non_elemento.shape[0])])

print(f"Tamanho de X: {X.shape}")
print(f"Tamanho de Y: {Y.shape}")

smote = SMOTE(sampling_strategy='auto', random_state=42)

X_resampled, Y_resampled = smote.fit_resample(X, Y)

X_resampled_final=np.zeros((50000,350))
Y_resampled_final=np.zeros(50000)

X_resampled_final[:]=X_resampled[0:50000,:]
Y_resampled_final[:]=Y_resampled[0:50000]


print(f"Distribuição das classes após o over-sampling: {Counter(Y_resampled)}")

seed = 7

X_train, X_test, Y_train, Y_test = train_test_split(X_resampled_final, Y_resampled_final, test_size=0.3, random_state=seed)

print(f"Tamanho do conjunto de treinamento: {X_train.shape[0]}")
print(f"Tamanho do conjunto de teste: {X_test.shape[0]}")

#Determinando o melhor valor para "n_estimators"
accuracies = []
for n in range(1, 11):
    model = RandomForestClassifier(n_estimators=n, random_state=seed)
    scores = cross_val_score(model, X_train, Y_train, cv=5)
    accuracies.append(scores.mean())
    print(n)
n_best = np.argmax(accuracies) + 1
print("Best n_estimators:", n_best)

#Determinando o melhor valor para "max_depth"
accuracies = []
for d in range(1, 11):
    model = RandomForestClassifier(n_estimators=n_best, max_depth=d, random_state=seed)
    scores = cross_val_score(model, X_train, Y_train, cv=5)
    accuracies.append(scores.mean())
    print(d)
d_best = np.argmax(accuracies) + 1
print("Best max_depth:", d_best)

#Determinando o melhor valor para "min_samples_leaf"
accuracies = []
for m in range(1, 11):
    model = RandomForestClassifier(
        n_estimators=n_best, max_depth=d_best, min_samples_leaf=m, random_state=seed)
    scores = cross_val_score(model, X_train, Y_train, cv=5)
    accuracies.append(scores.mean())
    print(m)
m_best = np.argmax(accuracies) + 1
print("Best min_samples_leaf:", m_best)

#Modelo final
final_model = RandomForestClassifier(
    n_estimators=n_best,
    max_depth=d_best,
    min_samples_leaf=m_best,
    random_state=seed
)
final_model.fit(X_train, Y_train)
Y_pred = final_model.predict(X_test)

print("Final Accuracy: {:.2f}".format(accuracy_score(Y_test, Y_pred)*100))



