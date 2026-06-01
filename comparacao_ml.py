# -*- coding: utf-8 -*-
# PASSO 5 — Comparação de Modelos de Machine Learning

import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, f1_score, classification_report
from xgboost import XGBClassifier   # pip install xgboost

SEED     = 7
ELEMENTO = 7   # mesmo mineral usado nos passos anteriores

# Carregar dados preparados pelo Passo 4
X_train = np.load('X_train.npy')
X_test  = np.load('X_test.npy')
Y_train = np.load('Y_train.npy')
Y_test  = np.load('Y_test.npy')

print(f'Treino : {X_train.shape}  |  classes: {dict(zip(*np.unique(Y_train, return_counts=True)))}')
print(f'Teste  : {X_test.shape}   |  classes: {dict(zip(*np.unique(Y_test,  return_counts=True)))}')


modelos = {
    'Random Forest': RandomForestClassifier(
        n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1),

    'SVM (RBF)': SVC(
        kernel='rbf', C=10, gamma='scale', random_state=SEED),

    'XGBoost': XGBClassifier(
        n_estimators=100, max_depth=6, learning_rate=0.1,
        random_state=SEED, n_jobs=-1,
        eval_metric='logloss', verbosity=0),

    'KNN': KNeighborsClassifier(
        n_neighbors=5, n_jobs=-1),

    'Rede Neural (MLP)': MLPClassifier(
        hidden_layer_sizes=(128, 64), max_iter=300,
        random_state=SEED, early_stopping=True),
}

print(f'\n{"Modelo":<22} {"CV Acc":>8} {"Teste Acc":>10} {"F1":>8} {"Tempo":>8}')
print('-' * 62)

resultados = {}
for nome, modelo in modelos.items():
    t0 = time.time()

    # Cross-validation no treino — mede generalização sem ver o teste
    cv_scores = cross_val_score(modelo, X_train, Y_train, cv=5, n_jobs=-1)

    # Treino final e avaliação no conjunto de teste
    modelo.fit(X_train, Y_train)
    Y_pred = modelo.predict(X_test)

    tempo = time.time() - t0
    acc   = accuracy_score(Y_test, Y_pred) * 100
    f1    = f1_score(Y_test, Y_pred, average='weighted', zero_division=0)

    resultados[nome] = {
        'cv': cv_scores.mean() * 100,
        'acc': acc,
        'f1': f1,
        'tempo': tempo,
        'modelo': modelo,
        'pred': Y_pred,
    }
    print(f'{nome:<22} {cv_scores.mean()*100:>7.2f}%  {acc:>9.2f}%  {f1:>7.4f}  {tempo:>6.1f}s')

melhor_nome = max(resultados, key=lambda k: resultados[k]['f1'])
melhor      = resultados[melhor_nome]
print(f'\nMelhor modelo: {melhor_nome}  (F1={melhor["f1"]:.4f})')

print('\n' + classification_report(
    Y_test, melhor['pred'],
    target_names=[f'Não-{ELEMENTO}', f'Mineral {ELEMENTO}'],
    zero_division=0
))

nomes  = list(resultados.keys())
accs   = [resultados[n]['acc'] for n in nomes]
f1s    = [resultados[n]['f1'] * 100 for n in nomes]
x      = np.arange(len(nomes))

fig, ax = plt.subplots(figsize=(10, 5))
bars1 = ax.bar(x - 0.2, accs, 0.4, label='Acurácia (%)',  color='steelblue')
bars2 = ax.bar(x + 0.2, f1s,  0.4, label='F1-score (%)', color='coral')

for bar in bars1 + bars2:
    ax.text(bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.3,
            f'{bar.get_height():.1f}',
            ha='center', va='bottom', fontsize=8)

ax.set_xticks(x)
ax.set_xticklabels(nomes, rotation=15, ha='right')
ax.set_ylabel('Score (%)')
ax.set_ylim(0, 110)
ax.set_title(f'Comparação de Modelos — Mineral {ELEMENTO} (binário)')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('comparacao_modelos.png', dpi=150)
plt.show()
print('Gráfico salvo: comparacao_modelos.png')