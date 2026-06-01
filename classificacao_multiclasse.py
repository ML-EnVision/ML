# -*- coding: utf-8 -*-
# PASSO 6 — Classificação Multiclasse

import os
import numpy as np
import scipy.io
import matplotlib.pyplot as plt
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

ARQUIVO   = os.path.join(os.path.dirname(__file__), 'CRISM_labeled_pixels_ratioed.mat')
IMAGEM_ID = 10
SEED      = 7

# ── 1. Carregar dados da imagem
print(f'[1/5] Carregando imagem {IMAGEM_ID}...')
data     = scipy.io.loadmat(ARQUIVO)
mask     = data['pixims'].flatten() == IMAGEM_ID
X        = data['pixspec'][mask]
Y        = data['pixlabs'].flatten()[mask].astype(int)   # rótulos REAIS — não 0/1
nome_img = str(data['im_names'].flatten()[IMAGEM_ID - 1])

classes_presentes = sorted(set(Y))
print(f'    {len(X):,} pixels | {X.shape[1]} bandas')
print(f'    Minerais presentes: {classes_presentes}')
print(f'    Distribuição: {dict(Counter(Y))}')

# ── 2. Pré-processamento (idêntico aos passos anteriores)
print('[2/5] Normalizando + PCA...')
scaler = StandardScaler()
X_norm = scaler.fit_transform(X)

pca   = PCA(n_components=0.99, random_state=SEED)
X_pca = pca.fit_transform(X_norm)
print(f'    {X.shape[1]} bandas -> {X_pca.shape[1]} componentes PCA')
print(f'    Variância mantida: {pca.explained_variance_ratio_.cumsum()[-1]*100:.2f}%')

# ── 3. SMOTE multiclasse — k_neighbors adaptativo
# O SMOTE funciona igual ao binário, só que para N classes ao mesmo tempo
print('[3/5] SMOTE multiclasse adaptativo...')
min_classe = int(min(Counter(Y).values()))
k_viz      = min(5, min_classe - 1)

if k_viz >= 1 and len(classes_presentes) > 1:
    smote        = SMOTE(sampling_strategy='auto', random_state=SEED, k_neighbors=k_viz)
    X_res, Y_res = smote.fit_resample(X_pca, Y)
    print(f'    k_neighbors={k_viz}  |  {len(Y_res):,} amostras após SMOTE')
    print(f'    {dict(Counter(Y_res))}')
else:
    X_res, Y_res = X_pca, Y
    print('    SMOTE pulado — amostras insuficientes na menor classe.')

# ── 4. Divisão estratificada treino/teste
X_train, X_test, Y_train, Y_test = train_test_split(
    X_res, Y_res,
    test_size=0.3,
    random_state=SEED,
    stratify=Y_res    # garante proporção igual em treino e teste
)
print(f'\n[4/5] Split: {len(X_train):,} treino | {len(X_test):,} teste')

# ── 5. Random Forest multiclasse
# Nenhuma mudança na chamada — sklearn já suporta múltiplas classes nativamente
print('[5/5] Treinando Random Forest multiclasse...')
modelo = RandomForestClassifier(
    n_estimators=100, max_depth=10, random_state=SEED, n_jobs=-1)
modelo.fit(X_train, Y_train)
Y_pred = modelo.predict(X_test)

# ── Métricas
acc = accuracy_score(Y_test, Y_pred) * 100
print(f'\nAcurácia multiclasse: {acc:.2f}%')
print('\n' + classification_report(
    Y_test, Y_pred,
    target_names=[f'Mineral {c}' for c in sorted(set(Y_test))],
    zero_division=0
))

# ── Matriz de confusão
fig, ax = plt.subplots(figsize=(8, 6))
ConfusionMatrixDisplay.from_predictions(
    Y_test, Y_pred,
    ax=ax,
    cmap='Blues',
    display_labels=[f'M{c}' for c in sorted(set(Y_test))]
)
plt.title(f'Matriz de Confusão Multiclasse — Imagem {IMAGEM_ID} ({nome_img})')
plt.tight_layout()
plt.savefig(f'matriz_img{IMAGEM_ID}.png', dpi=150)
plt.show()
print(f'Gráfico salvo: matriz_img{IMAGEM_ID}.png')

# ── Importância de features por classe
# Mapeia importância dos componentes PCA de volta para as 350 bandas originais
importancias_pca = modelo.feature_importances_
contrib_bandas   = np.abs(pca.components_).T @ importancias_pca
top15_idx        = np.argsort(contrib_bandas)[::-1][:15]

print('\nTop 15 bandas espectrais mais discriminativas (todas as classes):')
print(f'{"Rank":>5}  {"Banda":>6}  {"Contribuição":>14}')
print('-' * 30)
for rank, b in enumerate(top15_idx, 1):
    print(f'{rank:>5}  {b+1:>6}  {contrib_bandas[b]:>14.4f}')

# Salvar contribuição de todas as bandas para uso posterior
np.save('contrib_bandas_multiclasse.npy', contrib_bandas)
print('\nArquivo salvo: contrib_bandas_multiclasse.npy')