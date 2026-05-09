# from scipy.io import loadmat

# # carregar arquivo
# arquivo = loadmat(r"C:\Users\SarahSperduti\OneDrive - Instituto Mauá de Tecnologia\ML EnVision\CRISM_labeled_pixels_ratioed.mat")

# # filtrar só dados úteis
# dados = {k: v for k, v in arquivo.items() if not k.startswith("__")}

# print("=== TAMANHO DAS VARIÁVEIS ===\n")

# for nome, valor in dados.items():
#     print(f"Variável: {nome}")
    
#     if hasattr(valor, "shape"):
#         print(f"Shape: {valor.shape}")
        
#         # quantidade total de elementos
#         print(f"Total de elementos: {valor.size}")
        
#         # dimensão detalhada
#         if len(valor.shape) == 1:
#             print(f"→ Vetor com {valor.shape[0]} elementos")
#         elif len(valor.shape) == 2:
#             print(f"→ {valor.shape[0]} linhas x {valor.shape[1]} colunas")
#         else:
#             print(f"→ {len(valor.shape)} dimensões")
#     else:
#         print("Sem shape")
    
#     print("-" * 40)


# from scipy.io import loadmat
# import numpy as np
# from collections import Counter

# arquivo = loadmat(r"C:\Users\SarahSperduti\OneDrive - Instituto Mauá de Tecnologia\ML EnVision\CRISM_labeled_pixels_ratioed.mat")

# labels  = arquivo['pixlabs'].flatten()
# pixims  = arquivo['pixims'].flatten()
# im_names = arquivo['im_names'].flatten()
# pixpats = arquivo['pixpats'].flatten()

# # ── 1. Quais são os nomes das 77 imagens?
# print("=" * 55)
# print("NOMES DAS 77 IMAGENS CRISM:")
# print("=" * 55)
# for i, nome in enumerate(im_names):
#     print(f"  [{i:>2}] {nome}")

# # ── 2. Quantos pixels por imagem?
# print("\n" + "=" * 55)
# print("PIXELS POR IMAGEM:")
# print("=" * 55)
# contagem_imgs = Counter(pixims)
# for img_id in sorted(contagem_imgs.keys()):
#     nome = im_names[img_id - 1] if img_id <= len(im_names) else "?"
#     print(f"  Imagem {img_id:>3} | {contagem_imgs[img_id]:>7,} pixels | {nome}")

# # ── 3. Quais classes (minerais) existem?
# print("\n" + "=" * 55)
# print("CLASSES DE MINERAIS (pixlabs):")
# print("=" * 55)
# contagem_classes = Counter(labels)
# total = len(labels)
# print(f"\n  Total de pixels : {total:,}")
# print(f"  Classes únicas  : {len(contagem_classes)}\n")
# print(f"  {'Classe':<10} {'Pixels':>10} {'%':>8}")
# print(f"  {'-'*30}")
# for classe, qtd in sorted(contagem_classes.items()):
#     print(f"  {int(classe):<10} {qtd:>10,} {(qtd/total*100):>7.2f}%")

# # ── 4. Cruzamento: quais minerais aparecem em quais imagens?
# print("\n" + "=" * 55)
# print("MINERAIS POR IMAGEM (quais classes cada imagem tem):")
# print("=" * 55)
# for img_id in sorted(contagem_imgs.keys()):
#     mask = pixims == img_id
#     classes_na_img = sorted(set(labels[mask]))
#     nome = im_names[img_id - 1] if img_id <= len(im_names) else "?"
#     print(f"  Img {img_id:>3} | classes: {[int(c) for c in classes_na_img]} | {nome}")

# # ── 5. Patches
# print("\n" + "=" * 55)
# print("PATCHES (pixpats):")
# print("=" * 55)
# print(f"  Patches únicos: {len(set(pixpats))}")
# print(f"  ID mínimo: {pixpats.min()} | ID máximo: {pixpats.max()}")


# -*- coding: utf-8 -*-
"""
Pipeline CRISM otimizado — classifica minerais por imagem
"""

import os
import numpy as np
from scipy.io import loadmat
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# ─── CONFIGURAÇÃO ────────────────────────────────────────────────
ARQUIVO   = r"C:\Users\SarahSperduti\OneDrive - Instituto Mauá de Tecnologia\ML EnVision\CRISM_labeled_pixels_ratioed.mat"
IMAGEM_ID = 10       # ← mude aqui para testar imagens diferentes
SEED      = 42
USAR_PCA  = True     # reduz 350 bandas para N componentes (muito mais rápido)
N_PCA     = 50       # componentes PCA (captura ~99% da variância normalmente)
# ─────────────────────────────────────────────────────────────────


# ── 1. Carregar só o necessário
print(f"\n{'='*55}")
print(f"CARREGANDO IMAGEM {IMAGEM_ID}...")
print(f"{'='*55}")

arquivo  = loadmat(ARQUIVO)
pixims   = arquivo['pixims'].flatten()
labels   = arquivo['pixlabs'].flatten()
spectra  = arquivo['pixspec']
im_names = arquivo['im_names'].flatten()

nome_img = im_names[IMAGEM_ID - 1]
print(f"Nome da imagem: {nome_img}")


# ── 2. Filtrar só os pixels da imagem escolhida
mask    = pixims == IMAGEM_ID
X       = spectra[mask]
Y       = labels[mask]

print(f"Pixels carregados : {X.shape[0]:,}")
print(f"Bandas espectrais : {X.shape[1]}")
print(f"\nClasses encontradas:")
contagem = Counter(Y)
for classe, qtd in sorted(contagem.items()):
    print(f"  Classe {int(classe):>3}  →  {qtd:>6,} pixels  ({qtd/len(Y)*100:.1f}%)")


# ── 3. Normalização (importante para espectros CRISM)
print(f"\n[1/5] Normalizando espectros...")
scaler = StandardScaler()
X = scaler.fit_transform(X)


# ── 4. Redução de dimensionalidade com PCA (opcional mas recomendado)
if USAR_PCA:
    print(f"[2/5] Aplicando PCA ({N_PCA} componentes)...")
    pca = PCA(n_components=N_PCA, random_state=SEED)
    X   = pca.fit_transform(X)
    variancia = pca.explained_variance_ratio_.cumsum()[-1]
    print(f"      Variância explicada: {variancia*100:.1f}%")
else:
    print(f"[2/5] PCA desativado — usando todas as {X.shape[1]} bandas")


# ── 5. Balanceamento com SMOTE (só se necessário)
print(f"[3/5] Verificando balanceamento...")
min_classe = min(contagem.values())

if min_classe >= 6 and len(contagem) > 1:
    smote = SMOTE(sampling_strategy='auto', random_state=SEED, k_neighbors=min(5, min_classe-1))
    X, Y  = smote.fit_resample(X, Y)
    print(f"      SMOTE aplicado → {len(Y):,} amostras totais")
    print(f"      Nova distribuição: {dict(Counter(Y))}")
else:
    print(f"      SMOTE pulado (classe com poucos pixels: {min_classe})")


# ── 6. Divisão treino/teste
print(f"[4/5] Dividindo treino/teste (70/30)...")
X_train, X_test, Y_train, Y_test = train_test_split(
    X, Y, test_size=0.3, random_state=SEED, stratify=Y
)
print(f"      Treino: {len(X_train):,} amostras  |  Teste: {len(X_test):,} amostras")


# ── 7. Busca de hiperparâmetros (range maior e mais esperto)
print(f"[5/5] Buscando melhores hiperparâmetros...")

# n_estimators — range mais realista para Random Forest
melhor_acc = 0
n_best = 10
for n in [10, 30, 50, 100]:
    model  = RandomForestClassifier(n_estimators=n, random_state=SEED, n_jobs=-1)
    scores = cross_val_score(model, X_train, Y_train, cv=5)
    if scores.mean() > melhor_acc:
        melhor_acc = scores.mean()
        n_best     = n
    print(f"      n_estimators={n:<4}  →  acc={scores.mean():.4f}")

print(f"\n  ✔ Melhor n_estimators: {n_best}")

# max_depth
melhor_acc = 0
d_best = 10
for d in [5, 10, 20, None]:
    model  = RandomForestClassifier(n_estimators=n_best, max_depth=d, random_state=SEED, n_jobs=-1)
    scores = cross_val_score(model, X_train, Y_train, cv=5)
    if scores.mean() > melhor_acc:
        melhor_acc = scores.mean()
        d_best     = d
    print(f"      max_depth={str(d):<6}  →  acc={scores.mean():.4f}")

print(f"\n  ✔ Melhor max_depth: {d_best}")


# ── 8. Modelo final
print(f"\n{'='*55}")
print("TREINANDO MODELO FINAL...")
print(f"{'='*55}")

modelo_final = RandomForestClassifier(
    n_estimators = n_best,
    max_depth    = d_best,
    random_state = SEED,
    n_jobs       = -1       # usa todos os núcleos do processador
)
modelo_final.fit(X_train, Y_train)
Y_pred = modelo_final.predict(X_test)


# ── 9. Resultados
print(f"\n{'='*55}")
print(f"RESULTADOS — IMAGEM {IMAGEM_ID} ({nome_img})")
print(f"{'='*55}")
print(f"\nAcurácia final: {accuracy_score(Y_test, Y_pred)*100:.2f}%\n")
print(classification_report(Y_test, Y_pred, zero_division=0))


# ── 10. Importância das bandas (ou componentes PCA)
importancias = modelo_final.feature_importances_
top5_idx = np.argsort(importancias)[::-1][:5]
print(f"\nTop 5 {'componentes PCA' if USAR_PCA else 'bandas espectrais'} mais importantes:")
for i, idx in enumerate(top5_idx):
    print(f"  {i+1}. {'Componente' if USAR_PCA else 'Banda'} {idx+1:>3}  →  importância: {importancias[idx]:.4f}")