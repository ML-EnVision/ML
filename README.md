# Inicialização do Sistema

## Instrução Inicial
Antes de executar qualquer etapa, ajuste os caminhos locais no arquivo:

- `testando_crism.py`

Garanta que os dados a serem analisados estejam corretamente configurados.

---

## Ordem Recomendada de Execução

### Etapa 1 — Preparação e Exploração
1. `testando_crism.py`  
   **Passo 0**

2. `exploracao_banco_de_dados.py`  
   **Passo 1**

---

### Etapa 2 — Pré-processamento
3. `normalizacao.py`  
4. `pca.py`  
   **Passo 2**

---

### Etapa 3 — Tratamento de Dados
5. `filtragem_por_imagem.py`  
6. `separacao_binaria.py`  
   **Passo 3**

---

### Etapa 4 — Balanceamento e Divisão
7. `balanceamento_smote.py`  
8. `divisao_teste.py`  
   **Passo 4**

---

### Etapa 5 — Modelagem
9. `comparacao_ml.py`  
   **Passo 5**

10. `classificacao_multiclasse.py`  
    **Passo 6**

---

### Etapa 6 — Análise de Assinaturas
11. `agrupamento_assinatura.py`  
    **Passo 7**

12. `assinatura_espectral.py`  

---

### Etapa 7 — Geração de Resultados
13. `mapa_mineral.py`  
    **Passo 8**

14. `pipeline_final.py`  

---

## Observações
- Execute os scripts na ordem indicada para evitar inconsistências.
- Algumas etapas dependem diretamente da saída das anteriores.
