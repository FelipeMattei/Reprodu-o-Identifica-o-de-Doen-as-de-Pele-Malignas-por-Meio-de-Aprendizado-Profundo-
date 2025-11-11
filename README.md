# Reprodução do estudo “Identifying Malignant Skin Diseases Through Deep Learning (PAD-UFES-20)”

Este repositório traz uma **reprodução parcial e documentada** do estudo (RITA 2025) sobre classificação de lesões cutâneas no **PAD-UFES-20** usando **Vision Transformer (ViT-B/16)**.  
O objetivo é permitir que qualquer pessoa **organize os dados, gere splits, treine, avalie, crie tabelas e figuras, faça predições avulsas e reconstrua resultados a partir de checkpoints** — de forma simples e rastreável.

---

## 1) O que este código faz (visão geral)

- **Tarefa**: classificação multiclasse no PAD-UFES-20.  
- **Backbone**: ViT-B/16 pré-treinado (ImageNet).  
- **Perdas**: Cross-Entropy (NF) e Focal Loss (F).  
- **Splits utilizados**: `PAD-UFES`, `PAD-UFES-FULL`, `PAD-UFES-IS`.  
  > Observação: o script também gera `PAD-UFES-AUG`, mas **não** é usado nos resultados principais desta reprodução.  
- **Validação**: k-fold estratificada (`k=3`), **seleção por F1 macro** e **early stopping**.  
- **Saídas**: checkpoints (`.pt`), CSVs por fold/split, tabelas agregadas e gráficos prontos para o relatório.

---

## 2) Estrutura esperada do repositório

```
.
├─ README.md
├─ requirements.txt
├─ config.yaml
├─ src/
│  ├─ datasets.py
│  ├─ evaluate.py
│  ├─ losses.py
│  ├─ make_figs.py
│  ├─ models.py
│  ├─ predict.py
│  ├─ rebuild_results_from_ckpts.py
│  └─ split_maker.py
├─ data/
│  └─ PAD-UFES-20/
│     ├─ metadata.csv           # colunas: image_path,label
│     ├─ imgs_part_1/...
│     ├─ imgs_part_2/...
│     └─ imgs_part_3/...
└─ reports/
   ├─ checkpoints/
   ├─ tables/
   └─ figs/
```

As pastas em `reports/` são criadas automaticamente pelos scripts.

---

## 3) Links úteis (dados e checkpoints)

- **Dataset (PAD-UFES-20) para baixar**: (https://www.kaggle.com/datasets/mahdavi1202/skin-cancer?resource=download)
  Descompacte em: `data/PAD-UFES-20/` mantendo `metadata.csv` e as pastas `imgs_part_*`.

- **Checkpoints prontos (.pt)**: <COLOQUE_AQUI_O_LINK_DOS_CHECKPOINTS_NO_DRIVE>  
  Coloque os arquivos em: `reports/checkpoints/`  
  Padrão de nome esperado: `vit_base_<SPLIT>_fold<F>_{focal|nofocal}.pt`

Se usar os checkpoints fornecidos, é possível **pular o treino** e apenas reconstruir os CSVs (Seção 7.2).

---

## 4) Configuração mínima

1. Instale as dependências:
   ```
   pip install -r requirements.txt
   ```
2. Verifique `config.yaml` (valores padrão já atendem):
   - `device: cuda` (se houver GPU; caso contrário o código usa CPU)
   - `data.root: data/PAD-UFES-20`
   - `data.split_dir: data/PAD-UFES-20/splits`
   - `data.img_size: 224`
   - `train.k_folds: 3`, `train.batch_size: 32`, `train.max_epochs: 100`, `train.early_stopping_patience: 20`
   - `log.save_dir: reports`

---

## 5) Passo a passo essencial (reprodução do zero)

### 5.1) Gerar splits
Gera `PAD-UFES`, `PAD-UFES-FULL`, `PAD-UFES-IS` (e `PAD-UFES-AUG`, não usado no principal):
```
python -m src.split_maker   --meta data/PAD-UFES-20/metadata.csv   --root data/PAD-UFES-20   --out  data/PAD-UFES-20/splits   --seed 123
```

### 5.2) Treinar e testar (k=3)
Executa k-fold por split e por perda (NF/F), salva logs, checkpoints e resultados:
```
python -m src.train --config config.yaml
```
Saída consolidada: `reports/tables/all_results_test_by_fold.csv`

### 5.3) Tabelas agregadas (para o artigo)
Gera tabelas organizadas e resumo (médias e desvios):
```
python -m src.evaluate
```
Saídas:
- `reports/tables/table2_vit.csv`
- `reports/tables/summary_means_stds.csv`

### 5.4) Figuras (para o artigo)
Cria gráficos de barras, boxplots e linhas por fold:
```
python -m src.make_figs
```
Arquivos em `reports/figs/`:
- `bar_Acc_mean_vit_base.png`, `bar_F1_mean_vit_base.png`
- `box_F1_by_split_vit_base.png`
- `line_F1_by_split_folds_vit_base_NF.png`, `line_F1_by_split_folds_vit_base_F.png`

---

## 6) Predição em imagens avulsas (top-3 classes)

Use um checkpoint salvo e o `metadata.csv` para obter as top-3 classes por imagem:
```
python -m src.predict   --images caminho/img1.jpg caminho/img2.jpg   --ckpt reports/checkpoints/vit_base_PAD-UFES-FULL_fold0_nofocal.pt   --model vit_base   --metadata data/PAD-UFES-20/metadata.csv   --device cuda   --img-size 224
```

Exemplo de saída:
```
caminho/img1.jpg:
  1. classe_X  0.6123
  2. classe_Y  0.2311
  3. classe_Z  0.1022
```

---

## 7) Usando checkpoints prontos

### 7.1) Apenas gerar tabelas e figuras
Se você já treinou (ou baixou) os checkpoints:
```
python -m src.evaluate
python -m src.make_figs
```

### 7.2) Reconstruir `all_results_test_by_fold.csv` a partir dos `.pt`
Sem treinar de novo:
```
python -m src.rebuild_results_from_ckpts --config config.yaml
```
O script lê cada `.pt` em `reports/checkpoints/`, infere `split/fold/focal` pelo nome do arquivo, avalia no teste correspondente e recria `reports/tables/all_results_test_by_fold.csv`.

---

## 8) Onde olhar os resultados

- `reports/tables/all_results_test_by_fold.csv`: resultados brutos por split/fold/perda.  
- `reports/tables/summary_means_stds.csv`: médias e desvios por split/perda.  
- `reports/tables/table2_vit.csv`: tabela final em formato de artigo.  
- `reports/figs/*.png`: figuras de barras, boxplots e linhas por fold.

Leitura recomendada:
- **PAD-UFES-IS** é o recorte mais exigente; tende a ter médias menores e maior dispersão.  
- **NF × F**: o efeito da Focal é dependente do contexto; em alguns splits melhora pouco, em outros pode piorar.  
- **ViT-B/16** oferece linha de base estável em `PAD-UFES` e `PAD-UFES-FULL`.

---

## 9) Dúvidas comuns

- FileNotFoundError em `all_results_test_by_fold.csv`:  
  Rode o treino (`src.train`) ou reconstrua com `src.rebuild_results_from_ckpts`.

- GPU com pouca memória:  
  Reduza `train.batch_size` no `config.yaml` e mantenha `train.amp: true`. Em CPU funciona, mas é mais lento.

- Resultados diferentes do artigo:  
  O artigo usa `k=5`, escopo mais amplo e possivelmente outros hiperparâmetros. Aqui: `k=3`, `224×224`, um único backbone e sem o split com augmentation global. Foque nas **tendências relativas** entre perdas e entre splits.

---

## 10) Referência

- Estudo base: *Identifying Malignant Skin Diseases Through Deep Learning* (RITA, 2025)  
- Dataset: **PAD-UFES-20**

Para sugestões ou problemas, abra uma issue neste repositório.
