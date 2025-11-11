# Reprodução — “Identifying Malignant Skin Diseases Through Deep Learning (PAD-UFES-20)”

Reprodução **parcial e documentada** do estudo _Identifying Malignant Skin Diseases Through Deep Learning_ (RITA 2025), usando **Vision Transformer (ViT-B/16)** no dataset **PAD-UFES-20**. O objetivo é verificar a reprodutibilidade dos resultados em diferentes *splits*, comparando **Cross-Entropy (NF)** e **Focal Loss (F)**, com validação estratificada.

---

##  Visão geral

- **Tarefa**: classificação de lesões cutâneas (multiclasse) no **PAD-UFES-20**.  
- **Backbone**: **ViT-B/16** pré-treinado (ImageNet).  
- **Comparação**: NF (Cross-Entropy) × F (Focal Loss).  
- **Splits usados**: `PAD-UFES`, `PAD-UFES-FULL`, `PAD-UFES-IS`.  
  > O *split* `PAD-UFES-AUG` **não** é utilizado nesta reprodução.
- **Protocolo**: imagens **224×224**, validação **k=3** (estratificada), seleção por **F1 macro** em validação, *early stopping*.  
- **Saídas**: checkpoints `.pt`, CSVs por *fold/split*, tabelas agregadas e gráficos.

---

##  Ambiente e requisitos

**Sistema testado**: Windows + **NVIDIA RTX 2050 (CUDA)**, **VS Code**, **Python 3.12**  
*(Funciona também em 3.10–3.12 com as versões abaixo.)*

##  Estrutura do repositorio

pad-ufes20-vit-repro/
├─ README.md
├─ requirements.txt
├─ .gitignore
├─ config.yaml
├─ src/
│  ├─ datasets.py
│  ├─ evaluate.py
│  ├─ losses.py
│  ├─ make_figs.py
│  ├─ models.py
│  ├─ predict.py
│  ├─ rebuild_results_from_ckpts.py
│  ├─ split_maker.py
│  └─ train.py
├─ data/
│  └─ PAD-UFES-20/
│     ├─ metadata.csv
│     ├─ imgs_part_1/ …
│     ├─ imgs_part_2/ …
│     ├─ imgs_part_3/ …
│     └─ splits/            # gerado pelo script
└─ reports/
   ├─ checkpoints/          # .pt (considerar Git LFS)
   ├─ tables/               # CSVs de resultados
   └─ figs/                 # Gráficos .png
