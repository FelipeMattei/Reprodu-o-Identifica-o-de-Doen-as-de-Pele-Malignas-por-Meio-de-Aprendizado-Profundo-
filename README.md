# Reprodução — “Identifying Malignant Skin Diseases Through Deep Learning (PAD-UFES-20)”

Reprodução **parcial e documentada** do estudo _Identifying Malignant Skin Diseases Through Deep Learning_ (RITA 2025), usando **Vision Transformer (ViT-B/16)** no dataset **PAD-UFES-20**. O objetivo é verificar a reprodutibilidade dos resultados em diferentes *splits*, comparando **Cross-Entropy (NF)** e **Focal Loss (F)**, com validação estratificada.

---

## ✨ Visão geral

- **Tarefa**: classificação de lesões cutâneas (multiclasse) no **PAD-UFES-20**.  
- **Backbone**: **ViT-B/16** pré-treinado (ImageNet).  
- **Comparação**: NF (Cross-Entropy) × F (Focal Loss).  
- **Splits usados**: `PAD-UFES`, `PAD-UFES-FULL`, `PAD-UFES-IS`.  
  > O *split* `PAD-UFES-AUG` **não** é utilizado nesta reprodução.
- **Protocolo**: imagens **224×224**, validação **k=3** (estratificada), seleção por **F1 macro** em validação, *early stopping*.  
- **Saídas**: checkpoints `.pt`, CSVs por *fold/split*, tabelas agregadas e gráficos.

---

## 🧰 Ambiente e requisitos

**Sistema testado**: Windows + **NVIDIA RTX 2050 (CUDA)**, **VS Code**, **Python 3.12**  
*(Funciona também em 3.10–3.12 com as versões abaixo.)*

### `requirements.txt`
```txt
torch==2.4.0
torchvision==0.19.0
torchaudio==2.4.0
scikit-learn==1.5.1
pandas==2.2.2
matplotlib==3.9.1
numpy==1.26.4
tqdm==4.66.4
opencv-python==4.10.0.84
