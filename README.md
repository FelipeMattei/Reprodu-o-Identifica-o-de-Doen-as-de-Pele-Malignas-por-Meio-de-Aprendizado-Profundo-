# Reprodução do estudo “Identifying Malignant Skin Diseases Through Deep Learning (PAD-UFES-20)”

Este repositório traz uma **reprodução parcial e documentada** do estudo de 2025 sobre classificação de lesões cutâneas usando **Vision Transformer (ViT-B/16)** no **PAD-UFES-20**. O objetivo é permitir que qualquer pessoa consiga **baixar os dados, organizar os splits, treinar, avaliar, gerar tabelas/figuras, prever em imagens avulsas e reconstruir resultados** a partir de checkpoints — tudo de forma simples e rastreável.

---

## 1) O que este código faz (visão geral)

- **Tarefa**: classificação multiclasse de lesões do PAD-UFES-20.  
- **Backbone**: ViT-B/16 pré-treinado (ImageNet).  
- **Perdas comparadas**: Cross-Entropy (NF) e Focal Loss (F).  
- **Splits utilizados**: `PAD-UFES`, `PAD-UFES-FULL`, `PAD-UFES-IS`  
  > Observação: o script gera também `PAD-UFES-AUG`, mas **não** é utilizado na avaliação principal desta reprodução.  
- **Validação**: k-fold estratificada (`k=3`), **seleção por F1 macro** na validação e **early stopping**.  
- **Saídas**: checkpoints (`.pt`), resultados por fold/split em CSV, **tabelas agregadas** e **gráficos** prontos para usar no relatório (artigo).

---

## 2) Estrutura esperada do repositório

.
├─ README.md
├─ requirements.txt # já fornecido
├─ config.yaml # exemplo de configuração
├─ src/
│ ├─ datasets.py # pipeline de dados + Dataset baseado em CSV
│ ├─ evaluate.py # gera tabelas agregadas e resumo (means/stds)
│ ├─ losses.py # FocalLoss
│ ├─ make_figs.py # gera gráficos a partir dos CSVs
│ ├─ models.py # ViT-B/16 e ResNet-101 (para extensão)
│ ├─ predict.py # predição top-3 em imagens avulsas
│ ├─ rebuild_results_from_ckpts.py# reavalia testes a partir dos .pt
│ ├─ split_maker.py # gera os splits a partir do metadata.csv
│ └─ train.py # loop de treino k-fold + teste final
├─ data/
│ └─ PAD-UFES-20/
│ ├─ metadata.csv # colunas: image_path,label
│ ├─ imgs_part_1/…
│ ├─ imgs_part_2/…
│ └─ imgs_part_3/…
└─ reports/
├─ checkpoints/ # .pt gerados
├─ tables/ # CSVs gerados
└─ figs/ # gráficos gerados

yaml
Copiar código

Se `reports/` e `data/` ainda não existirem, os próprios scripts criam as pastas necessárias.

---

## 3) Preparando o ambiente rapidamente

1. **Clonar e entrar na pasta**
   ```bash
   git clone https://github.com/<seu-usuario>/<seu-repo>.git
   cd <seu-repo>
Criar ambiente e instalar dependências

bash
Copiar código
# Ambiente virtual (Windows)
python -m venv .venv
.venv\Scripts\activate

# Alternativa Linux/macOS:
# python -m venv .venv
# source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
Configurar o config.yaml (já existe um exemplo no repositório)

device: cuda (usa GPU se disponível; caso contrário cai para CPU automaticamente)

data.root: data/PAD-UFES-20

data.split_dir: data/PAD-UFES-20/splits

data.img_size: 224

train.k_folds: 3, train.batch_size: 32, train.max_epochs: 100, train.early_stopping_patience: 20

log.save_dir: reports

4) Preparando os dados (PAD-UFES-20)
Organize os arquivos conforme abaixo e garanta que o metadata.csv contenha exatamente as colunas: image_path,label.

kotlin
Copiar código
data/PAD-UFES-20/
├─ metadata.csv
├─ imgs_part_1/
├─ imgs_part_2/
└─ imgs_part_3/
Dica: se o metadata.csv tiver caminhos duplicados (por exemplo imgs_part_1/imgs_part_1/...), o script de split corrige isso automaticamente.

5) Gerando os splits (train/test)
Crie os 4 splits a partir do metadata.csv:

bash
Copiar código
python -m src.split_maker \
  --meta data/PAD-UFES-20/metadata.csv \
  --root data/PAD-UFES-20 \
  --out  data/PAD-UFES-20/splits \
  --seed 123
Serão gerados:

PAD-UFES_train.csv, PAD-UFES_test.csv

PAD-UFES-FULL_train.csv, PAD-UFES-FULL_test.csv

PAD-UFES-IS_train.csv, PAD-UFES-IS_test.csv

PAD-UFES-AUG_train.csv, PAD-UFES-AUG_test.csv (não usado nos resultados principais)

6) Treinando e testando (k-fold + teste final)
Execute o treino k-fold (segundo o config.yaml):

bash
Copiar código
python -m src.train --config config.yaml
Isso irá:

Executar StratifiedKFold k=3 para cada split e para cada perda (NF/F).

Salvar logs de treino em reports/tables/trainlog_<run>.csv.

Salvar checkpoints em reports/checkpoints/<model>_<split>_fold<k>_{focal|nofocal}.pt.

Avaliar no teste ao final, gerando o arquivo consolidado:

bash
Copiar código
reports/tables/all_results_test_by_fold.csv
Overrides úteis (opcional)
Mudar otimizador/épocas:

bash
Copiar código
python -m src.train --config config.yaml train.optimizer=sgd train.max_epochs=120
Ajustar Focal Loss:

bash
Copiar código
python -m src.train --config config.yaml train.focal_loss.alpha=0.5 train.focal_loss.gamma=1.5
7) Gerando as tabelas agregadas
Com base no all_results_test_by_fold.csv:

bash
Copiar código
python -m src.evaluate
Saídas principais:

reports/tables/table2_vit.csv (resultados por split/fold organizados)

reports/tables/summary_means_stds.csv (médias e desvios por split/focal/model)

8) Gerando os gráficos
A partir dos CSVs, gere as figuras:

bash
Copiar código
python -m src.make_figs
Arquivos produzidos em reports/figs/:

bar_Acc_mean_<model>.png e bar_F1_mean_<model>.png

box_F1_by_split_<model>.png

line_F1_by_split_folds_<model>_NF.png e line_F1_by_split_folds_<model>_F.png

Essas figuras já foram pensadas para ilustrar bem as diferenças entre NF e F, a estabilidade entre folds e o efeito de cada split no desempenho.

9) Predizendo em imagens avulsas
Use um checkpoint salvo para obter as top-3 classes por imagem:

bash
Copiar código
python -m src.predict \
  --images path/da_imagem1.jpg path/da_imagem2.jpg \
  --ckpt reports/checkpoints/vit_base_PAD-UFES-FULL_fold0_nofocal.pt \
  --model vit_base \
  --metadata data/PAD-UFES-20/metadata.csv \
  --device cuda \
  --img-size 224
Saída ilustrativa:

bash
Copiar código
path/da_imagem1.jpg:
  1. classe_X  0.6123
  2. classe_Y  0.2311
  3. classe_Z  0.1022
10) Reconstruindo o CSV de resultados a partir dos checkpoints
Se você já tem .pt em reports/checkpoints/ e quer recriar all_results_test_by_fold.csv sem rodar todo o treino:

bash
Copiar código
python -m src.rebuild_results_from_ckpts --config config.yaml
O script:

Lê o nome de cada .pt para inferir model, split, fold e focal.

Carrega o conjunto de teste do respectivo split e computa Acc, Prec, Rec e F1.

Gera novamente reports/tables/all_results_test_by_fold.csv.

11) Fluxo mínimo para reproduzir os resultados
Preparar o ambiente (venv + pip install -r requirements.txt).

Colocar o PAD-UFES-20 em data/PAD-UFES-20/ e conferir metadata.csv.

Gerar splits:

bash
Copiar código
python -m src.split_maker --meta data/PAD-UFES-20/metadata.csv --root data/PAD-UFES-20 --out data/PAD-UFES-20/splits --seed 123
Treinar e testar:

bash
Copiar código
python -m src.train --config config.yaml
Agregar resultados e plotar:

bash
Copiar código
python -m src.evaluate
python -m src.make_figs
12) Dúvidas e problemas comuns (Troubleshooting)
Erro: FileNotFoundError: reports/tables/all_results_test_by_fold.csv
Causa: você ainda não treinou, ou apagou resultados.
Solução: rode python -m src.train --config config.yaml ou python -m src.rebuild_results_from_ckpts --config config.yaml.

GPU sem memória
Solução: reduza train.batch_size, mantenha train.amp=true e, se necessário, rode em CPU alterando device: cpu no config.yaml (treino fica mais lento).

Resultados diferentes do artigo
Nota: o artigo usa k=5, outro escopo de splits e possivelmente hiperparâmetros distintos. Aqui, a reprodução usa k=3, imagens 224×224, um único backbone (ViT-B/16), e não avalia o split com augmentation global. Diferenças numéricas são esperadas; foque nas tendências relativas entre perdas e splits.

13) Boas práticas de reprodutibilidade neste repositório
Versões fixadas em requirements.txt.

Parâmetros experimentais em config.yaml.

seed controlado, k-fold estratificado, métrica de seleção clara (F1 macro).

Resultados salvos como CSV, com gráficos gerados diretamente dos CSVs.

Scripts independentes para cada etapa (splits, treino, avaliação, figuras, predição, reconstrução de resultados).
