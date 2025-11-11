# src/datasets.py
from pathlib import Path
from PIL import Image
import pandas as pd, torch
import torchvision.transforms as T
from torch.utils.data import Dataset


def tf_train(img_size: int = 224, aug: bool = False) -> T.Compose:
    """
    Transforms para TREINO.
    - Redimensiona para (img_size, img_size).
    - Aplica ou não augmentations leves (flip/rotação/afinidade/cor) conforme `aug`.
    - Converte para tensor e normaliza com estatísticas do ImageNet.
    """
    base = [
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(p=0.5) if aug else T.RandomHorizontalFlip(p=0.0),
        T.RandomApply([T.RandomRotation(30, expand=False)], p=0.5 if aug else 0.0),
        T.RandomApply(
            [T.RandomAffine(degrees=0, translate=(20/224, 20/224), shear=17.18, scale=(0.8, 1.2))],
            p=0.5 if aug else 0.0
        ),
        T.RandomApply(
            [T.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.2, hue=0.04)],
            p=0.7 if aug else 0.0
        ),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ]
    return T.Compose(base)


def tf_val(img_size: int = 224) -> T.Compose:
    """
    Transforms para VALIDAÇÃO/TESTE.
    - Apenas resize, tensor e normalização (sem aleatoriedade).
    """
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])


class CSVDataset(Dataset):
    """
    Dataset baseado em CSV com duas colunas obrigatórias:
        - image_path: caminho relativo ao diretório raiz de imagens (`img_root`)
        - label: rótulo textual da classe

    Observações:
    - O mapeamento classe->índice é deduzido a partir do conjunto de classes
      informado (parâmetro `classes`) ou, se ausente, inferido do próprio CSV.
    - Quando `csv_path=None`, os caminhos/rótulos podem ser injetados depois
      via atributos `paths` e `labels` (usado no k-fold em `train.py`).
    """

    def __init__(
        self,
        csv_path,
        img_root,
        classes=None,
        train: bool = False,
        aug: bool = False,
        img_size: int = 224
    ):
        # Diretório raiz onde as imagens residem
        self.root = Path(img_root)

        # Escolha de pipeline de transform conforme fase (train vs val/test)
        self.tfm = tf_train(img_size, aug) if train else tf_val(img_size)

        # Define a lista de classes e o dicionário classe->índice
        if classes is not None:
            # Classes fornecidas externamente (mantém a ordem recebida)
            self.classes = list(classes)
        else:
            # Sem classes fornecidas: infere do CSV (ordem alfabética estável)
            if csv_path is None:
                raise ValueError("Se classes=None, csv_path não pode ser None.")
            df_tmp = pd.read_csv(csv_path)
            self.classes = sorted(df_tmp["label"].astype(str).unique().tolist())
        self.cls2idx = {c: i for i, c in enumerate(self.classes)}

        # Carrega caminhos e rótulos do CSV, ou inicializa para injeção posterior
        if csv_path is None:
            # Modo “vazio”: valores serão preenchidos depois (ex.: splits do k-fold)
            self.paths, self.labels = [], []
        else:
            df = pd.read_csv(csv_path)
            self.paths = df["image_path"].tolist()
            labs = df["label"].astype(str).tolist()
            self.labels = [self.cls2idx[c] for c in labs]

    def __len__(self) -> int:
        # Número de amostras disponíveis
        return len(self.paths)

    def __getitem__(self, i: int):
        """
        Retorna (x, y) em que:
        - x: imagem processada pelos transforms (tensor normalizado)
        - y: índice da classe (tensor long)
        """
        im = Image.open(self.root / self.paths[i]).convert("RGB")
        x = self.tfm(im)
        y = torch.tensor(self.labels[i], dtype=torch.long)
        return x, y
