# src/predict.py
import argparse, torch
from pathlib import Path
from PIL import Image
import pandas as pd
import torchvision.transforms as T
from src.models import build_model

def tfm(size=224):
    """
    Pipeline de pré-processamento para inferência:
    - Redimensiona para (size, size)
    - Converte para tensor [0,1]
    - Normaliza com estatísticas do ImageNet
    """
    return T.Compose([
        T.Resize((size, size)),
        T.ToTensor(),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

@torch.no_grad()  # desativa gradientes (inferência mais leve/rápida)
def main():
    # ----------------------
    # 1) Argumentos de CLI
    # ----------------------
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", nargs="+", required=True)  # lista de caminhos de imagens para predizer
    ap.add_argument("--ckpt", required=True)               # caminho do checkpoint (.pt) salvo no treino
    ap.add_argument("--model", default="vit_base")         # backbone: "vit_base" (padrão) ou "resnet101"
    ap.add_argument("--metadata", default="./data/PAD-UFES-20/metadata.csv")  # usado para descobrir as classes
    ap.add_argument("--device", default="cuda")            # "cuda" ou "cpu"
    ap.add_argument("--img-size", type=int, default=224)   # resolução de entrada
    a = ap.parse_args()

    # -----------------------------------------
    # 2) Descobre classes a partir do metadata
    #    (ordem ordenada lexicograficamente)
    # -----------------------------------------
    classes = sorted(pd.read_csv(a.metadata)["label"].astype(str).unique().tolist())

    # ---------------------------------------------------
    # 3) Constrói o modelo e carrega pesos do checkpoint
    #    - pretrained=False porque vamos substituir pelo ckpt
    #    - ckpt pode estar sob {"model": state_dict} ou direto
    # ---------------------------------------------------
    model = build_model(a.model, num_classes=len(classes), pretrained=False)
    sd = torch.load(a.ckpt, map_location=a.device)
    sd = sd.get("model", sd)  # suporta checkpoints salvos como {"model": state_dict}
    model.load_state_dict(sd)
    model.to(a.device).eval()  # envia ao device e coloca em modo avaliação

    # -----------------------------------
    # 4) Transforms de entrada (inferência)
    # -----------------------------------
    tr = tfm(a.img_size)

    # -----------------------------------
    # 5) Loop de predição nas imagens
    #    - abre a imagem em RGB
    #    - aplica transform
    #    - faz forward e softmax
    #    - imprime Top-3 classes com probabilidades
    # -----------------------------------
    for p in a.images:
        im = tr(Image.open(p).convert("RGB")).unsqueeze(0).to(a.device)  # [1,C,H,W]
        probs = torch.softmax(model(im), dim=1)[0]                        # vetor de probs por classe
        topv, topi = probs.topk(3)                                        # top-3
        print(f"\n{p}:")
        for r, (i, v) in enumerate(zip(topi.tolist(), topv.tolist()), 1):
            print(f"  {r}. {classes[i]}  {v:.4f}")

# Executa apenas quando chamado via CLI: `python -m src.predict ...`
if __name__=="__main__":
    main()
