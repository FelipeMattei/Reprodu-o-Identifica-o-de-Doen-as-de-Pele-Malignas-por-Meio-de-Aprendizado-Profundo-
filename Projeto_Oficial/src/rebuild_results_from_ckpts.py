# src/rebuild_results_from_ckpts.py
import re, argparse
from pathlib import Path
import pandas as pd, torch
from torch.utils.data import DataLoader

from src.models import build_model
from src.datasets import CSVDataset
from src.utils import macro_metrics

def parse_ckpt_name(name: str):
    """
    Extrai metadados a partir do nome do checkpoint.
    Espera padrão: <model>_<split>_fold<k>_<focal|nofocal>.pt
    Ex.: vit_base_PAD-UFES-FULL_fold1_focal.pt

    Retorna dict com:
      - model: "vit_base" ou "resnet101"
      - split: nome do split (ex.: "PAD-UFES-FULL")
      - fold: int do fold
      - focal: bool (True se "focal", False se "nofocal")
    Se não bater o padrão, retorna None.
    """
    m = re.match(
        r'(?P<model>vit_base|resnet101)_(?P<split>[^_]+)_fold(?P<fold>\d+)_(?P<focal>focal|nofocal)\.pt',
        name
    )
    if not m:
        return None
    d = m.groupdict()
    d["fold"] = int(d["fold"])
    d["focal"] = (d["focal"] == "focal")
    return d

@torch.no_grad()
def eval_ckpt(ckpt_path: Path, split_csv_test: Path, img_root: Path,
              classes: list, batch_size: int, device: torch.device, img_size: int):
    """
    Recarrega um checkpoint e avalia no CSV de TEST do split correspondente.

    Passos:
      1) Monta dataset de teste a partir do CSV informado (paths + labels).
      2) Instancia o modelo no device e carrega state_dict do checkpoint.
      3) Faz forward em todo o teste (sem gradiente).
      4) Retorna métricas macro (Acc, Prec, Rec, F1).
    """
    ds_te = CSVDataset(str(split_csv_test), img_root, classes=classes,
                       train=False, aug=False, img_size=img_size)
    dl_te = DataLoader(ds_te, batch_size=batch_size, shuffle=False,
                       num_workers=2, pin_memory=True)

    # Constrói o modelo de acordo com o nome presente no checkpoint
    meta = parse_ckpt_name(ckpt_path.name)
    model = build_model(meta["model"], num_classes=len(classes), pretrained=False).to(device)

    # Carrega state_dict salvo (aceita {"model": state_dict} ou state_dict direto)
    sd = torch.load(ckpt_path, map_location=device)
    if "model" in sd:
        sd = sd["model"]
    model.load_state_dict(sd)
    model.eval()

    # Inferência e coleta de predições
    y_true, y_pred = [], []
    for x, y in dl_te:
        x = x.to(device)
        logits = model(x)
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(1).tolist())

    # Calcula métricas macro
    acc, p, r, f1 = macro_metrics(y_true, y_pred)
    return {"Acc": float(acc), "Prec": float(p), "Rec": float(r), "F1": float(f1)}

def main():
    """
    Varre a pasta de checkpoints, reavalia cada .pt no respectivo TEST split,
    e reconstrói o arquivo unificado 'all_results_test_by_fold.csv'.

    Útil quando:
      - o treino já foi feito e você só tem os .pt;
      - o CSV de resultados foi perdido/corrompido e precisa ser regenerado.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")  # usa para ler device, batch e img_size
    args = ap.parse_args()

    # Lê configurações globais (device, caminhos, batch_size, img_size)
    import yaml
    cfg = yaml.safe_load(open(args.config, "r"))
    device = torch.device(cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")
    img_root = Path(cfg["data"]["root"])
    split_dir = Path(cfg["data"]["split_dir"])
    batch_size = int(cfg["train"]["batch_size"])
    img_size = int(cfg["data"]["img_size"])

    # Garante diretórios de saída
    save_dir = Path(cfg["log"]["save_dir"])
    (save_dir / "tables").mkdir(parents=True, exist_ok=True)

    ckpt_dir = save_dir / "checkpoints"
    rows = []

    # Percorre todos os .pt do diretório
    for ck in sorted(ckpt_dir.glob("*.pt")):
        meta = parse_ckpt_name(ck.name)
        if meta is None:
            # Ignora arquivos que não seguem o padrão de nome esperado
            continue

        split = meta["split"]

        # Localiza CSVs de treino/teste do split (mesma convenção do train.py)
        train_csv = split_dir / f"{split}_train.csv"
        test_csv  = split_dir / f"{split}_test.csv"
        if not (train_csv.exists() and test_csv.exists()):
            print(f"[skip] CSVs ausentes para {split}")
            continue

        # Descobre as classes a partir do CSV de treino (garante ordem consistente)
        cls_df = pd.read_csv(train_csv)
        classes = sorted(cls_df["label"].astype(str).unique().tolist())

        print(f"[eval] {ck.name}  ->  {split} test")
        res = eval_ckpt(ck, test_csv, img_root, classes, batch_size, device, img_size)

        # Guarda linha de resultado no formato usado pelo pipeline
        rows.append({
            "split": split,
            "focal": meta["focal"],
            "fold": meta["fold"],
            "model": meta["model"],
            **res
        })

    if not rows:
        # Sem arquivos válidos ou sem CSVs correspondentes
        print("Nenhum resultado gerado. Verifique nomes dos checkpoints.")
        return

    # Ordena e salva CSV final para reuso pelos scripts de avaliação/gráficos
    df = pd.DataFrame(rows).sort_values(["model","split","focal","fold"]).reset_index(drop=True)
    out_csv = save_dir / "tables" / "all_results_test_by_fold.csv"
    df.to_csv(out_csv, index=False)
    print(f"OK -> {out_csv}")

if __name__ == "__main__":
    main()
