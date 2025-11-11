# src/train.py
import argparse, os, random, contextlib
import numpy as np, pandas as pd, torch
from pathlib import Path
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedKFold
from tqdm import tqdm

from src.datasets import CSVDataset
from src.models import build_model
from src.losses import FocalLoss
from src.utils import macro_metrics, EarlyStopper

# AMP moderno (precisão mista automática em CUDA)
from torch import amp as torch_amp


def seed_all(s: int):
    """
    Define semente única para Python, NumPy e PyTorch (CPU/GPU),
    garantindo reprodutibilidade de particionamento e inicializações.
    """
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)


def train_one_epoch(model, dl, loss_fn, opt, device, scaler=None):
    """
    Executa 1 época de treinamento:
      - forward -> loss -> backward -> step (com AMP se 'scaler' existir)
      - acumula métricas macro no conjunto de treino
    Retorna: média das perdas, (acc, prec, rec, f1) macro no epoch.
    """
    model.train()
    losses, y_true, y_pred = [], [], []
    iterator = tqdm(dl, leave=False, desc="train")
    for x, y in iterator:
        x, y = x.to(device), y.to(device)
        opt.zero_grad(set_to_none=True)
        # Usa autocast somente se houver scaler (AMP ativo), caso contrário, contexto nulo
        ctx = torch_amp.autocast("cuda") if (scaler is not None) else contextlib.nullcontext()
        with ctx:
            logits = model(x)
            loss = loss_fn(logits, y)

        # Backprop com/sem AMP
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
        else:
            loss.backward()
            opt.step()

        # Coleta para média de loss e métricas macro de treino
        losses.append(loss.item())
        yp = logits.argmax(1).tolist()
        y_true.extend(y.tolist())
        y_pred.extend(yp)

    acc, p, r, f1 = macro_metrics(y_true, y_pred)
    return float(np.mean(losses)), acc, p, r, f1


@torch.no_grad()
def evaluate(model, dl, device):
    """
    Avaliação (sem gradiente) em um DataLoader:
      - produz (acc, prec, rec, f1) macro no conjunto fornecido.
    """
    model.eval()
    y_true, y_pred = [], []
    for x, y in dl:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        y_true.extend(y.tolist())
        y_pred.extend(logits.argmax(1).tolist())
    return macro_metrics(y_true, y_pred)


if __name__ == "__main__":
    # ======== Leitura da configuração (YAML) + overrides na linha de comando ========
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)      # caminho do arquivo config.yaml
    ap.add_argument("overrides", nargs="*", default=[])  # ex.: train.lr=3e-4 data.img_size=256
    args = ap.parse_args()

    # carrega yaml simples
    import yaml
    cfg = yaml.safe_load(open(args.config, "r"))

    # aplica overrides do tipo section.key=value (ex.: "train.lr=1e-4")
    # tenta converter para bool/num quando aplicável; caso contrário, mantém string
    for kv in args.overrides:
        k, v = kv.split("=", 1)
        if v.lower() in ["true", "false"]:
            v = v.lower() == "true"
        else:
            try:
                v_num = float(v)
                v = int(v_num) if v_num.is_integer() else v_num
            except:
                pass
        sect, key = k.split(".")
        cfg[sect][key] = v

    # ======== Semente e dispositivo ========
    seed_all(int(cfg["seed"]))
    device = torch.device(cfg["device"] if torch.cuda.is_available() and cfg["device"] == "cuda" else "cpu")

    # ======== Caminhos e hiperparâmetros globais ========
    root = Path(cfg["data"]["root"])
    split_dir = Path(cfg["data"]["split_dir"])
    img_size = int(cfg["data"]["img_size"])
    num_workers = int(cfg["data"]["num_workers"])
    k_folds = int(cfg["train"]["k_folds"])

    # Pastas de saída (checkpoints, tabelas e figuras)
    save_dir = Path(cfg["log"]["save_dir"])
    (save_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (save_dir / "tables").mkdir(parents=True, exist_ok=True)
    (save_dir / "figs").mkdir(parents=True, exist_ok=True)

    # Splits padronizados do projeto (serão checados na pasta de splits)
    splits = ["PAD-UFES", "PAD-UFES-FULL", "PAD-UFES-IS", "PAD-UFES-AUG"]

    rows = []  # acumula resultados finais em TEST por (split, focal, fold, modelo)
    for split in splits:
        train_csv = split_dir / f"{split}_train.csv"
        test_csv = split_dir / f"{split}_test.csv"
        # Garante que os CSVs do split existem; orienta gerar via split_maker/rebuild
        assert train_csv.exists() and test_csv.exists(), f"Split {split} ausente. Gere com split_maker/rebuild_splits."

        # Descobre classes a partir do treino (ordem alfabética para consistência)
        cls_df = pd.read_csv(train_csv)
        classes = sorted(cls_df["label"].astype(str).unique().tolist())

        # Listas para StratifiedKFold: caminhos (X) e rótulos (y) do conjunto de treino
        X = cls_df["image_path"].tolist()
        y = cls_df["label"].astype(str).tolist()
        skf = StratifiedKFold(n_splits=k_folds, shuffle=True, random_state=int(cfg["seed"]))

        # Modelos a treinar (lista em cfg['train'].models ou único em cfg['train'].model)
        models_cfg = cfg["train"].get("models") or [cfg["train"].get("model", "vit_base")]
        for model_name in models_cfg:
            # Laço sobre uso de Focal (True/False) mantendo tudo igual para comparação justa
            for use_focal in [True, False]:
                # K-fold estratificado: separa tr/va dentro do conjunto de treino
                for fold_id, (tr_idx, va_idx) in enumerate(skf.split(X, y)):
                    df_tr = cls_df.iloc[tr_idx].reset_index(drop=True)
                    df_va = cls_df.iloc[va_idx].reset_index(drop=True)

                    # No split PAD-UFES-AUG ativamos as aug do dataset (como proxy de oversampling)
                    aug = (split == "PAD-UFES-AUG")

                    # Datasets: injeção direta de paths/labels para tr e va
                    ds_tr = CSVDataset(csv_path=None, img_root=root, classes=classes, train=True, aug=aug, img_size=img_size)
                    ds_va = CSVDataset(csv_path=None, img_root=root, classes=classes, train=False, aug=False, img_size=img_size)
                    ds_tr.paths = df_tr["image_path"].tolist()
                    ds_tr.labels = [ds_tr.cls2idx[s] for s in df_tr["label"].astype(str)]
                    ds_va.paths = df_va["image_path"].tolist()
                    ds_va.labels = [ds_va.cls2idx[s] for s in df_va["label"].astype(str)]

                    # DataLoaders com shuffle no treino; pin_memory acelera em CUDA
                    dl_tr = DataLoader(ds_tr, batch_size=int(cfg["train"]["batch_size"]), shuffle=True,
                                       num_workers=num_workers, pin_memory=True)
                    dl_va = DataLoader(ds_va, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                                       num_workers=num_workers, pin_memory=True)

                    # Constrói o modelo com a cabeça ajustada ao nº de classes e envia ao device
                    model = build_model(model_name, num_classes=len(classes), pretrained=bool(cfg["train"]["pretrained"]))
                    model = model.to(device)

                    # Otimizador:
                    #  - Se cfg['train']['optimizer'] == 'sgd', usa SGD com momentum e weight decay (perfil paper)
                    #  - Caso contrário, usa AdamW (perfil default para ViT via override)
                    if str(cfg["train"]["optimizer"]).lower() == "sgd":
                        opt = torch.optim.SGD(
                            model.parameters(),
                            lr=float(cfg["train"]["lr"]),
                            momentum=float(cfg["train"]["momentum"]),
                            weight_decay=float(cfg["train"]["weight_decay"])
                        )
                    else:
                        opt = torch.optim.AdamW(
                            model.parameters(),
                            lr=float(cfg.get("train", {}).get("lr", 3e-4)),
                            weight_decay=float(cfg.get("train", {}).get("weight_decay", 0.05))
                        )

                    # Função de perda: FocalLoss (α, γ) ou CrossEntropy padrão (NF)
                    loss_fn = FocalLoss(alpha=float(cfg["train"]["focal_loss"]["alpha"]),
                                        gamma=float(cfg["train"]["focal_loss"]["gamma"])) if use_focal \
                              else torch.nn.CrossEntropyLoss()

                    # AMP (GradScaler) apenas se torch.cuda disponível e cfg['train']['amp'] True
                    scaler = torch_amp.GradScaler("cuda") if (bool(cfg["train"]["amp"]) and device.type == "cuda") else None

                    # Early stopping monitora F1 macro de validação; 'mode=max'
                    stopper = EarlyStopper(patience=int(cfg["train"]["early_stopping_patience"]), mode="max")

                    # Nome único do run e CSV de log por época (para auditoria)
                    run_name = f"{model_name}_{split}_fold{fold_id}_{'focal' if use_focal else 'nofocal'}"
                    print(f"\n=== [{split}] model={model_name} focal={use_focal} fold={fold_id} ===")
                    elog = (save_dir / "tables" / f"trainlog_{run_name}.csv")
                    with open(elog, "w", encoding="utf-8") as f:
                        f.write("epoch,train_loss,train_acc,train_prec,train_rec,train_f1,val_acc,val_prec,val_rec,val_f1\n")

                    # Loop de épocas: retém melhor estado (maior F1 de validação)
                    best_f1, best_state = -1.0, None
                    max_epochs = int(cfg["train"]["max_epochs"])

                    for ep in range(max_epochs):
                        tr_loss, tr_acc, tr_p, tr_r, tr_f1 = train_one_epoch(model, dl_tr, loss_fn, opt, device, scaler)
                        va_acc, va_p, va_r, va_f1 = evaluate(model, dl_va, device)

                        improved = stopper.step(va_f1)
                        if improved:
                            best_f1 = va_f1
                            best_state = {"model": model.state_dict()}

                        # Log no console e no CSV (uma linha por época)
                        print(f"[{run_name}] ep={ep:03d} tr_loss={tr_loss:.4f} tr_F1={tr_f1:.4f} "
                              f"val_F1={va_f1:.4f} val_acc={va_acc:.4f}")
                        with open(elog, "a", encoding="utf-8") as f:
                            f.write(f"{ep},{tr_loss:.6f},{tr_acc:.6f},{tr_p:.6f},{tr_r:.6f},{tr_f1:.6f},"
                                    f"{va_acc:.6f},{va_p:.6f},{va_r:.6f},{va_f1:.6f}\n")

                        # Para antecipadamente quando não há melhora por 'patience' épocas
                        if stopper.should_stop():
                            print(f"[{run_name}] early stopping (best val_F1={best_f1:.4f})")
                            break

                    # Salva checkpoint do melhor epoch (dict com chave "model")
                    ck = save_dir / "checkpoints" / f"{run_name}.pt"
                    torch.save(best_state, ck)

                    # Avaliação final no TEST do split (modelo reinstanciado e carregado do ckpt)
                    ds_te = CSVDataset(str(test_csv), root, classes=classes, train=False, aug=False, img_size=img_size)
                    dl_te = DataLoader(ds_te, batch_size=int(cfg["train"]["batch_size"]), shuffle=False,
                                       num_workers=num_workers, pin_memory=True)

                    model_final = build_model(model_name, num_classes=len(classes), pretrained=False).to(device)
                    model_final.load_state_dict(torch.load(ck, map_location=device)["model"])
                    te_acc, te_p, te_r, te_f1 = evaluate(model_final, dl_te, device)

                    # Acumula linha de resultados para o CSV geral (um registro por fold)
                    rows.append({
                        "split": split, "focal": use_focal, "fold": int(fold_id), "model": model_name,
                        "Acc": float(te_acc), "Prec": float(te_p), "F1": float(te_f1), "Rec": float(te_r)
                    })

    # ======== Consolida e salva resultados por fold em TEST ========
    df = pd.DataFrame(rows)
    out_csv = save_dir / "tables" / "all_results_test_by_fold.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nOK -> {out_csv}")
