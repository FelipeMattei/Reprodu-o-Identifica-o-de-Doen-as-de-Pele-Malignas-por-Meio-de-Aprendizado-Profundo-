# src/split_maker.py
import argparse, pandas as pd, shutil
from pathlib import Path
from sklearn.model_selection import train_test_split

def fix_path(p: str) -> str:
    """
    Normaliza caminhos vindos do metadata:
      - troca '\' por '/'
      - remove duplicações do tipo 'imgs_part_1/imgs_part_1/...'
    Retorna o caminho corrigido (string).
    """
    p = p.replace("\\", "/")
    for k in ("imgs_part_1","imgs_part_2","imgs_part_3"):
        p = p.replace(f"{k}/{k}/", f"{k}/")
    return p

def make_pad_ufes(df, seed, out):
    """
    Gera split PAD-UFES:
      - 80% treino / 20% teste (estratificado por label)
      - salva CSVs *_train.csv e *_test.csv no diretório 'out'
    """
    tr, te = train_test_split(df, test_size=0.20, random_state=seed, stratify=df['label'])
    tr.to_csv(out/"PAD-UFES_train.csv", index=False); te.to_csv(out/"PAD-UFES_test.csv", index=False)

def make_pad_ufes_full(df, seed, out):
    """
    Gera split PAD-UFES-FULL:
      - remove 15% aleatoriamente (drop) para evitar redundâncias
      - no restante (keep), faz 85/15 estratificado (equivale a 0.15/0.85 de teste)
    """
    drop = df.sample(frac=0.15, random_state=seed); keep = df.drop(drop.index)
    tr, te = train_test_split(keep, test_size=(0.15/0.85), random_state=seed, stratify=keep['label'])
    tr.to_csv(out/"PAD-UFES-FULL_train.csv", index=False); te.to_csv(out/"PAD-UFES-FULL_test.csv", index=False)

def make_pad_ufes_is(df, seed, out):
    """
    Gera split PAD-UFES-IS:
      - mesmo procedimento do FULL, mas com semente deslocada (seed+42)
      - produz uma partição alternativa (IS) para comparação
    """
    seed2 = seed + 42
    drop = df.sample(frac=0.15, random_state=seed2); keep = df.drop(drop.index)
    tr, te = train_test_split(keep, test_size=(0.15/0.85), random_state=seed2, stratify=keep['label'])
    tr.to_csv(out/"PAD-UFES-IS_train.csv", index=False); te.to_csv(out/"PAD-UFES-IS_test.csv", index=False)

def make_pad_ufes_aug(df, seed, out):
    """
    Gera split PAD-UFES-AUG:
      - mesmo corte 80/20 estratificado do PAD-UFES (sem remover 15%)
      - pensado para experimentos com aumento de dados (augmentation)
    """
    tr, te = train_test_split(df, test_size=0.20, random_state=seed, stratify=df['label'])
    tr.to_csv(out/"PAD-UFES-AUG_train.csv", index=False); te.to_csv(out/"PAD-UFES-AUG_test.csv", index=False)

if __name__ == "__main__":
    # CLI: permite customizar caminhos e semente ao rodar como módulo/script
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", default="data/PAD-UFES-20/metadata.csv")  # CSV original com image_path,label
    ap.add_argument("--root", default="data/PAD-UFES-20")               # pasta raiz do dataset
    ap.add_argument("--out",  default="data/PAD-UFES-20/splits")        # onde salvar os CSVs dos splits
    ap.add_argument("--seed", type=int, default=123)                    # semente base para reprodutibilidade
    a = ap.parse_args()

    ROOT = Path(a.root); META = ROOT / "metadata.csv"; OUT = Path(a.out)

    # 1) Carrega e corrige o metadata (normaliza paths) e salva como ROOT/metadata.csv
    df = pd.read_csv(a.meta)
    assert {"image_path","label"} <= set(df.columns), "metadata.csv deve ter colunas: image_path,label"
    df["image_path"] = df["image_path"].astype(str).map(fix_path)
    df.to_csv(META, index=False)

    # 2) Limpa diretório de splits (se existir) para evitar lixo de execuções anteriores
    if OUT.exists(): shutil.rmtree(OUT)
    OUT.mkdir(parents=True, exist_ok=True)

    # 3) Gera os 4 splits padronizados (arquivos *_train.csv e *_test.csv)
    make_pad_ufes(df, a.seed, OUT)
    make_pad_ufes_full(df, a.seed, OUT)
    make_pad_ufes_is(df, a.seed, OUT)
    make_pad_ufes_aug(df, a.seed, OUT)

    print(f"OK — splits gerados em {OUT}")
