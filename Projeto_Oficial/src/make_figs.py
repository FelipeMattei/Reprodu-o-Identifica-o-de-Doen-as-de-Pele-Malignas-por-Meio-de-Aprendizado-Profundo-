# src/make_figs.py
from pathlib import Path
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # Pastas padrão de entrada (tabelas) e saída (figuras)
    tdir = Path("./reports/tables")
    fdir = Path("./reports/figs")
    fdir.mkdir(parents=True, exist_ok=True)

    # Arquivos de entrada gerados por src.evaluate.py
    summary_path = tdir / "summary_means_stds.csv"       # médias/desvios por (modelo, split, focal)
    results_path = tdir / "all_results_test_by_fold.csv"  # resultados por fold

    # Garante que o CSV bruto de resultados exista
    if not results_path.exists():
        raise FileNotFoundError(
            f"Arquivo não encontrado: {results_path}\n"
            "Rode primeiro: python -m src.evaluate"
        )

    # Carrega resultados por fold
    df_all = pd.read_csv(results_path)

    # Se houver o resumo pronto (médias e desvios), usa; caso contrário, agrupa aqui
    if summary_path.exists():
        df_sum = pd.read_csv(summary_path)
    else:
        df_sum = (
            df_all
            .groupby(["model", "split", "focal"], as_index=False)
            .agg(
                Acc_mean=("Acc","mean"), Acc_std=("Acc","std"),
                F1_mean=("F1","mean"), F1_std=("F1","std")
            )
        )

    # Descobre automaticamente o número de folds (assumindo numeração 0..k-1)
    if "fold" in df_all.columns and len(df_all):
        k_folds = df_all["fold"].max() + 1
    else:
        k_folds = None  # caso excepcional

    # Normaliza tipos para garantir consistência (True/False)
    df_sum["focal"] = df_sum["focal"].astype(bool)
    df_all["focal"] = df_all["focal"].astype(bool)

    # Ordem preferida dos splits; mantém apenas os que existem no dataset atual
    split_order_pref = ["PAD-UFES-FULL", "PAD-UFES", "PAD-UFES-IS", "PAD-UFES-AUG"]
    split_order = [s for s in split_order_pref if s in df_sum["split"].unique().tolist()]

    # Lista de modelos presentes (ex.: "vit_base" e/ou "resnet101")
    models = df_sum["model"].unique().tolist()

    def focal_label(b: bool) -> str:
        """Converte booleano de focal em rótulo curto para legendas/nomes de arquivo."""
        return "F" if b else "NF"

    # ------------------------------------------------------------------------------------
    # 1) Gráficos de BARRAS com erro (média ± desvio) para Acc e F1 por split x Focal
    #    Um arquivo por modelo e por métrica, barras lado a lado (NF vs F) em cada split.
    # ------------------------------------------------------------------------------------
    for metric, ylabel in [("Acc_mean","Acurácia"), ("F1_mean","F1 (macro)")]:
        for m in models:
            sub = df_sum[df_sum["model"] == m].copy()
            # Garante a ordem categórica dos splits
            sub["split"] = pd.Categorical(sub["split"], categories=split_order, ordered=True)
            sub = sub.sort_values(["split","focal"])

            # Posições das barras por split
            x = np.arange(len(split_order))
            width = 0.35

            # Separa F e NF para montar as séries
            sub_f  = sub[sub["focal"] == True]
            sub_nf = sub[sub["focal"] == False]

            # Extrai médias/desvios alinhados por split (ou NaN/0 quando não houver)
            f_means = [sub_f[sub_f["split"] == s][metric].values[0] if s in sub_f["split"].values else np.nan for s in split_order]
            nf_means = [sub_nf[sub_nf["split"] == s][metric].values[0] if s in sub_nf["split"].values else np.nan for s in split_order]
            f_stds  = [sub_f[sub_f["split"] == s][metric.replace("_mean","_std")].values[0] if s in sub_f["split"].values else 0.0 for s in split_order]
            nf_stds = [sub_nf[sub_nf["split"] == s][metric.replace("_mean","_std")].values[0] if s in sub_nf["split"].values else 0.0 for s in split_order]

            plt.figure(figsize=(9, 4.5))
            plt.bar(x - width/2, nf_means, width, yerr=nf_stds, capsize=4, label="NF", alpha=0.9)
            plt.bar(x + width/2, f_means,  width, yerr=f_stds,  capsize=4, label="F",  alpha=0.9)

            plt.xticks(x, split_order, rotation=0)
            title = f"{ylabel} por split (modelo: {m}"
            if k_folds: title += f", média ± desvio em {k_folds}-fold"
            title += ")"
            plt.title(title)
            plt.ylabel(ylabel)
            plt.ylim(0, 1.0)  # métricas normalizadas
            plt.legend(title="Focal")
            plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
            out = fdir / f"bar_{metric}_{m}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()

    # ------------------------------------------------------------------------------------
    # 2) BOXLOT de F1 por split mostrando distribuição entre folds
    #    Um arquivo por modelo, com dois painéis: NF (esq) e F (dir).
    # ------------------------------------------------------------------------------------
    for m in models:
        sub = df_all[df_all["model"] == m].copy()
        sub["split"] = pd.Categorical(sub["split"], categories=split_order, ordered=True)
        plt.figure(figsize=(10, 4.5))

        for i, foc in enumerate([False, True], start=1):
            ax = plt.subplot(1, 2, i)
            sub_foc = sub[sub["focal"] == foc]
            # Lista de vetores de F1 por split (na ordem desejada)
            data = [sub_foc[sub_foc["split"] == s]["F1"].values for s in split_order]
            ax.boxplot(data, labels=split_order, showmeans=True)
            ax.set_title(f"F1 por split — Focal: {focal_label(foc)} (modelo: {m})")
            ax.set_ylim(0, 1.0)
            ax.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)

        sup = f"Distribuição de F1 por split e focal (modelo: {m}"
        sup += f", k={k_folds})" if k_folds else ")"
        plt.suptitle(sup)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        out = fdir / f"box_F1_by_split_{m}.png"
        plt.savefig(out, dpi=150)
        plt.close()

    # ------------------------------------------------------------------------------------
    # 3) LINHAS por fold: F1 por split, cada linha representa um fold
    #    Gera um arquivo por modelo e por condição de Focal (NF/F).
    # ------------------------------------------------------------------------------------
    for m in models:
        sub = df_all[df_all["model"] == m].copy()
        sub["split"] = pd.Categorical(sub["split"], categories=split_order, ordered=True)

        for foc in [False, True]:
            sub_f = sub[sub["focal"] == foc].copy()
            if sub_f.empty:
                continue

            plt.figure(figsize=(9, 4.5))
            # Uma linha por fold, ordenando os pontos pela ordem de splits
            for fold_id, grp in sub_f.groupby("fold"):
                grp = grp.sort_values("split")
                plt.plot(range(len(grp)), grp["F1"].values, marker="o", label=f"fold {fold_id}")

            plt.xticks(range(len(split_order)), split_order)
            plt.ylim(0, 1.0)
            plt.ylabel("F1 (macro)")
            plt.title(f"F1 por split (linhas por fold) — modelo: {m}, Focal: {focal_label(foc)}")
            plt.grid(axis="y", linestyle="--", linewidth=0.6, alpha=0.5)
            plt.legend(ncol=2, fontsize=8)
            out = fdir / f"line_F1_by_split_folds_{m}_{focal_label(foc)}.png"
            plt.tight_layout()
            plt.savefig(out, dpi=150)
            plt.close()

    # ------------------------------------------------------------------------------------
    # 4) CSV-resumo: melhor configuração por split (maior F1_mean)
    #    Útil para citar rapidamente qual combinação (modelo, focal) lidera em cada split.
    # ------------------------------------------------------------------------------------
    best_rows = []
    for s in split_order:
        ss = df_sum[df_sum["split"] == s]
        if ss.empty:
            continue
        row = ss.sort_values("F1_mean", ascending=False).iloc[0]
        best_rows.append({
            "split": s,
            "model": row["model"],
            "focal": focal_label(bool(row["focal"])),
            "F1_mean": round(float(row["F1_mean"]), 4),
            "F1_std": round(float(row["F1_std"]), 4),
            "Acc_mean": round(float(row["Acc_mean"]), 4),
            "Acc_std": round(float(row["Acc_std"]), 4),
        })

    if best_rows:
        df_best = pd.DataFrame(best_rows)
        df_best.to_csv(tdir / "top_configs_by_split.csv", index=False)

    print("OK -> gráficos gerados em reports/figs e resumo em reports/tables/top_configs_by_split.csv")
