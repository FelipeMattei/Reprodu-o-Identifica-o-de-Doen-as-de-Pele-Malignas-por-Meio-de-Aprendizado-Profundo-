# src/evaluate.py
import pandas as pd
from pathlib import Path

if __name__=="__main__":
    # Diretório onde os CSVs de resultados foram salvos pelo treinamento
    base = Path("./reports/tables")

    # Carrega a planilha consolidada com os resultados de TESTE por split/fold/modelo
    # Espera colunas: ["split","focal","fold","model","Acc","Prec","F1","Rec"]
    df = pd.read_csv(base/"all_results_test_by_fold.csv")

    # =========================
    # RESNET-101 (gera Tabela 1)
    # =========================
    # Filtra apenas linhas do modelo ResNet-101
    r = df[df["model"]=="resnet101"].copy()
    # Converte o booleano 'focal' para string legível na tabela
    r["Focal Loss"] = r["focal"].map({True:"TRUE", False:"FALSE"})

    # Monta linhas da Tabela 1 no formato desejado (uma linha por fold)
    tbl1 = []
    for split in ["PAD-UFES-FULL","PAD-UFES-AUG","PAD-UFES","PAD-UFES-IS"]:
        for foc in ["TRUE","FALSE"]:
            # Seleciona subconjunto por split e configuração de Focal, ordena por fold
            part = r[(r["split"]==split) & (r["Focal Loss"]==foc)].sort_values("fold")
            # Adiciona as métricas de cada fold (fold numerado a partir de 1)
            for _, row in part.iterrows():
                tbl1.append([
                    split,
                    foc,
                    int(row["fold"])+1,
                    row["Acc"],
                    row["Prec"],
                    row["F1"],
                    row["Rec"]
                ])

    # Constrói DataFrame final e salva CSV da Tabela 1
    t1 = pd.DataFrame(tbl1, columns=["Split","Focal Loss","Fold","Acc.","Prec.","F1","Rec."])
    t1.to_csv(base/"table1_resnet101.csv", index=False)

    # =========================
    # ViT (gera Tabela 2)
    # =========================
    # Filtra apenas linhas do modelo ViT-B/16 (chave "vit_base")
    v = df[df["model"]=="vit_base"].copy()
    v["Focal Loss"] = v["focal"].map({True:"TRUE", False:"FALSE"})

    # Monta linhas da Tabela 2 no mesmo formato (uma linha por fold)
    tbl2 = []
    for split in ["PAD-UFES-FULL","PAD-UFES-AUG","PAD-UFES","PAD-UFES-IS"]:
        for foc in ["TRUE","FALSE"]:
            part = v[(v["split"]==split) & (v["Focal Loss"]==foc)].sort_values("fold")
            for _, row in part.iterrows():
                tbl2.append([
                    split,
                    foc,
                    int(row["fold"])+1,
                    row["Acc"],
                    row["Prec"],
                    row["F1"],
                    row["Rec"]
                ])

    # Constrói DataFrame final e salva CSV da Tabela 2
    t2 = pd.DataFrame(tbl2, columns=["Split","Focal Loss","Fold","Acc.","Prec.","F1","Rec."])
    t2.to_csv(base/"table2_vit.csv", index=False)

    # ============================================================
    # Resumo por grupo (para gráficos): média e desvio por grupo
    # Agrupa por (modelo, split, uso de focal) e computa mean/std
    # ============================================================
    grp = df.groupby(["model","split","focal"]).agg(
        Acc_mean=("Acc","mean"),
        Acc_std=("Acc","std"),
        F1_mean=("F1","mean"),
        F1_std=("F1","std")
    ).reset_index()

    # Salva planilha resumida usada pelo script de figuras
    grp.to_csv(base/"summary_means_stds.csv", index=False)

    print("OK -> Tabelas 1/2 e resumo gerados.")
