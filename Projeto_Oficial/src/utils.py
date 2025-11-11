# src/utils.py
import numpy as np, torch
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

def macro_metrics(y_true, y_pred):
    """
    Calcula métricas de classificação no esquema macro (média não ponderada entre classes).
    Params:
      y_true: lista/array de rótulos verdadeiros (inteiros)
      y_pred: lista/array de rótulos previstos (inteiros)
    Returns:
      (acc, p, r, f1): tupla com acurácia, precisão macro, recall macro e F1 macro.
    Obs.:
      - average="macro" dá o mesmo peso a cada classe, útil sob desbalanceamento.
      - zero_division=0 evita warnings quando alguma classe não aparece nas previsões.
    """
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="macro", zero_division=0
    )
    return acc, p, r, f1


class EarlyStopper:
    """
    Controle simples de early stopping baseado em uma métrica de validação.
    - 'mode="max"': melhora quando o valor aumenta (ex.: F1, acurácia).
    - 'mode="min"': melhora quando o valor diminui (ex.: perda).
    Para cada chamada a step(val):
      - zera o contador se houve melhora e atualiza 'best'
      - caso contrário, incrementa o contador
    should_stop() retorna True quando 'patience' épocas se passaram sem melhora.
    """
    def __init__(self, patience=20, mode="max"):
        self.patience, self.mode = patience, mode
        # valor inicial do melhor: -inf para 'max', +inf para 'min'
        self.best = -np.inf if mode == "max" else np.inf
        self.count = 0  # épocas consecutivas sem melhora

    def step(self, val):
        """
        Atualiza estado do early stopping dado o valor de validação 'val'.
        Returns:
          True se houve melhora (e 'best' foi atualizado), False caso contrário.
        """
        improved = (val > self.best) if self.mode == "max" else (val < self.best)
        if improved:
            self.best = val
            self.count = 0
            return True
        self.count += 1
        return False

    def should_stop(self):
        """
        Indica se atingiu o limite de paciência sem melhora.
        """
        return self.count >= self.patience
