# src/losses.py
import torch, torch.nn as nn, torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Implementação da Focal Loss para classificação multiclasse.
    Ideia geral:
      - Penalizar mais forte os exemplos difíceis (onde p_t é baixo)
      - Reduzir o peso dos exemplos fáceis (onde p_t é alto)
    Parâmetros:
      alpha: fator de balanceamento entre classes (escala o CE)
      gamma: fator de focalização; quanto maior, mais foco em erros
      reduction: "mean" (média) ou soma dos termos individuais
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction

    def forward(self, logits, target):
        """
        logits: tensor [B, C] com scores antes do softmax
        target: tensor [B] com rótulos inteiros (0..C-1)
        Passos:
          1) CE por amostra, sem redução.
          2) pt = exp(-CE) = probabilidade atribuída à classe correta.
          3) FocalLoss = alpha * (1 - pt)^gamma * CE.
          4) Agrega por média (ou soma) conforme `reduction`.
        """
        # CE por amostra (sem reduzir) para obter pt corretamente
        ce = F.cross_entropy(logits, target, reduction="none")

        # pt é a probabilidade prevista para a classe correta
        # (derivada da CE: pt = exp(-CE))
        pt = torch.exp(-ce)

        # termo focal: down-weight em exemplos fáceis (pt alto)
        loss = self.alpha * (1 - pt) ** self.gamma * ce

        # redução final (média por padrão)
        return loss.mean() if self.reduction == "mean" else loss.sum()
