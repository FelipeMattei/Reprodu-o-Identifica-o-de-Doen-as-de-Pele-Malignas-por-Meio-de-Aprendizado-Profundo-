# src/models.py
import torch, torchvision.models as M
import torch.nn as nn

def build_model(name, num_classes, pretrained=True):
    """
    Constrói e devolve um modelo de classificação de imagens a partir do nome.

    Parâmetros
    ----------
    name : str
        Identificador do backbone. Aceita "resnet101" ou "vit_base".
    num_classes : int
        Número de classes de saída para ajustar a última camada (cabeça) do modelo.
    pretrained : bool
        Se True, carrega pesos pré-treinados no ImageNet (torchvision).
        Se False, inicializa pesos aleatoriamente.

    Retorno
    -------
    torch.nn.Module
        Modelo com a cabeça final ajustada para 'num_classes'.
    """
    if name == "resnet101":
        # Carrega ResNet-101 do torchvision; usa pesos do ImageNet se 'pretrained' for True.
        m = M.resnet101(weights=M.ResNet101_Weights.IMAGENET1K_V2 if pretrained else None)
        # Substitui a fully-connected final (m.fc) para ter 'num_classes' saídas.
        m.fc = nn.Linear(m.fc.in_features, num_classes)
        return m

    elif name == "vit_base":
        # Carrega Vision Transformer Base/16 (ViT-B/16) do torchvision 0.19;
        # usa pesos do ImageNet se 'pretrained' for True.
        m = M.vit_b_16(weights=M.ViT_B_16_Weights.IMAGENET1K_V1 if pretrained else None)
        # Substitui a cabeça de classificação (m.heads.head) para 'num_classes' saídas.
        m.heads.head = nn.Linear(m.heads.head.in_features, num_classes)
        return m

    else:
        # Proteção: nome inválido de modelo.
        raise ValueError("model deve ser 'resnet101' ou 'vit_base'")
