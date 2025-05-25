import torch.nn as nn
import torch.nn.functional as F


class FaceVerificationModel(nn.Module):
    def __init__(self, backbone, embedding_size=256, dropout=0.3):
        super().__init__()
        self.backbone = backbone
        self._freeze_layers()

        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Identity()

        self.embedding_layer = nn.Sequential(
            nn.Linear(in_features, 1024),
            nn.BatchNorm1d(1024),
            nn.PReLU(),
            nn.Dropout(dropout),
            nn.Linear(1024, embedding_size),
        )

    def _freeze_layers(self):
        for param in self.backbone.parameters():
            param.requires_grad = False

        for param in self.backbone.layer4.parameters():
            param.requires_grad = True

    def forward(self, x):
        features = self.backbone(x)
        emb = self.embedding_layer(features)
        return F.normalize(emb, p=2, dim=1)
