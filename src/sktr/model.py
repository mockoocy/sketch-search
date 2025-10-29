from typing import Protocol
import torch
import torch.nn as nn
import timm

from sktr.config.config import DEVICE
from sktr.type_defs import (
    EmbeddingBatch,
    Loss,
    RGBTorchBatch,
    GrayTorchBatch,
)


class TimmCovolutionalModel(Protocol):
    def forward_features(self, x: torch.Tensor) -> torch.Tensor: ...

    def forward(self, x: torch.Tensor) -> torch.Tensor: ...


class TimmBackbone(nn.Module):
    """
    Wrapper around a timm model that outputs a *global* feature vector.

    This uses timm to build a model with the classifier removed and
    global average pooling enabled, so forward(x) → [B, C].

    Args:
        name: timm model name (e.g. 'resnet50', 'tresnet_m')
        pretrained: load ImageNet weights
    Attributes:
        feature_dim: dimensionality C of the pooled feature
    """

    def __init__(self, name: str, pretrained: bool = True):
        super().__init__()
        self.model = timm.create_model(
            name, pretrained=pretrained, num_classes=0, global_pool="avg"
        )
        self.feature_dim = self.model.num_features

    def _gray_to_rgb(self, image: GrayTorchBatch) -> RGBTorchBatch:
        return image.unsqueeze(1).repeat(1, 3, 1, 1)

    def forward(self, image: RGBTorchBatch | GrayTorchBatch) -> EmbeddingBatch:
        """
        Args:
            image: Tensor representing a batch of rgb or
                grayscale images.
        Returns:
            features: [B, C] pooled features (float32)
        """
        # [B, H, W] -> [B, 3, H, W]
        # TODO: use ndim, because it'd break for batch_size of 3 :D
        return self.model(image if image.shape[1] == 3 else self._gray_to_rgb(image))

    def forward_features(self, image: RGBTorchBatch | GrayTorchBatch) -> torch.Tensor:
        """
        Args:
            image: Tensor representing a batch of rgb or
                grayscale images.

        Returns:
            features: [B, C, H_out, W_out] unpooled features (float32)
        """
        return self.model.forward_features(image if image.shape[1] == 3 else self._gray_to_rgb(image))


def dcl_loss(
    photo_embeddings: EmbeddingBatch,
    sketch_embeddings: EmbeddingBatch,
    temperature: float = 0.2,
) -> Loss:
    """
    Computes the DCL loss between photo and sketch embeddings.

    This is a "two-way" DCL loss, which means that it is computed for
    both photo and sketch embeddings, treating them as anchors,
    instead of just one of them.
    Args:
        photo_embeddings: embeddings for photos
        sketch_embeddings: embeddings for sketches
        temperature: temperature parameter for scaling the logits
    Returns:
        Computed DCL loss.
    """
    photo_embeddings = nn.functional.normalize(photo_embeddings, dim=1)  # [B, D]
    sketch_embeddings = nn.functional.normalize(sketch_embeddings, dim=1)  # [B, D]

    batch_size = photo_embeddings.shape[0]
    # Each row of the cross similarity matrix contains the similarity
    # between a photo and all sketches of the batch.
    cross_similarity = (
        photo_embeddings @ sketch_embeddings.t()
    ) / temperature  # [B, B]
    # On the diagonal, we have the similarity between the same photo and corresponding sketch.
    positive_loss = -torch.diag(cross_similarity)
    # For the negative loss, we will want to compare every other similarity;

    inter_photo_similarity = (
        photo_embeddings @ photo_embeddings.t()
    ) / temperature  # [B, B]
    inter_sketch_similarity = (
        sketch_embeddings @ sketch_embeddings.t()
    ) / temperature  # [B, B]

    # We will combine all these matrices, so that we have a single matrix
    # with all kinds of similarities.
    all_similarities = torch.cat(
        [
            cross_similarity,
            # It is transposed, so that we also have the similarity
            # between an 'anchor' sketch and all other photos in the batch.
            cross_similarity.t(),
            inter_photo_similarity,
            inter_sketch_similarity,
        ],
        dim=1,
    )  # [B, 4B]
    # We want to mask the diagonals, because we do not want to compare
    # the same photo with itself or the same sketch with itself.
    negative_mask = torch.eye(
        batch_size, device=photo_embeddings.device, dtype=torch.bool
    ).repeat(1, 4)
    negative_loss = torch.logsumexp(
        # Each row of this matrix will contain similarities between
        # - the anchor photo and all other sketches
        # - the anchor sketch and all other photos
        # - the anchor photo and all other photos
        # - the anchor sketch and all other sketches
        # all the other similarities are masked out.
        # and the resulting vector will have logsumexp of the row as its value.
        all_similarities.masked_fill(negative_mask, -torch.inf),
        dim=1,
    )  # [B]
    # Edge case: if whole row is filled with zeroes, ... somehow, logsumexp will return -inf.
    return (
        positive_loss + torch.nan_to_num(negative_loss, nan=0.0, posinf=0.0, neginf=0.0)
    ).mean()


def one_way_dcl_loss(
    anchor_embeddings: EmbeddingBatch,
    positive_embeddings: EmbeddingBatch,
    temperature: float = 0.2,
) -> Loss:
    anchor = torch.nn.functional.normalize(anchor_embeddings, dim=1)
    positive = torch.nn.functional.normalize(positive_embeddings, dim=1)
    batch_size = anchor.size(0)

    cross_similarity = torch.mm(anchor, positive.t())
    positive_loss = -torch.diag(cross_similarity) / temperature
    inter_anchor_distance = anchor @ anchor.t()
    neg_similarity = torch.cat((inter_anchor_distance, cross_similarity), dim=1) / temperature
    neg_mask = torch.eye(batch_size, device=anchor.device, dtype=torch.bool).repeat(1, 2)
    negative_loss = torch.logsumexp(
        neg_similarity.masked_fill(neg_mask, -torch.inf),
        dim=1,
    )
    return (
        positive_loss + torch.nan_to_num(negative_loss, nan=0.0, posinf=0.0, neginf=0.0)
    ).mean()

def two_way_dcl_loss(
    photo_embeddings: EmbeddingBatch,
    sketch_embeddings: EmbeddingBatch,
    temperature: float = 0.2,
) -> Loss:
    return (one_way_dcl_loss(photo_embeddings, sketch_embeddings, temperature) +
            one_way_dcl_loss(sketch_embeddings, photo_embeddings, temperature)) / 2

def _mlp(in_dim: int, hidden: int, out_dim: int, hidden_layers: int) -> nn.Sequential:
    """Small helper function to create a multi-layer perceptron.
    It creates an MLP with a specified number of hidden layers,
    each of the same dimension.
    It uses BatchNorm1d and ReLU activation after each layer except the last one.

    Args:
        in_dim: input dimension
        hidden: hidden layer dimension
        out_dim: output dimension
        layers: number of layers in the MLP

    Returns:
        _description_
    """
    dims = [in_dim] + [hidden] * hidden_layers + [out_dim]
    mods: list[nn.Module] = []
    for i in range(len(dims) - 1):
        mods.append(nn.Linear(dims[i], dims[i + 1], bias=False))
        if i < len(dims) - 2:
            mods += [nn.BatchNorm1d(dims[i + 1]), nn.ReLU(inplace=True)]
    return nn.Sequential(*mods)


class SupConLoss(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.1, contrast_mode='all',
                 base_temperature=0.1):
        super(SupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature

    def forward(self, features, labels=None, mask=None):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [batch_size,  D].
            labels: ground truth of shape [batch_size].
            mask: contrastive mask of shape [batch_size, batch_size], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """


        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(DEVICE)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(DEVICE)
        else:
            mask = mask.float().to(DEVICE)

        features_normalized = nn.functional.normalize(features, dim=-1)
        contrast_count = features_normalized.shape[1]
        contrast_feature = torch.cat(torch.unbind(features_normalized, dim=1), dim=0)
        if self.contrast_mode == 'one':
            anchor_feature = features_normalized[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T),
            self.temperature)
        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        mask = mask.repeat(anchor_count, contrast_count)
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(DEVICE),
            0
        )
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        # modified to handle edge cases when there is no positive pair
        # for an anchor point. 
        # Edge case e.g.:- 
        # features of shape: [4,1,...]
        # labels:            [0,1,1,2]
        # loss before mean:  [nan, ..., ..., nan] 
        mask_pos_pairs = mask.sum(1)
        mask_pos_pairs = torch.where(mask_pos_pairs < 1e-6, 1, mask_pos_pairs)
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask_pos_pairs

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss

class SKTR(nn.Module):
    def __init__(
        self,
        encoder: TimmBackbone,
        hidden_layer_size: int = 2048,
        embedding_size: int = 512,
    ):
        super().__init__()

        self.encoder = encoder
        self.encoder.eval()

        for param in self.encoder.parameters():
            param.requires_grad = False
        self.projection_head = _mlp(
            in_dim=encoder.feature_dim,
            hidden=hidden_layer_size,
            out_dim=embedding_size,
            hidden_layers=2,
        )

    def forward(
        self, photo: RGBTorchBatch, sketch: GrayTorchBatch
    ) -> tuple[EmbeddingBatch, EmbeddingBatch]:
        """
        Forward pass through the model.
        Args:
            photo: batch of photos
            sketch: batch of sketches
        Returns:
            Tuple of embeddings for photos and sketches.
        """

        photo_sketch_batch = torch.cat([
            photo,
            sketch.unsqueeze(1).repeat(1, 3, 1, 1),
        ], dim=0)
        photo_sketch_features = self.encoder(photo_sketch_batch)
        photo_sketch_embeddings = self.projection_head(photo_sketch_features)
        photo_embeddings, sketch_embeddings = torch.split(
            photo_sketch_embeddings, [photo.size(0), sketch.size(0)], dim=0
        )
        return photo_embeddings, sketch_embeddings


    def embed_photo(self, photo: RGBTorchBatch) -> EmbeddingBatch:
        """
        Computes embeddings for a batch of photos.

        Args:
            photo: batch of photos

        Returns:
            Embeddings for the photos.
        """
        photo_features = self.encoder(photo)
        photo_embeddings = self.projection_head(photo_features)
        return photo_embeddings
    
    def embed_sketch(self, sketch: GrayTorchBatch) -> EmbeddingBatch:
        """
        Computes embeddings for a batch of sketches.

        Args:
            sketch: batch of sketches

        Returns:
            Embeddings for the sketches.
        """
        sketch_features = self.encoder(sketch)
        sketch_embeddings = self.projection_head(sketch_features)
        return sketch_embeddings