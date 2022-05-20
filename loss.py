"""Loss functions"""

import random
from typing import Optional

import torch
from einops import repeat  # noqa
from torch import nn, Tensor

from rigids import Rigids
from utils import exists, get_max_val, safe_norm, masked_mean


class FAPELoss(nn.Module):
    """FAPE Loss"""

    def __init__(
            self,
            d_clamp: float = 10,
            eps: float = 1e-8,
            scale: float = 10,
            clamp_prob: float = 0.9,
    ):
        """Clamped FAPE loss

        :param d_clamp: maximum distance value allowed for loss - all values larger will be clamped to d_clamp
        :param eps: tolerance factor for computing norms (so gradient is defined in sqrt)
        :param scale: (Inverse) Amount to scale loss by (usually equal tho d_clamp)
        :param clamp_prob: Probability with which loss values are clamped [0,1]. If this
        value is not 1, then a (1-clamp_prob) fraction of samples will not have clamping applied to
        distance deviations
        """
        super(FAPELoss, self).__init__()
        self._d_clamp, self.eps, self.scale = d_clamp, eps, scale
        self.clamp_prob = clamp_prob
        self.clamp = lambda diffs: torch.clamp_max(
            diffs, self._d_clamp if random.uniform(0, 1) < self.clamp_prob
            else get_max_val(diffs)
        )

    def forward(
            self,
            pred_coords: Tensor,
            true_coords: Tensor,
            pred_rigids: Optional[Rigids] = None,
            true_rigids: Optional[Rigids] = None,
            coord_mask: Optional[Tensor] = None,
            reduce: bool = True,
            clamp_val: Optional[float] = None,
    ) -> Tensor:
        """Compute FAPE Loss

        :param pred_coords: tensor of shape (b,n,a,3)
        :param true_coords: tensor of shape (b,n,a,3)
        :param pred_rigids: (Optional) predicted rigid transformation
        :param true_rigids: (Optional) rigid transformation computed on native structure.
        If missing, the transformation is computed assuming true_points[:,:,:3]
        are N,CA, and C coordinates
        :param coord_mask: (Optional) tensor of shape (b,n,a) indicating which atom coordinates
        to compute loss for
        :param reduce: whether to output mean of loss (True), or return loss for each input coordinate
        :param clamp_val: value to use for clamping (optional)

        :return: FAPE loss between predicted and true coordinates
        """
        b, n, a = true_coords.shape[:3]
        assert pred_coords.ndim == true_coords.ndim == 4

        true_rigids = true_rigids if exists(true_rigids) else Rigids.RigidFromBackbone(true_coords)
        pred_rigids = pred_rigids if exists(pred_rigids) else Rigids.RigidFromBackbone(pred_coords)

        pred_coords, true_coords = map(lambda x: repeat(x, "b n a c -> b () (n a) c"),
                                       (pred_coords, true_coords))

        # rotate and translate coordinates into local frame
        true_coords_local = true_rigids.apply_inverse(true_coords)
        pred_coords_local = pred_rigids.apply_inverse(pred_coords)

        # compute deviations of predicted and actual coords.
        unclamped_diffs = safe_norm(pred_coords_local - true_coords_local, dim=-1, eps=self.eps)
        diffs = torch.clamp_max(unclamped_diffs, clamp_val) if exists(clamp_val) \
            else self.clamp(unclamped_diffs)

        residue_mask = torch.any(coord_mask, dim=-1, keepdim=True) if exists(coord_mask) else None
        coord_mask = repeat(coord_mask, "b n a -> b m (n a)", m=n) if exists(coord_mask) else None
        coord_mask = coord_mask.masked_fill(~residue_mask, False) if exists(coord_mask) else None

        if reduce:
            fape = torch.mean(masked_mean(diffs, coord_mask, dim=(-1, -2)))
            return (1 / self.scale) * fape

        return (1 / self.scale) * (diffs.masked_fill(~coord_mask, 0) if exists(coord_mask) else diffs)
