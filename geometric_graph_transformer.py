"""Geometric Graph Transformer"""
from typing import Optional, Tuple, Dict, Any

import torch.nn.functional as F  # noqa
from einops import rearrange, repeat  # noqa
from torch import nn, Tensor

from loss import FAPELoss
from node_updates import NodeUpdateBlock
from pair_updates import PairUpdateBlock
from rigids import Rigids
from utils import (
    exists,
)

RIGID_SCALE = 10


class GraphTransformer(nn.Module):
    """Graph Transformer"""

    def __init__(
            self,
            node_dim: int,
            pair_dim: Optional[int],
            depth: int,
            use_ipa: bool,
            node_update_kwargs: Dict[str, Any],
            pair_update_kwargs: Optional[Dict[str, Any]] = None,
            share_weights: bool = False,
    ):

        super(GraphTransformer, self).__init__()
        self.depth, self.use_ipa = depth, use_ipa

        # shared rigid update accross all layers
        self.to_rigid_update = nn.Sequential(
            nn.LayerNorm(node_dim),
            nn.Linear(node_dim, 6)
        ) if use_ipa else None

        self.share_weights = share_weights
        if share_weights and exists(use_ipa):
            self.fape_aux = FAPELoss(clamp_prob=0.9, scale=10)

        depth = 1 if share_weights else depth
        # Node Updates
        self.node_updates = nn.ModuleList([NodeUpdateBlock(
            node_dim=node_dim,
            pair_dim=pair_dim,
            use_ipa=use_ipa,
            **node_update_kwargs,
        ) for _ in range(depth)])

        # Pair Updates
        self.pair_updates = nn.ModuleList([PairUpdateBlock(
            node_dim=node_dim,
            pair_dim=pair_dim,
            **pair_update_kwargs,
        ) for _ in range(depth)]) if exists(pair_update_kwargs) else [None] * depth

    def update_rigids(self, feats: Tensor, rigids: Rigids) -> Optional[Rigids]:
        """Update rigid transformations"""
        if exists(self.to_rigid_update):
            quaternion_update, translation_update = self.to_rigid_update(feats).chunk(2, dim=-1)
            quaternion_update = F.pad(quaternion_update, (1, 0), value=1.)
            return rigids.compose(Rigids(quaternion_update, translation_update))
        return None

    def forward(
            self,
            node_feats: Tensor,
            pair_feats: Optional[Tensor],
            rigids: Optional[Rigids] = None,
            true_rigids: Optional[Rigids] = None,
            res_mask: Optional[Tensor] = None,
            pair_mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor, Optional[Rigids], Optional[Tensor]]:
        """
        :param node_feats: node features (b,n,d_node)
        :param pair_feats: pair features (b,n,n,d_pair)
        :param rigids: rigids to use for IPA (Identity is used o.w.)
        :param true_rigids: native protein rigids for aux. loss
        (applicable only if weights are shared)
        :param res_mask: residue mask (b,n)
        :param pair_mask: pair mask (b,n,n)

        :return:
            (1) scalar features
            (2) pair features
            (3) rigids (optional)
            (4) auxilliary loss (optional) - computed only if weights are shared
            and true rigids are supplied
        """
        assert exists(true_rigids) or not self.share_weights
        node_feats, pair_feats, device = node_feats, pair_feats, node_feats.device
        b, n, *_ = node_feats.shape
        assert node_feats.ndim == 3 and pair_feats.ndim == 4, f"scalar and pair feats must have batch dimension!"

        if self.use_ipa:
            # if no initial rigids passed in, start from identity
            rigids = rigids if exists(rigids) else \
                Rigids.IdentityRigid(leading_shape=(b, n), device=device)  # noqa
            rigids = rigids.scale(1 / RIGID_SCALE)

        node_updates, pair_updates, aux_loss = self.node_updates, self.pair_updates, None
        if self.share_weights:
            aux_loss = 0
            node_updates = [node_updates[0]] * self.depth
            pair_updates = [pair_updates[0]] * self.depth

        for node_update, pair_update in zip(node_updates, pair_updates):
            forward_kwargs = dict(node_feats=node_feats, pair_feats=pair_feats, mask=pair_mask)
            if self.use_ipa:
                rigids = rigids.detach_rot()
                forward_kwargs["rigids"] = rigids

            node_feats = node_update(**forward_kwargs)

            if exists(pair_update):
                pair_feats = pair_update(
                    node_feats=node_feats,
                    pair_feats=pair_feats,
                    mask=pair_mask
                )

            rigids = self.update_rigids(node_feats, rigids)
            if self.share_weights and self.use_ipa:
                aux_loss = self.aux_loss(
                    res_mask=res_mask,
                    pred_rigids=rigids,
                    true_rigids=true_rigids,
                ) / self.depth + aux_loss

        rigids = rigids.scale(RIGID_SCALE) if exists(rigids) else Rigids
        return node_feats, pair_feats, rigids, aux_loss

    def aux_loss(
            self,
            res_mask: Optional[Tensor],
            pred_rigids: Rigids,
            true_rigids: Rigids,
    ):
        """Auxiliary loss when IPA layers have shared weights"""
        scaled_pred_rigids = pred_rigids.scale(RIGID_SCALE)
        pred_ca = scaled_pred_rigids.translations
        native_ca = true_rigids.translations
        assert pred_ca.shape == native_ca.shape, \
            f"{pred_ca.shape, native_ca.shape}"
        pred_ca, native_ca = map(
            lambda x: rearrange(x, "b n c -> b n () c"),
            (pred_ca, native_ca)
        )
        assert pred_ca.ndim == 4, f"{pred_ca.shape}"
        if exists(res_mask):
            assert res_mask.ndim == 2, f"{res_mask.shape}"
            res_mask = res_mask.unsqueeze(-1)
        return self.fape_aux.forward(
            pred_coords=pred_ca,
            true_coords=native_ca.detach(),
            pred_rigids=scaled_pred_rigids,
            true_rigids=true_rigids.detach_all(),
            coord_mask=res_mask,
            reduce=True,
        )
