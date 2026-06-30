

from torch.nn import ModuleList
import copy
from typing import Optional, Any, Union, Callable
import torch
from torch import Tensor
from torch.nn import functional as F
import torch.nn as nn
from torch.nn.modules import Module
from torch.nn import MultiheadAttention
from torch.nn import LayerNorm
from torch.nn import Dropout
from torch.nn import Linear
from einops.layers.torch import Rearrange


def _get_activation_fn(activation: str) -> Callable[[Tensor], Tensor]:
    if activation == "relu":
        return F.relu
    elif activation == "gelu":
        return F.gelu

    raise RuntimeError(f"activation should be relu/gelu, not {activation}")


class FiLMLayer(nn.Module):
    """
    Applies Feature-wise Linear Modulation (FiLM) to an input tensor.
    The modulation is an affine transformation: output = x * gamma + beta.

    Args:
        feature_dim (int): The dimension of the features to be modulated.
    """

    def __init__(self, cond_dim, out_dim: int, enable=False):
        super(FiLMLayer, self).__init__()
        self.feature_dim = out_dim
        self.cond_encoder = nn.Sequential(
            nn.Mish(),
            nn.Linear(cond_dim, out_dim * 2),  # Predict both scale and bias
        )
        self.enable = enable

    def forward(self, cond: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): The input tensor to be modulated.
            gamma (torch.Tensor): The scaling parameter, expected to be broadcastable to x.
            beta (torch.Tensor): The shifting parameter, expected to be broadcastable to x.

        Returns:
            torch.Tensor: The modulated tensor.
        """
        # We apply the modulation directly. In the FiLM paper, they often use
        # gamma as a scaling factor, but the term is sometimes simplified.
        # Here we use x * gamma + beta, which is the standard affine transform.
        if self.enable:
            embed = self.cond_encoder(cond)
            embed = embed.reshape(embed.shape[0], 2, 1, self.feature_dim)
            gamma = embed[:, 0, ...]
            beta = embed[:, 1, ...]
            return x * gamma + beta
        else:
            return x


class FilMSelfAttnDecoderLayer(Module):
    r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
    This standard decoder layer is based on the paper "Attention Is All You Need".
    Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
    Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
    Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
    in a different way during application.

    Args:
        d_model: the number of expected features in the input (required).
        nhead: the number of heads in the multiheadattention models (required).
        dim_feedforward: the dimension of the feedforward network model (default=2048).
        dropout: the dropout value (default=0.1).
        activation: the activation function of the intermediate layer, can be a string
            ("relu" or "gelu") or a unary callable. Default: relu
        layer_norm_eps: the eps value in layer normalization components (default=1e-5).
        batch_first: If ``True``, then the input and output tensors are provided
            as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
        norm_first: if ``True``, layer norm is done prior to self attention, multihead
            attention and feedforward operations, respectively. Otherwise it's done after.
            Default: ``False`` (after).
        bias: If set to ``False``, ``Linear`` and ``LayerNorm`` layers will not learn an additive
            bias. Default: ``True``.

    Examples::
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
        >>> memory = torch.rand(10, 32, 512)
        >>> tgt = torch.rand(20, 32, 512)
        >>> out = decoder_layer(tgt, memory)

    Alternatively, when ``batch_first`` is ``True``:
        >>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
        >>> memory = torch.rand(32, 10, 512)
        >>> tgt = torch.rand(32, 20, 512)
        >>> out = decoder_layer(tgt, memory)
    """
    __constants__ = ['norm_first']

    def __init__(self, d_model: int, nhead: int,
                 cond_dim: int, enable_film_self_attn: bool = True, enable_film_ffn: bool = True,  # FILM Plugin
                 dim_feedforward: int = 2048, dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
                 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
                 bias: bool = True, device=None, dtype=None, ) -> None:
        factory_kwargs = {'device': device, 'dtype': dtype}
        super().__init__()
        self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                            bias=bias, **factory_kwargs)
        self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
                                                 bias=bias, **factory_kwargs)
        # Implementation of Feedforward model
        self.linear1 = Linear(d_model, dim_feedforward, bias=bias, **factory_kwargs)
        self.dropout = Dropout(dropout)
        self.linear2 = Linear(dim_feedforward, d_model, bias=bias, **factory_kwargs)

        self.norm_first = norm_first
        self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, bias=bias, **factory_kwargs)
        self.dropout1 = Dropout(dropout)
        self.dropout3 = Dropout(dropout)

        # Legacy string support for activation function.
        if isinstance(activation, str):
            self.activation = _get_activation_fn(activation)
        else:
            self.activation = activation

        # FiLM Plugin
        self.film_self_attn = FiLMLayer(cond_dim, d_model, enable_film_self_attn)
        self.film_ffn = FiLMLayer(cond_dim, d_model, enable_film_ffn)
        # /FiLM Plugin

    def __setstate__(self, state):
        if 'activation' not in state:
            state['activation'] = F.relu
        super().__setstate__(state)

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        cond: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder layer.

        Args:
            tgt: the sequence to the decoder layer (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
        Shape:
            see the docs in Transformer class.
        """
        # see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

        x = tgt
        if self.norm_first:
            x = x + self.film_self_attn(cond, self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask, tgt_is_causal))
            x = x + self.film_ffn(cond, self._ff_block(self.norm3(x)))
        else:
            x = self.norm1(x + self.film_self_attn(cond, self._sa_block(x, tgt_mask, tgt_key_padding_mask, tgt_is_causal)))
            x = self.norm3(x + self.film_ffn(cond, self._ff_block(x)))

        return x
    # self-attention block

    def _sa_block(self, x: Tensor,
                  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], is_causal: bool = False) -> Tensor:
        x = self.self_attn(x, x, x,
                           attn_mask=attn_mask,
                           key_padding_mask=key_padding_mask,
                           is_causal=is_causal,
                           need_weights=False)[0]
        return self.dropout1(x)

    # feed forward block

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout3(x)


# --- New FilMTransformerDecoder class ---


class FilMTransformerDecoder(Module):
    """
    FilMTransformerDecoder is a stack of N FilMSelfAttnDecoderLayer.

    Args:
        decoder_layer: an instance of the FilMSelfAttnDecoderLayer (required).
        num_layers: the number of sub-decoder-layers in the decoder (required).
        norm: the layer normalization component (optional).

    Examples:
        >>> decoder_layer = FilMSelfAttnDecoderLayer(d_model=512, nhead=8, cond_dim=128)
        >>> transformer_decoder = FilMTransformerDecoder(decoder_layer, num_layers=6)
        >>> tgt = torch.rand(10, 32, 512)
        >>> cond = torch.rand(32, 128)
        >>> out = transformer_decoder(tgt, cond)
    """

    def __init__(self, decoder_layer: FilMSelfAttnDecoderLayer, num_layers: int, norm=None):
        super().__init__()
        # Ensure the decoder layer is a valid instance.
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        tgt: Tensor,
        tgt_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        tgt_is_causal: bool = False,
        cond: Optional[Tensor] = None
    ) -> Tensor:
        r"""Pass the inputs (and mask) through the decoder.
        Args:
            tgt: the sequence to the decoder (required).
            tgt_mask: the mask for the tgt sequence (optional).
            tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
            tgt_is_causal: If specified, applies a causal mask as attention mask.
            cond: The condition tensor for the FiLM layers in each layer (optional).
        """
        output = tgt

        for mod in self.layers:
            output = mod(output, tgt_mask=tgt_mask, tgt_key_padding_mask=tgt_key_padding_mask,
                         tgt_is_causal=tgt_is_causal, cond=cond)

        if self.norm is not None:
            output = self.norm(output)

        return output


def _get_clones(module, N):
    # FIXME: copy.deepcopy() is not defined on nn.module
    return ModuleList([copy.deepcopy(module) for i in range(N)])
