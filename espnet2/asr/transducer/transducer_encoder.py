from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

import ast
import numpy as np
import torch
from typeguard import check_argument_types

from espnet.nets.pytorch_backend.transducer.rnn_encoder import RNN
from espnet.nets.pytorch_backend.transducer.rnn_encoder import RNNP
from espnet.nets.pytorch_backend.transducer.rnn_encoder import VGG2L

from espnet.nets.e2e_asr_common import get_vgg2l_odim
from espnet.nets.pytorch_backend.nets_utils import make_pad_mask
from espnet.nets.pytorch_backend.nets_utils import to_device
from espnet.nets.pytorch_backend.transducer.utils import valid_aux_encoder_output_layers

from espnet2.asr.encoder.abs_encoder import AbsEncoder

class TransducerEncoder(AbsEncoder):
    """TransducerEncoder class.

    Args:
        input_size: The number of expected features in the input
        bidirectional: If ``True`` becomes a bidirectional LSTM
        use_projection: Use projection layer or not
        num_layers: Number of recurrent layers
        hidden_size: The number of hidden features
        output_size: The number of output features
        dropout: dropout probability

    """

    def __init__(
        self,
        input_size: int,
        auxiliary_conf: Optional[Dict],
        rnn_type: str = "lstm",
        bidirectional: bool = True,
        num_layers: int = 4,
        hidden_size: int = 320,
        output_size: int = 320,
        dropout: float = 0.0,
        in_channel: int = 1,
        use_projection: bool = False,
    ):
        assert check_argument_types()
        super().__init__()
        self._output_size = output_size
        self.rnn_type = rnn_type
        self.bidirectional = bidirectional
        # self.use_projection = use_projection
        if rnn_type not in {"lstm", "gru"}:
            raise ValueError(f"Not supported rnn_type={rnn_type}")

        # Subsample is not used for VGGRNN
        subsample = np.ones(num_layers + 1, dtype=np.int)
        rnn_type = ("b" if bidirectional else "") + rnn_type

        # print(f"auxiliary_conf: {auxiliary_conf}")
        # FIXME: Not implemented yet.
        if auxiliary_conf:
            aux_transducer_loss_enc_output_layers = ast.literal_eval(auxiliary_conf['aux_transducer_loss_enc_output_layers'])
            use_symm_kl_div_loss = auxiliary_conf['use_symm_kl_div_loss']

            aux_enc_output_layers = valid_aux_encoder_output_layers(
                aux_transducer_loss_enc_output_layers,
                num_layers-1,
                use_symm_kl_div_loss,
                subsample,
            )
        else:
            aux_enc_output_layers = []
        
        if use_projection:
            self.enc = torch.nn.ModuleList(
                [
                    VGG2L(in_channel),
                    RNNP(
                        get_vgg2l_odim(input_size, in_channel=in_channel),
                        rnn_type,
                        num_layers,
                        hidden_size,
                        output_size,
                        subsample,
                        dropout_rate=dropout,
                        aux_output_layers=aux_enc_output_layers,
                    ),
                ]
            )
        else:
            self.enc = torch.nn.ModuleList(
                [
                    VGG2L(in_channel),
                    RNN(
                        get_vgg2l_odim(input_size, in_channel=in_channel),
                        rnn_type,
                        num_layers,
                        hidden_size,
                        output_size,
                        dropout_rate=dropout,
                        aux_output_layers=aux_enc_output_layers,
                    ),
                ]
            )
        
    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        feats: torch.Tensor,
        feats_len: torch.Tensor,
        prev_states: Optional[List[torch.Tensor]] = None,
    ):
        """Forward encoder.

        Args:
            feats: Feature sequences. (B, F, D_feats)
            feats_len: Feature sequences lengths. (B,)
            prev_states: Previous encoder hidden states. [N x (B, T, D_enc)]

        Returns:
            enc_out: Encoder output sequences. (B, T, D_enc)
                   with or without encoder intermediate output sequences.
                   ((B, T, D_enc), [N x (B, T, D_enc)])
            enc_out_len: Encoder output sequences lengths. (B,)
            current_states: Encoder hidden states. [N x (B, T, D_enc)]

        """
        if prev_states is None:
            prev_states = [None] * len(self.enc)
        assert len(prev_states) == len(self.enc)

        _enc_out = feats
        _enc_out_len = feats_len
        current_states = []
        for rnn_module, prev_state in zip(self.enc, prev_states):
            _enc_out, _enc_out_len, states = rnn_module(
                _enc_out,
                _enc_out_len,
                prev_states=prev_state,
            )
            current_states.append(states)

        if isinstance(_enc_out, tuple):
            enc_out, aux_enc_out = _enc_out[0], _enc_out[1]
            enc_out_len, aux_enc_out_len = _enc_out_len[0], _enc_out_len[1]

            enc_out_mask = to_device(enc_out, make_pad_mask(enc_out_len).unsqueeze(-1))
            enc_out = enc_out.masked_fill(enc_out_mask, 0.0)

            for i in range(len(aux_enc_out)):
                aux_mask = to_device(
                    aux_enc_out[i], make_pad_mask(aux_enc_out_len[i]).unsqueeze(-1)
                )
                aux_enc_out[i] = aux_enc_out[i].masked_fill(aux_mask, 0.0)

            return (
                (enc_out, aux_enc_out),
                (enc_out_len, aux_enc_out_len),
                current_states,
            )
        else:
            enc_out_mask = to_device(
                _enc_out, make_pad_mask(_enc_out_len).unsqueeze(-1)
            )

            return _enc_out.masked_fill(enc_out_mask, 0.0), _enc_out_len, current_states