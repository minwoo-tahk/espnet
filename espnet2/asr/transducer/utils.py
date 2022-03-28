"""Utility functions for Transducer models."""

from typing import List
from typing import Optional
from typing import Tuple

import torch

from espnet.nets.pytorch_backend.nets_utils import pad_list

def get_decoder_input(
    labels: torch.Tensor,
    blank_id: int,
    ignore_id: int
) -> torch.Tensor:
    """Prepare decoder input.

    Args:
        labels: Label ID sequences. (B, L)

    Returns:
        decoder_input: Label ID sequences with blank prefix. (B, U)

    """
    device = labels.device

    labels_unpad = [label[label != ignore_id] for label in labels]
    blank = labels[0].new([blank_id])

    decoder_input = pad_list(
        [torch.cat([blank, label], dim=0) for label in labels_unpad], blank_id
    ).to(device)

    return decoder_input


def get_transducer_tasks_io(
    labels: torch.Tensor,
    enc_out_len: torch.Tensor,
    aux_enc_out_len: Optional[List],
    ignore_id: int = -1,
    blank_id: int = 0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get Transducer tasks inputs and outputs.

    Args:
        labels: Label ID sequences. (B, U)
        enc_out_len: Time lengths. (B,)
        aux_enc_out_len: Auxiliary time lengths. [N X (B,)]

    Returns:
        target: Target label ID sequences. (B, L)
        lm_loss_target: LM loss target label ID sequences. (B, U)
        t_len: Time lengths. (B,)
        aux_t_len: Auxiliary time lengths. [N x (B,)]
        u_len: Label lengths. (B,)

    """
    device = labels.device

    labels_unpad = [label[label != ignore_id] for label in labels]
    blank = labels[0].new([blank_id])

    target = pad_list(labels_unpad, blank_id).type(torch.int32).to(device)
    lm_loss_target = (
        pad_list(
            [torch.cat([y, blank], dim=0) for y in labels_unpad], ignore_id
        )
        .type(torch.int64)
        .to(device)
    )

    if enc_out_len.dim() > 1:
        enc_mask_unpad = [m[m != 0] for m in enc_out_len]
        enc_out_len = list(map(int, [m.size(0) for m in enc_mask_unpad]))
    else:
        enc_out_len = list(map(int, enc_out_len))

    t_len = torch.IntTensor(enc_out_len).to(device)
    u_len = torch.IntTensor([label.size(0) for label in labels_unpad]).to(device)

    if aux_enc_out_len:
        aux_t_len = []

        for i in range(len(aux_enc_out_len)):
            if aux_enc_out_len[i].dim() > 1:
                aux_mask_unpad = [aux[aux != 0] for aux in aux_enc_out_len[i]]
                aux_t_len.append(
                    torch.IntTensor(
                        list(map(int, [aux.size(0) for aux in aux_mask_unpad]))
                    ).to(device)
                )
            else:
                aux_t_len.append(
                    torch.IntTensor(list(map(int, aux_enc_out_len[i]))).to(device)
                )
    else:
        aux_t_len = aux_enc_out_len

    return target, lm_loss_target, t_len, aux_t_len, u_len
