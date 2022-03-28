from contextlib import contextmanager
from distutils.version import LooseVersion
import logging
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple
from typing import Union

import torch
from typeguard import check_argument_types

from espnet.nets.e2e_asr_common import ErrorCalculator
from espnet.nets.pytorch_backend.nets_utils import th_accuracy
from espnet.nets.pytorch_backend.transformer.add_sos_eos import add_sos_eos
from espnet.nets.pytorch_backend.transformer.label_smoothing_loss import (
    LabelSmoothingLoss,  # noqa: H301
)
from espnet2.asr.ctc import CTC
from espnet2.asr.decoder.abs_decoder import AbsDecoder
from espnet2.asr.encoder.abs_encoder import AbsEncoder
from espnet2.asr.frontend.abs_frontend import AbsFrontend
from espnet2.asr.postencoder.abs_postencoder import AbsPostEncoder
from espnet2.asr.preencoder.abs_preencoder import AbsPreEncoder
from espnet2.asr.specaug.abs_specaug import AbsSpecAug
from espnet2.asr.transducer.error_calculator import ErrorCalculatorTransducer
from espnet2.asr.transducer.utils import (
    get_decoder_input,
    get_transducer_tasks_io,
)
from espnet2.layers.abs_normalize import AbsNormalize
from espnet2.torch_utils.device_funcs import force_gatherable
from espnet2.train.abs_espnet_model import AbsESPnetModel

if LooseVersion(torch.__version__) >= LooseVersion("1.6.0"):
    from torch.cuda.amp import autocast
else:
    # Nothing to do if torch<1.6.0
    @contextmanager
    def autocast(enabled=True):
        yield


class ESPnetASRModel(AbsESPnetModel):
    """CTC-attention hybrid Encoder-Decoder model"""

    def __init__(
        self,
        vocab_size: int,
        token_list: Union[Tuple[str, ...], List[str]],
        frontend: Optional[AbsFrontend],
        specaug: Optional[AbsSpecAug],
        normalize: Optional[AbsNormalize],
        preencoder: Optional[AbsPreEncoder],
        encoder: AbsEncoder,
        postencoder: Optional[AbsPostEncoder],
        decoder: AbsDecoder,
        ctc: CTC,
        joint_network: Optional[torch.nn.Module],
        auxiliary_conf: Optional[Dict],
        ctc_weight: float = 0.5,
        ignore_id: int = -1,
        lsm_weight: float = 0.0,
        length_normalized_loss: bool = False,
        report_cer: bool = True,
        report_wer: bool = True,
        sym_space: str = "<space>",
        sym_blank: str = "<blank>",
        extract_feats_in_collect_stats: bool = True,
    ):
        assert check_argument_types()
        assert 0.0 <= ctc_weight <= 1.0, ctc_weight

        super().__init__()
        # note that eos is the same as sos (equivalent ID)
        self.blank_id = 0
        self.sos = vocab_size - 1
        self.eos = vocab_size - 1
        self.vocab_size = vocab_size
        self.ignore_id = ignore_id
        self.ctc_weight = ctc_weight
        self.token_list = token_list.copy()

        self.frontend = frontend
        self.specaug = specaug
        self.normalize = normalize
        self.preencoder = preencoder
        self.postencoder = postencoder
        self.encoder = encoder

        self.use_transducer_decoder = joint_network is not None
        # Added by Minwoo
        self.use_auxiliary_loss = self.use_transducer_decoder and auxiliary_conf is not None
        self.use_symm_kl_div_loss = self.use_auxiliary_loss and ('use_symm_kl_div_loss' in auxiliary_conf and auxiliary_conf['use_symm_kl_div_loss'] is not False)
        self.use_lm_loss = self.use_auxiliary_loss and ('use_lm_loss' in auxiliary_conf and auxiliary_conf['use_lm_loss'] is not False)
        
        self.error_calculator = None

        if self.use_transducer_decoder:
            # from warprnnt_pytorch import RNNTLoss
            from torchaudio.transforms import RNNTLoss

            self.decoder = decoder
            self.joint_network = joint_network

            self.transducer_loss_weight = 1.0
            self.aux_transducer_loss_weight = 1.0
            self.symm_kl_div_loss_weight = 1.0
            self.lm_loss_weight = 1.0

            # self.criterion_transducer = RNNTLoss(
            #     blank=self.blank_id,
            #     fastemit_lambda=0.0,
            # )
            self.criterion_transducer = RNNTLoss(
                blank=self.blank_id,
            )

            if ctc_weight == 0.0:
                self.ctc = None
            else:
                self.ctc = ctc

            # Added by Minwoo
            if self.use_auxiliary_loss:
                last_layer = [layer for name, layer in self.encoder.enc[-1].named_modules() if name != 'dropout'][-1]
                encoder_output_size = last_layer.state_dict()['weight'].shape[0]
                aux_transducer_loss_mlp_dim = auxiliary_conf['aux_transducer_loss_mlp_dim']
                aux_trans_loss_mlp_dropout_rate = auxiliary_conf['aux_transducer_loss_mlp_dropout_rate']
                joint_space_size = self.joint_network.lin_enc.state_dict()['weight'].shape[0]

                self.mlp = torch.nn.Sequential(
                    torch.nn.Linear(encoder_output_size, aux_transducer_loss_mlp_dim),
                    torch.nn.LayerNorm(aux_transducer_loss_mlp_dim),
                    torch.nn.Dropout(p=aux_trans_loss_mlp_dropout_rate),
                    torch.nn.ReLU(),
                    torch.nn.Linear(aux_transducer_loss_mlp_dim, joint_space_size),
                )

                if self.use_symm_kl_div_loss:
                    self.kl_div = torch.nn.KLDivLoss(reduction="sum")

                self.aux_transducer_loss_weight = auxiliary_conf['aux_transducer_loss_weight']
                self.symm_kl_div_loss_weight = auxiliary_conf['symm_kl_div_loss_weight']

            if self.use_lm_loss:
                decoder_dim = self.decoder.embed.state_dict()['weight'].shape[1]
                lm_loss_smoothing_rate = auxiliary_conf['lm_loss_smoothing_rate']

                self.lm_lin = torch.nn.Linear(decoder_dim, vocab_size)
                self.criterion_att = LabelSmoothingLoss(
                    size=self.vocab_size,
                    padding_idx=self.ignore_id,
                    smoothing=lm_loss_smoothing_rate,
                    normalize_length=False,
                )
                if report_cer or report_wer:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
                self.lm_loss_weight = auxiliary_conf['lm_loss_weight']

            if report_cer or report_wer:
                self.error_calculator_trans = ErrorCalculatorTransducer(
                    decoder,
                    joint_network,
                    token_list,
                    sym_space,
                    sym_blank,
                    report_cer=report_cer,
                    report_wer=report_wer,
                )
            else:
                self.error_calculator_trans = None

                if self.ctc_weight != 0:
                    self.error_calculator = ErrorCalculator(
                        token_list, sym_space, sym_blank, report_cer, report_wer
                    )
        else:
            # we set self.decoder = None in the CTC mode since
            # self.decoder parameters were never used and PyTorch complained
            # and threw an Exception in the multi-GPU experiment.
            # thanks Jeff Farris for pointing out the issue.
            if ctc_weight == 1.0:
                self.decoder = None
            else:
                self.decoder = decoder

            self.criterion_att = LabelSmoothingLoss(
                size=vocab_size,
                padding_idx=ignore_id,
                smoothing=lsm_weight,
                normalize_length=length_normalized_loss,
            )

            if ctc_weight == 0.0:
                self.ctc = None
            else:
                self.ctc = ctc

            if report_cer or report_wer:
                self.error_calculator = ErrorCalculator(
                    token_list, sym_space, sym_blank, report_cer, report_wer
                )

        self.extract_feats_in_collect_stats = extract_feats_in_collect_stats

    def forward(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor], torch.Tensor]:
        """Frontend + Encoder + Decoder + Calc loss

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
            text: (Batch, Length)
            text_lengths: (Batch,)
        """
        assert text_lengths.dim() == 1, text_lengths.shape
        # Check that batch_size is unified
        assert (
            speech.shape[0]
            == speech_lengths.shape[0]
            == text.shape[0]
            == text_lengths.shape[0]
        ), (speech.shape, speech_lengths.shape, text.shape, text_lengths.shape)
        batch_size = speech.shape[0]

        # for data-parallel
        text = text[:, : text_lengths.max()]

        # 1. Encoder
        encoder_out, encoder_out_lens = self.encode(speech, speech_lengths)

        loss_att, acc_att, cer_att, wer_att = None, None, None, None
        loss_ctc, cer_ctc = None, None
        loss_transducer, cer_transducer, wer_transducer = None, None, None
        loss_aux_trans, loss_kl_div = None, None

        if self.use_transducer_decoder:
            aux_encoder_out, aux_encoder_out_lens = None, None

            if self.use_auxiliary_loss and isinstance(encoder_out, tuple):
                encoder_out, aux_encoder_out = encoder_out[0], encoder_out[1]
                encoder_out_lens, aux_encoder_out_lens = encoder_out_lens[0], encoder_out_lens[1]

            # 1. CTC branch
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 2a. Transducer decoder branch
            (
                loss_att,
                loss_transducer,
                loss_aux_trans,
                loss_kl_div,
                acc_att,
                cer_att,
                wer_att,
                cer_transducer,
                wer_transducer,
            ) = self._calc_transducer_tasks(
                encoder_out,
                aux_encoder_out,
                encoder_out_lens,
                aux_encoder_out_lens,
                text,
            )

            loss = self.transducer_loss_weight * loss_transducer

            if loss_ctc is not None:
                loss += (self.ctc_weight * loss_ctc)
            if loss_aux_trans is not None:
                loss += (self.aux_transducer_loss_weight * loss_aux_trans)
            if loss_kl_div is not None:
                loss += (self.symm_kl_div_loss_weight * loss_kl_div)
            if loss_att is not None:
                loss += (self.lm_loss_weight * loss_att)
        else:
            # 1. CTC branch
            if self.ctc_weight != 0.0:
                loss_ctc, cer_ctc = self._calc_ctc_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )
            
            # 2b. Attention decoder branch
            if self.ctc_weight != 1.0:
                loss_att, acc_att, cer_att, wer_att = self._calc_att_loss(
                    encoder_out, encoder_out_lens, text, text_lengths
                )

            # 3. CTC-Att loss definition
            if self.ctc_weight == 0.0:
                loss = loss_att
            elif self.ctc_weight == 1.0:
                loss = loss_ctc
            else:
                loss = self.ctc_weight * loss_ctc + (1 - self.ctc_weight) * loss_att

        stats = dict(
            loss=loss.detach(),
            loss_att=loss_att.detach() if loss_att is not None else None,
            loss_ctc=loss_ctc.detach() if loss_ctc is not None else None,
            loss_transducer=loss_transducer.detach()
            if loss_transducer is not None
            else None,
            loss_aux_trans=loss_aux_trans.detach()
            if loss_aux_trans is not None
            else None,
            loss_kl_div=loss_kl_div.detach()
            if loss_kl_div is not None
            else None,
            acc=acc_att,
            cer=cer_att,
            wer=wer_att,
            cer_ctc=cer_ctc,
            cer_transducer=cer_transducer,
            wer_transducer=wer_transducer,
        )

        # force_gatherable: to-device and to-tensor if scalar for DataParallel
        loss, stats, weight = force_gatherable((loss, stats, batch_size), loss.device)
        return loss, stats, weight

    def collect_feats(
        self,
        speech: torch.Tensor,
        speech_lengths: torch.Tensor,
        text: torch.Tensor,
        text_lengths: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if self.extract_feats_in_collect_stats:
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)
        else:
            # Generate dummy stats if extract_feats_in_collect_stats is False
            logging.warning(
                "Generating dummy stats for feats and feats_lengths, "
                "because encoder_conf.extract_feats_in_collect_stats is "
                f"{self.extract_feats_in_collect_stats}"
            )
            feats, feats_lengths = speech, speech_lengths
        return {"feats": feats, "feats_lengths": feats_lengths}

    def encode(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Frontend + Encoder. Note that this method is used by asr_inference.py

        Args:
            speech: (Batch, Length, ...)
            speech_lengths: (Batch, )
        """
        with autocast(False):
            # 1. Extract feats
            feats, feats_lengths = self._extract_feats(speech, speech_lengths)

            # 2. Data augmentation
            if self.specaug is not None and self.training:
                feats, feats_lengths = self.specaug(feats, feats_lengths)

            # 3. Normalization for feature: e.g. Global-CMVN, Utterance-CMVN
            if self.normalize is not None:
                feats, feats_lengths = self.normalize(feats, feats_lengths)

        # Pre-encoder, e.g. used for raw input data
        if self.preencoder is not None:
            feats, feats_lengths = self.preencoder(feats, feats_lengths)

        # 4. Forward encoder
        # feats: (Batch, Length, Dim)
        # -> encoder_out: (Batch, Length2, Dim2)
        encoder_out, encoder_out_lens, _ = self.encoder(feats, feats_lengths)

        # Added by Minwoo
        if self.use_transducer_decoder \
            and self.use_auxiliary_loss \
            and isinstance(encoder_out, tuple):
            encoder_out, aux_encoder_out = encoder_out[0], encoder_out[1]
            encoder_out_lens, aux_encoder_out_lens = encoder_out_lens[0], encoder_out_lens[1]

            # Post-encoder, e.g. NLU
            if self.postencoder is not None:
                encoder_out, encoder_out_lens = self.postencoder(
                    encoder_out, encoder_out_lens
                )

            assert encoder_out.size(0) == speech.size(0), (
                encoder_out.size(),
                speech.size(0),
            )
            assert encoder_out.size(1) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )
            if self.training:
                encoder_out = (encoder_out, aux_encoder_out)
                encoder_out_lens = (encoder_out_lens, aux_encoder_out_lens)
        else:
            # Post-encoder, e.g. NLU
            if self.postencoder is not None:
                encoder_out, encoder_out_lens = self.postencoder(
                    encoder_out, encoder_out_lens
                )

            assert encoder_out.size(0) == speech.size(0), (
                encoder_out.size(),
                speech.size(0),
            )
            assert encoder_out.size(1) <= encoder_out_lens.max(), (
                encoder_out.size(),
                encoder_out_lens.max(),
            )

        return encoder_out, encoder_out_lens

    def _extract_feats(
        self, speech: torch.Tensor, speech_lengths: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        assert speech_lengths.dim() == 1, speech_lengths.shape

        # for data-parallel
        speech = speech[:, : speech_lengths.max()]

        if self.frontend is not None:
            # Frontend
            #  e.g. STFT and Feature extract
            #       data_loader may send time-domain signal in this case
            # speech (Batch, NSamples) -> feats: (Batch, NFrames, Dim)
            feats, feats_lengths = self.frontend(speech, speech_lengths)
        else:
            # No frontend and no feature extract
            feats, feats_lengths = speech, speech_lengths
        return feats, feats_lengths

    def nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative log likelihood(nll) from transformer-decoder

        Normally, this function is called in batchify_nll.

        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
        """
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )  # [batch, seqlen, dim]
        batch_size = decoder_out.size(0)
        decoder_num_class = decoder_out.size(2)
        # nll: negative log-likelihood
        nll = torch.nn.functional.cross_entropy(
            decoder_out.view(-1, decoder_num_class),
            ys_out_pad.view(-1),
            ignore_index=self.ignore_id,
            reduction="none",
        )
        nll = nll.view(batch_size, -1)
        nll = nll.sum(dim=1)
        assert nll.size(0) == batch_size
        return nll

    def batchify_nll(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
        batch_size: int = 100,
    ):
        """Compute negative log likelihood(nll) from transformer-decoder

        To avoid OOM, this fuction seperate the input into batches.
        Then call nll for each batch and combine and return results.
        Args:
            encoder_out: (Batch, Length, Dim)
            encoder_out_lens: (Batch,)
            ys_pad: (Batch, Length)
            ys_pad_lens: (Batch,)
            batch_size: int, samples each batch contain when computing nll,
                        you may change this to avoid OOM or increase
                        GPU memory usage
        """
        total_num = encoder_out.size(0)
        if total_num <= batch_size:
            nll = self.nll(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)
        else:
            nll = []
            start_idx = 0
            while True:
                end_idx = min(start_idx + batch_size, total_num)
                batch_encoder_out = encoder_out[start_idx:end_idx, :, :]
                batch_encoder_out_lens = encoder_out_lens[start_idx:end_idx]
                batch_ys_pad = ys_pad[start_idx:end_idx, :]
                batch_ys_pad_lens = ys_pad_lens[start_idx:end_idx]
                batch_nll = self.nll(
                    batch_encoder_out,
                    batch_encoder_out_lens,
                    batch_ys_pad,
                    batch_ys_pad_lens,
                )
                nll.append(batch_nll)
                start_idx = end_idx
                if start_idx == total_num:
                    break
            nll = torch.cat(nll)
        assert nll.size(0) == total_num
        return nll

    def _calc_att_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        ys_in_pad, ys_out_pad = add_sos_eos(ys_pad, self.sos, self.eos, self.ignore_id)
        ys_in_lens = ys_pad_lens + 1

        # 1. Forward decoder
        decoder_out, _ = self.decoder(
            encoder_out, encoder_out_lens, ys_in_pad, ys_in_lens
        )

        # 2. Compute attention loss
        loss_att = self.criterion_att(decoder_out, ys_out_pad)
        acc_att = th_accuracy(
            decoder_out.view(-1, self.vocab_size),
            ys_out_pad,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = decoder_out.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), ys_pad.cpu())

        return loss_att, acc_att, cer_att, wer_att

    def _calc_ctc_loss(
        self,
        encoder_out: torch.Tensor,
        encoder_out_lens: torch.Tensor,
        ys_pad: torch.Tensor,
        ys_pad_lens: torch.Tensor,
    ):
        # Calc CTC loss
        loss_ctc = self.ctc(encoder_out, encoder_out_lens, ys_pad, ys_pad_lens)

        # Calc CER using CTC
        cer_ctc = None
        if not self.training and self.error_calculator is not None:
            ys_hat = self.ctc.argmax(encoder_out).data
            cer_ctc = self.error_calculator(ys_hat.cpu(), ys_pad.cpu(), is_ctc=True)
        return loss_ctc, cer_ctc

    def _calc_transducer_loss(
        self,
        enc_out: torch.Tensor,
        dec_out: torch.tensor,
        target: torch.Tensor,
        t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute Transducer loss.

        Args:
            enc_out: Encoder output sequences. (B, T, D_enc)
            dec_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, L)
            t_len: Time lengths. (B,)
            u_len: Label lengths. (B,)

        Returns:
            (joint_out, loss_trans):
                Joint output sequences. (B, T, U, D_joint),
                Transducer loss value.

        """
        joint_out = self.joint_network(enc_out.unsqueeze(2), dec_out.unsqueeze(1))

        loss_trans = self.criterion_transducer(joint_out, target, t_len, u_len)
        loss_trans /= joint_out.size(0)

        return joint_out, loss_trans

    def _calc_transducer_tasks(
        self,
        encoder_out: torch.Tensor,
        aux_encoder_out: List[torch.Tensor],
        encoder_out_lens: torch.Tensor,
        aux_encoder_out_lens: Optional[List],
        labels: torch.Tensor,
    ):
        """Compute Transducer loss.

        Args:
            encoder_out: Encoder output sequences. (B, T, D_enc)
            aux_encoder_out: Encoder intermediate output sequences. (B, T_aux, D_enc_aux)
            encoder_out_lens: Encoder output sequences lengths. (B,)
            aux_encoder_out_lens: Auxiliary Encoder output sequences lengths. (B,)
            labels: Label ID sequences. (B, L)

        Return:
            loss_transducer: Transducer loss value.
            cer_transducer: Character error rate for Transducer.
            wer_transducer: Word Error Rate for Transducer.

        """
        loss_transducer = None
        loss_aux_trans = None
        loss_symm_kl_div = None
        loss_att, acc_att, cer_att, wer_att = None, None, None, None

        # 2. decoder
        decoder_in = get_decoder_input(labels, self.blank_id, self.ignore_id)

        self.decoder.set_device(encoder_out.device)
        decoder_out = self.decoder(decoder_in)

        target, lm_loss_target, t_len, aux_t_len, u_len = get_transducer_tasks_io(
            labels,
            encoder_out_lens,
            aux_encoder_out_lens,
            ignore_id=self.ignore_id,
            blank_id=self.blank_id,
        )

        joint_out, loss_transducer = self._calc_transducer_loss(
            encoder_out, decoder_out, target, t_len, u_len
        )

        if self.training and self.use_auxiliary_loss:
            (
                loss_aux_trans,
                loss_symm_kl_div,
            ) = self._calc_aux_transducer_and_symm_kl_div_losses(
                aux_encoder_out,
                decoder_out,
                joint_out,
                target,
                aux_t_len,
                u_len,
            )

        if self.training and self.use_lm_loss:
            (
                loss_att,
                acc_att,
                cer_att,
                wer_att
            ) = self._calc_att_loss_for_transducer(
                decoder_out,
                lm_loss_target
            )

        cer_transducer, wer_transducer = None, None
        if not self.training and self.error_calculator_trans is not None:
            cer_transducer, wer_transducer = self.error_calculator_trans(
                encoder_out, target
            )

        return (
            loss_att,
            loss_transducer,
            loss_aux_trans,
            loss_symm_kl_div,
            acc_att,
            cer_att,
            wer_att,
            cer_transducer,
            wer_transducer,
        )

    def _calc_aux_transducer_and_symm_kl_div_losses(
        self,
        aux_enc_out: torch.Tensor,
        dec_out: torch.Tensor,
        joint_out: torch.Tensor,
        target: torch.Tensor,
        aux_t_len: torch.Tensor,
        u_len: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute auxiliary Transducer loss and Jensen-Shannon divergence loss.

        Args:
            aux_enc_out: Encoder auxiliary output sequences. [N x (B, T_aux, D_enc_aux)]
            dec_out: Decoder output sequences. (B, U, D_dec)
            joint_out: Joint output sequences. (B, T, U, D_joint)
            target: Target character ID sequences. (B, L)
            aux_t_len: Auxiliary time lengths. [N x (B,)]
            u_len: True U lengths. (B,)

        Returns:
           : Auxiliary Transducer loss and KL divergence loss values.

        """
        loss_aux_trans = 0
        loss_kl_div = 0

        num_aux_layers = len(aux_enc_out)
        B, T, U, D = joint_out.shape

        for param in self.joint_network.parameters():
            param.requires_grad = False

        for i, aux_enc_out_i in enumerate(aux_enc_out):
            aux_mlp = self.mlp(aux_enc_out_i)

            aux_joint_out = self.joint_network(
                aux_mlp.unsqueeze(2),
                dec_out.unsqueeze(1),
                is_aux=True,
            )
            loss_aux_trans += self.criterion_transducer(
                aux_joint_out,
                target,
                aux_t_len[i],
                u_len,
            ) / B

            if self.use_symm_kl_div_loss:
                denom = B * T * U

                kl_main_aux = (
                    self.kl_div(
                        torch.log_softmax(joint_out, dim=-1),
                        torch.softmax(aux_joint_out, dim=-1),
                    )
                    / denom
                )

                kl_aux_main = (
                    self.kl_div(
                        torch.log_softmax(aux_joint_out, dim=-1),
                        torch.softmax(joint_out, dim=-1),
                    )
                    / denom
                )

                loss_kl_div += kl_main_aux + kl_aux_main

        for param in self.joint_network.parameters():
            param.requires_grad = True

        loss_aux_trans /= num_aux_layers

        if self.use_symm_kl_div_loss:
            loss_kl_div /= num_aux_layers

        return loss_aux_trans, loss_kl_div

    def _calc_att_loss_for_transducer(
        self,
        dec_out: torch.Tensor,
        target: torch.Tensor,
    ) -> torch.Tensor:
        """Forward LM loss.

        Args:
            dec_out: Decoder output sequences. (B, U, D_dec)
            target: Target label ID sequences. (B, U)

        Returns:
            : LM loss value.

        """
        lm_lin = self.lm_lin(dec_out)
        loss_att = self.criterion_att(lm_lin, target)
        acc_att = th_accuracy(
            lm_lin.view(-1, self.vocab_size),
            target,
            ignore_label=self.ignore_id,
        )

        # Compute cer/wer using attention-decoder
        if self.training or self.error_calculator is None:
            cer_att, wer_att = None, None
        else:
            ys_hat = lm_lin.argmax(dim=-1)
            cer_att, wer_att = self.error_calculator(ys_hat.cpu(), target.cpu())

        return loss_att, acc_att, cer_att, wer_att
