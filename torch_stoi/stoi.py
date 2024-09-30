import torch
from torch import nn
import numpy as np
from torch.nn.functional import unfold, pad
import torchaudio

from pystoi.stoi import FS, N_FRAME, NUMBAND, MINFREQ, N, BETA, DYN_RANGE
from pystoi.utils import thirdoct

EPS = 1e-8


class NegSTOILoss(nn.Module):
    """Negated Short Term Objective Intelligibility (STOI) metric, to be used
        as a loss function.
        Inspired from [1, 2, 3] but not exactly the same due to a different
        resampling technique. Use pystoi when evaluating your system.

    Args:
        sample_rate (int): sample rate of audio input
        use_vad (bool): Whether to use simple VAD (see Notes)
        extended (bool): Whether to compute extended version [3].
        do_resample (bool): Whether to resample audio input to `FS`

    Shapes:
        (time,) --> (1, )
        (batch, time) --> (batch, )
        (batch, n_src, time) --> (batch, n_src)

    Returns:
        torch.Tensor of shape (batch, *, ), only the time dimension has
        been reduced.

    Warnings:
        This function does not exactly match the "real" STOI metric due to a
        different resampling technique. Use pystoi when evaluating your system.

    Notes:
        `use_vad` can be set to `False` to skip the VAD for efficiency. However
        results can become substantially different compared to the "real" STOI.
        When `True` (default), results are very close but still slightly
        different due to a different resampling technique.
        Compared against mpariente/pystoi@84b1bd8.

    References
        [1] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'A Short-Time
            Objective Intelligibility Measure for Time-Frequency Weighted Noisy
            Speech', ICASSP 2010, Texas, Dallas.
        [2] C.H.Taal, R.C.Hendriks, R.Heusdens, J.Jensen 'An Algorithm for
            Intelligibility Prediction of Time-Frequency Weighted Noisy Speech',
            IEEE Transactions on Audio, Speech, and Language Processing, 2011.
        [3] Jesper Jensen and Cees H. Taal, 'An Algorithm for Predicting the
            Intelligibility of Speech Masked by Modulated Noise Maskers',
            IEEE Transactions on Audio, Speech and Language Processing, 2016.
    """

    def __init__(
        self,
        sample_rate: int,
        use_vad: bool = True,
        extended: bool = False,
        do_resample: bool = True,
    ):
        super().__init__()
        # Independant from FS
        self.sample_rate = sample_rate
        self.use_vad = use_vad
        self.extended = extended
        self.intel_frames = N
        self.beta = BETA
        self.dyn_range = DYN_RANGE
        self.do_resample = do_resample

        # Dependant from FS
        if self.do_resample:
            sample_rate = FS
            self.resample = torchaudio.transforms.Resample(
                orig_freq=self.sample_rate,
                new_freq=FS,
                resampling_method="sinc_interpolation",
            )
        self.win_len = (N_FRAME * sample_rate) // FS
        self.nfft = 2 * self.win_len
        win = torch.from_numpy(np.hanning(self.win_len + 2)[1:-1]).float()
        self.win = nn.Parameter(win, requires_grad=False)
        obm_mat = thirdoct(sample_rate, self.nfft, NUMBAND, MINFREQ)[0]
        self.OBM = nn.Parameter(torch.from_numpy(obm_mat).float(), requires_grad=False)

    def forward(
        self,
        est_targets: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        """Compute negative (E)STOI loss.

        Args:
            est_targets (torch.Tensor): Tensor containing target estimates.
            targets (torch.Tensor): Tensor containing clean targets.

        Shapes:
            (time,) --> (1, )
            (batch, time) --> (batch, )
            (batch, n_src, time) --> (batch, n_src)

        Returns:
            torch.Tensor, the batch of negative STOI loss
        """
        if targets.shape != est_targets.shape:
            raise RuntimeError(
                "targets and est_targets should have "
                "the same shape, found {} and "
                "{}".format(targets.shape, est_targets.shape)
            )
        # Compute STOI loss without batch size.
        if targets.ndim == 1:
            return self.forward(est_targets[None], targets[None])[0]
        # Pack additional dimensions in batch and unpack after forward
        if targets.ndim > 2:
            *inner, wav_len = targets.shape
            return self.forward(
                est_targets.view(-1, wav_len),
                targets.view(-1, wav_len),
            ).view(inner)
        batch_size = targets.shape[0]

        if self.do_resample and self.sample_rate != FS:
            targets = self.resample(targets)
            est_targets = self.resample(est_targets)

        if self.use_vad:
            targets, est_targets, mask = self.remove_silent_frames(
                targets,
                est_targets,
                self.dyn_range,
                self.win,
                self.win_len,
                self.win_len // 2,
            )
            # Remove the last mask frame to replicate pystoi behavior
            mask, _ = mask.sort(-1, descending=True)
            mask = mask[..., 1:]

        # Here comes the real computation, take STFT
        x_spec = self.stft(targets, self.win, self.nfft, overlap=2)
        y_spec = self.stft(est_targets, self.win, self.nfft, overlap=2)
        # Reapply the mask because of overlap in STFT
        if self.use_vad:
            x_spec *= mask.unsqueeze(1)
            y_spec *= mask.unsqueeze(1)

        """Uncommenting the following lines and the last block at the end
        allows to replicate the pystoi behavior when less than N speech frames
        are detected
        """
        # # Ensure at least 30 frames for intermediate intelligibility
        # if self.use_vad:
        #     not_enough = mask.sum(-1) < self.intel_frames
        #     if not_enough.any():
        #         import warnings
        #         warnings.warn('Not enough STFT frames to compute intermediate '
        #                       'intelligibility measure after removing silent '
        #                       'frames. Returning 1e-5. Please check you wav '
        #                       'files', RuntimeWarning)
        #         if not_enough.all():
        #             return torch.full(batch_size, 1e-5)
        #         x_spec = x_spec[~not_enough]
        #         y_spec = y_spec[~not_enough]
        #         mask = mask[~not_enough]

        # Apply OB matrix to the spectrograms as in Eq. (1)
        x_tob = (torch.matmul(self.OBM, x_spec.abs().pow(2)) + EPS).sqrt()
        y_tob = (torch.matmul(self.OBM, y_spec.abs().pow(2)) + EPS).sqrt()

        # Perform N-frame segmentation --> (batch, 15, N, n_chunks)
        x_seg = unfold(
            x_tob.unsqueeze(2), kernel_size=(1, self.intel_frames), stride=(1, 1)
        ).view(batch_size, x_tob.shape[1], self.intel_frames, -1)
        y_seg = unfold(
            y_tob.unsqueeze(2), kernel_size=(1, self.intel_frames), stride=(1, 1)
        ).view(batch_size, y_tob.shape[1], self.intel_frames, -1)
        # Reapply the mask because of overlap in N-frame segmentation
        if self.use_vad:
            mask = mask[..., self.intel_frames - 1 :]
            x_seg *= mask.unsqueeze(1).unsqueeze(2)
            y_seg *= mask.unsqueeze(1).unsqueeze(2)

        if self.extended:
            # Normalize rows and columns of intermediate intelligibility frames
            # No need to pass the mask because zeros do not affect statistics
            x_n = self.rowcol_norm(x_seg)
            y_n = self.rowcol_norm(y_seg)
            corr_comp = x_n * y_n
            corr_comp = corr_comp.sum(1)

        else:
            # Find normalization constants and normalize
            # No need to pass the mask because zeros do not affect statistics
            norm_const = x_seg.norm(p=2, dim=2, keepdim=True) / (
                y_seg.norm(p=2, dim=2, keepdim=True) + EPS
            )
            y_seg_normed = y_seg * norm_const
            # Clip as described in [1]
            clip_val = 10 ** (-self.beta / 20)
            y_prim = torch.min(y_seg_normed, x_seg * (1 + clip_val))
            # Mean/var normalize vectors
            # No need to pass the mask because zeros do not affect statistics
            y_prim = y_prim - y_prim.mean(2, keepdim=True)
            x_seg = x_seg - x_seg.mean(2, keepdim=True)
            y_prim = y_prim / (y_prim.norm(p=2, dim=2, keepdim=True) + EPS)
            x_seg = x_seg / (x_seg.norm(p=2, dim=2, keepdim=True) + EPS)
            # Matrix with entries summing to sum of correlations of vectors
            corr_comp = y_prim * x_seg
            corr_comp = corr_comp.sum(2)

        # Compute average (E)STOI w. or w/o VAD.
        output = corr_comp.mean(1)
        if self.use_vad:
            output = output.sum(-1) / (mask.sum(-1) + EPS)
        else:
            output = output.mean(-1)

        """Uncomment this to replicate the pystoi behavior when less than N
        speech frames are detected
        """
        # if np.any(not_enough):
        #     output_ = torch.empty(batch_size)
        #     output_[not_enough] = 1e-5
        #     output_[~not_enough] = output
        #     output = output_

        return -output

    @staticmethod
    def remove_silent_frames(x, y, dyn_range, window, framelen, hop):
        """Detects silent frames on input tensor.
        A frame is excluded if its energy is lower than max(energy) - dyn_range

        Args:
            x (torch.Tensor): batch of original speech wav file  (batch, time)
            dyn_range : Energy range to determine which frame is silent
            framelen : Window size for energy evaluation
            hop : Hop size for energy evaluation

        Returns:
            torch.BoolTensor, framewise mask.
        """
        x_frames = unfold(
            x[:, None, None, :], kernel_size=(1, framelen), stride=(1, hop)
        )
        y_frames = unfold(
            y[:, None, None, :], kernel_size=(1, framelen), stride=(1, hop)
        )
        x_frames *= window[:, None]
        y_frames *= window[:, None]

        # Compute energies in dB
        x_energies = 20 * torch.log10(torch.norm(x_frames, dim=1, keepdim=True) + EPS)
        # Find boolean mask of energies lower than dynamic_range dB
        # with respect to maximum clean speech energy frame
        mask = (x_energies.amax(2, keepdim=True) - dyn_range - x_energies) < 0
        mask = mask.squeeze(1)

        # Remove silent frames and pad with zeroes
        x_frames = x_frames.permute(0, 2, 1)
        y_frames = y_frames.permute(0, 2, 1)
        x_frames = _mask_audio(x_frames, mask)
        y_frames = _mask_audio(y_frames, mask)

        x_sil = _overlap_and_add(x_frames, hop)
        y_sil = _overlap_and_add(y_frames, hop)
        x_frames = x_frames.permute(0, 2, 1)
        y_frames = y_frames.permute(0, 2, 1)

        return x_sil, y_sil, mask.long()

    @staticmethod
    def stft(x, win, fft_size, overlap=4):
        """We can't use torch.stft:
        - It's buggy with center=False as it discards the last frame
        - It pads the frame left and right before taking the fft instead
        of padding right
        Instead we unfold and take rfft. This gives the same result as
        pystoi.utils.stft.
        """
        win_len = win.shape[0]
        hop = int(win_len / overlap)
        frames = unfold(x[:, None, None, :], kernel_size=(1, win_len), stride=(1, hop))[
            ..., :-1
        ]
        return torch.fft.rfft(frames * win[:, None], n=fft_size, dim=1)

    @staticmethod
    def rowcol_norm(x):
        """Mean/variance normalize axis 2 and 1 of input vector"""
        for dim in [2, 1]:
            x = x - x.mean(dim, keepdim=True)
            x = x / (x.norm(p=2, dim=dim, keepdim=True) + EPS)
        return x


def _overlap_and_add(x_frames, hop):
    batch_size, num_frames, framelen = x_frames.shape
    # Compute the number of segments, per frame.
    segments = -(-framelen // hop)  # Divide and round up.

    # Pad the framelen dimension to segments * hop and add n=segments frames
    signal = pad(x_frames, (0, segments * hop - framelen, 0, segments))

    # Reshape to a 4D tensor, splitting the framelen dimension in two
    signal = signal.reshape((batch_size, num_frames + segments, segments, hop))
    # Transpose dimensions so shape = (batch, segments, frame+segments, hop)
    signal = signal.permute(0, 2, 1, 3)
    # Reshape so that signal.shape = (batch, segments * (frame+segments), hop)
    signal = signal.reshape((batch_size, -1, hop))

    # Now behold the magic!! Remove last n=segments elements from second axis
    signal = signal[:, :-segments]
    # Reshape to (batch, segments, frame+segments-1, hop)
    signal = signal.reshape((batch_size, segments, num_frames + segments - 1, hop))
    # This has introduced a shift by one in all rows

    # Now, reduce over the columns and flatten the array to achieve the result
    signal = signal.sum(axis=1)
    end = (num_frames - 1) * hop + framelen
    signal = signal.reshape((batch_size, -1))[:end]
    return signal


def _mask_audio(x, mask):
    masked_audio = torch.stack(
        [pad(xi[mi], (0, 0, 0, len(xi) - mi.sum())) for xi, mi in zip(x, mask)]
    )
    return masked_audio
