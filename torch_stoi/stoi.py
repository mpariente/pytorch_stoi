import torch
from torch import nn
import numpy as np
from torch.nn.functional import unfold

from pystoi.stoi import FS, N_FRAME, NUMBAND, MINFREQ, N, BETA, DYN_RANGE
from pystoi.utils import thirdoct
EPS = 1e-8


class NegSTOILoss(nn.Module):
    """ Negated Short Term Objective Intelligibility (STOI) metric, to be used
        as a loss function.
        Inspired from [1, 2, 3] but not exactly the same : cannot be used as
        the STOI metric directly (use pystoi instead). See Notes.

    Args:
        sample_rate (int): sample rate of the audio files
        use_vad (bool): Whether to use simple VAD (see Notes)
        extended (bool): Whether to compute extended version [3].

    Shapes:
        (time,) --> (1, )
        (batch, time) --> (batch, )
        (batch, n_src, time) --> (batch, n_src)

    Returns:
        torch.Tensor of shape (batch, *, ), only the time dimension has
        been reduced.

    Warnings:
        This function cannot be used to compute the "real" STOI metric as
        we applied some changes to speed-up loss computation. See Notes section.

    Notes:
        - In the NumPy version, both signals were resampled to 10kHz
        as was done in the matlab code as well. Here, we take the choice
        not the resample the signal but to adapt the constants to have
        a similar effect. What if the sampling rate doesn't match the
        10kHz then?
        We compute a STFT with the same window support (in milliseconds) as
        in the original version. The STFT is reduced to 1/3 octave band
        powers (still 15 bands).
        - In the NumPy version, some kind of simple VAD was used to remove the
        silent frames before chunking the signal into short-term envelope
        vectors. We don't do the same here because removing frames in a
        batch is cumbersome and inefficient.
        If `use_vad` is set to True, instead we detect the silent frames and
        keep a mask tensor. At the end, the normalized correlation of
        short-term envelope vectors is masked using this mask (unfolded) and
        the mean is computed taking the mask values into account.

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
    def __init__(self,
                 sample_rate: int = 10000,
                 use_vad: bool = True,
                 extended: bool = False):
        super().__init__()
        # Independant from FS
        self.sample_rate = sample_rate
        self.use_vad = use_vad
        self.extended = extended
        self.intel_frames = N
        self.beta = BETA
        self.dyn_range = DYN_RANGE

        # Dependant from FS
        self.win_len = (N_FRAME * sample_rate) // FS
        self.nfft = 2 * self.win_len
        win = torch.from_numpy(np.hanning(self.win_len + 2)[1:-1]).float()
        self.win = nn.Parameter(win, requires_grad=False)
        obm_mat = thirdoct(sample_rate, self.nfft, NUMBAND, MINFREQ)[0]
        self.OBM = nn.Parameter(torch.from_numpy(obm_mat).float(),
                                requires_grad=False)

    def forward(self, est_targets: torch.Tensor,
                targets: torch.Tensor,) -> torch.Tensor:
        """ Compute negative (E)STOI loss.

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
            raise RuntimeError('targets and est_targets should have '
                               'the same shape, found {} and '
                               '{}'.format(targets.shape, est_targets.shape))
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

        # Here comes the real computation, take STFT
        x_spec = self.stft(targets, self.win, self.nfft, overlap=2)
        y_spec = self.stft(est_targets, self.win, self.nfft, overlap=2)
        # Apply OB matrix to the spectrograms as in Eq. (1)
        x_tob = torch.matmul(self.OBM, torch.norm(x_spec, 2, -1) ** 2).pow(0.5)
        y_tob = torch.matmul(self.OBM, torch.norm(y_spec, 2, -1) ** 2).pow(0.5)
        # Perform N-frame segmentation --> (batch, 15, N, n_chunks)
        batch = targets.shape[0]
        x_seg = unfold(x_tob.unsqueeze(2),
                       kernel_size=(1, self.intel_frames),
                       stride=(1, 1)).view(batch, x_tob.shape[1], N, -1)
        y_seg = unfold(y_tob.unsqueeze(2),
                       kernel_size=(1, self.intel_frames),
                       stride=(1, 1)).view(batch, y_tob.shape[1], N, -1)
        if self.extended:
            # Normalize rows and columns of intermediate intelligibility frames
            x_n = self.rowcol_norm(x_seg)
            y_n = self.rowcol_norm(y_seg)
            corr_comp = x_n * y_n
            correction = self.intel_frames * x_n.shape[-1]
        else:
            # Find normalization constants and normalize
            norm_const = (torch.norm(x_seg, 2, dim=2, keepdim=True) /
                          (torch.norm(y_seg, 2, dim=2, keepdim=True) + EPS))
            y_seg_normed = y_seg * norm_const
            # Clip as described in [1]
            clip_val = 10 ** (-self.beta / 20)
            y_prim = torch.min(y_seg_normed, x_seg * (1 + clip_val))
            # Mean/var normalize vectors
            y_prim = meanvar_norm(y_prim, dim=2)
            x_seg = meanvar_norm(x_seg, dim=2)
            # Matrix with entries summing to sum of correlations of vectors
            corr_comp = y_prim * x_seg
            # J, M as in [1], eq.6
            correction = x_seg.shape[1] * x_seg.shape[-1]

        # Compute average (E)STOI w. or w/o VAD.
        sum_over = list(range(1, x_seg.ndim))  # Keep batch dim
        if self.use_vad:
            # Detech silent frames (boolean mask of shape (batch, 1, frame_idx)
            mask = self.detect_silent_frames(targets, self.dyn_range,
                                             self.win_len, self.win_len // 2)
            # Unfold on the mask, to float and mean per frame.
            mask_f = unfold(mask.unsqueeze(2).float(),
                            kernel_size=(1, self.intel_frames),
                            stride=(1, 1)).view(batch, 1, N, -1)
            corr_comp = corr_comp * mask_f
            correction = correction * mask_f.mean(dim=sum_over)
        # Return -(E)STOI to optimize for
        return - torch.sum(corr_comp, dim=sum_over) / correction

    @staticmethod
    def detect_silent_frames(x, dyn_range, framelen, hop):
        """ Detects silent frames on input tensor.
        A frame is excluded if its energy is lower than max(energy) - dyn_range

        Args:
            x (torch.Tensor): batch of original speech wav file  (batch, time)
            dyn_range : Energy range to determine which frame is silent
            framelen : Window size for energy evaluation
            hop : Hop size for energy evaluation

        Returns:
            torch.BoolTensor, framewise mask.
        """
        x_frames = unfold(x[:, None, None, :], kernel_size=(1, framelen),
                          stride=(1, hop))[..., :-1]
        # Compute energies in dB
        x_energies = 20 * torch.log10(torch.norm(x_frames, dim=1,
                                                 keepdim=True) + EPS)
        # Find boolean mask of energies lower than dynamic_range dB
        # with respect to maximum clean speech energy frame
        mask = (torch.max(x_energies, dim=2, keepdim=True)[0] - dyn_range -
                x_energies) < 0
        return mask

    @staticmethod
    def stft(x, win, fft_size, overlap=4):
        win_len = win.shape[0]
        hop = int(win_len / overlap)
        # Last frame not taken because NFFT size is larger, torch bug IMO.
        x_padded = torch.nn.functional.pad(x, pad=[0, hop])
        return torch.stft(x_padded, fft_size, hop_length=hop, window=win,
                          center=False, win_length=win_len)

    @staticmethod
    def rowcol_norm(x):
        """ Mean/variance normalize axis 2 and 1 of input vector"""
        for dim in [2, 1]:
            x = meanvar_norm(x, dim=dim)
        return x


def meanvar_norm(x, dim=-1):
    x = (x - x.mean(dim=dim, keepdim=True)) / (
            x.norm(p=2, dim=dim, keepdim=True) + EPS)
    return x
