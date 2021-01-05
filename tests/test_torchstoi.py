import torch
from torch import nn
from torch.testing import assert_allclose
import random
import pytest

from torch_stoi import NegSTOILoss


@pytest.mark.parametrize("sample_rate", [8000, 10000, 16000])
@pytest.mark.parametrize("use_vad", [True, False])
@pytest.mark.parametrize("extended", [True, False])
def test_forward(sample_rate, use_vad, extended):
    loss_func = NegSTOILoss(sample_rate=sample_rate, use_vad=use_vad,
                            extended=extended)
    batch_size = 3
    est_targets = torch.randn(batch_size, 2 * sample_rate)
    targets = torch.randn(batch_size, 2 * sample_rate)
    loss_val = loss_func(est_targets, targets)
    assert loss_val.ndim == 1
    assert loss_val.shape[0] == batch_size


@pytest.mark.parametrize("sample_rate", [8000, 10000, 16000])
@pytest.mark.parametrize("use_vad", [True, False])
@pytest.mark.parametrize("extended", [True, False])
def test_backward(sample_rate, use_vad, extended):
    nnet = TestNet()
    loss_func = NegSTOILoss(sample_rate=sample_rate, use_vad=use_vad,
                            extended=extended)
    batch_size = 3
    mix = torch.randn(batch_size, 2 * sample_rate, requires_grad=True)
    targets = torch.randn(batch_size, 2 * sample_rate)
    est_targets = nnet(mix)
    loss_val = loss_func(est_targets, targets).mean()
    loss_val.backward()
    # Check that gradients exist
    assert nnet.conv.weight.grad is not None
    assert nnet.conv.bias.grad is not None


@pytest.mark.parametrize("ndim", [1, 2, 3, 4])
def test_shapes(ndim):
    loss_dim = [random.randint(1, 4) for _ in range(ndim - 1)]
    t_dim = loss_dim + [10000]
    targets = torch.randn(t_dim)
    est_targets = torch.randn(t_dim)
    loss_func = NegSTOILoss(sample_rate=10000)
    loss_batch = loss_func(est_targets, targets)
    assert loss_batch.shape == targets.shape[:-1]


@pytest.mark.parametrize("use_vad", [True, False])
@pytest.mark.parametrize("extended", [True, False])
def test_batchonly_equal(use_vad, extended):
    loss_func = NegSTOILoss(sample_rate=10000, use_vad=use_vad, extended=extended)
    targets = torch.randn(3, 2, 16000)
    est_targets = torch.randn(3, 2, 16000)
    threed = loss_func(est_targets, targets)
    twod = loss_func(est_targets[:, 0], targets[:, 0])
    oned = loss_func(est_targets[0, 0], targets[0, 0])
    assert_allclose(oned, twod[0])
    assert_allclose(twod, threed[:, 0])


@pytest.mark.parametrize("use_vad", [True, False])
@pytest.mark.parametrize("extended", [True, False])
def test_getbetter(use_vad, extended):
    loss_func = NegSTOILoss(sample_rate=10000, use_vad=use_vad, extended=extended)
    targets = torch.randn(1, 16000)
    old_val = None
    for eps in [5, 2, 1, 0.5, 0.1, 0.01]:
        est_targets = targets + eps * torch.randn(1, 16000)
        new_val = loss_func(est_targets, targets).mean()
        # First iteration is skipped
        if old_val is None:
            continue
        assert new_val < old_val
        old_val = new_val


@pytest.mark.parametrize("use_vad", [True, False])
@pytest.mark.parametrize("extended", [True, False])
@pytest.mark.parametrize("iteration", list(range(4)))
def test_more_than_minusone(use_vad, extended, iteration):
    loss_func = NegSTOILoss(sample_rate=10000, use_vad=use_vad, extended=extended)
    targets = torch.randn(1, 16000)
    est_targets = torch.randn(1, 16000)
    loss_vals = loss_func(est_targets, targets)
    assert (loss_vals > -1).all()


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv1d(1, 1, 1)

    def forward(self, x):
        x_len = x.shape[-1]
        return self.conv(x.view(-1, 1, x_len)).view(x.shape)
