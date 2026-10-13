import copy

import pytest
import torch

from music_project.models import build_model
from music_project.training.engine import loss_sum, optimize


@pytest.fixture(params=["lstm", "transformer"])
def model(request):
    torch.manual_seed(7)
    torch.set_num_threads(1)
    config = (
        {"kind": "lstm", "width": 16, "layers": 1, "dropout": 0.0}
        if request.param == "lstm"
        else {
            "kind": "transformer",
            "width": 16,
            "layers": 2,
            "heads": 4,
            "feedforward": 32,
            "dropout": 0.0,
        }
    )
    return build_model(config, 32, 0, 12)


def test_future_tokens_cannot_change_earlier_predictions(model):
    model.eval()
    first = torch.tensor([[1, 2, 3, 4, 5, 6]])
    changed = first.clone()
    changed[:, 4:] = torch.tensor([17, 18])
    mask = torch.ones_like(first, dtype=torch.bool)
    torch.testing.assert_close(
        model(first, mask)[:, :4], model(changed, mask)[:, :4], rtol=0, atol=1e-6
    )


def test_right_padding_does_not_change_valid_predictions(model):
    model.eval()
    first = torch.tensor([[1, 2, 3, 0, 0]])
    changed = torch.tensor([[1, 2, 3, 28, 29]])
    mask = torch.tensor([[True, True, True, False, False]])
    torch.testing.assert_close(
        model(first, mask)[:, :3], model(changed, mask)[:, :3], rtol=0, atol=1e-6
    )


def test_valid_token_weighted_accumulation_matches_combined_batch(model):
    accumulated = copy.deepcopy(model)
    inputs = torch.tensor([[1, 2, 3, 0], [4, 5, 6, 7]])
    targets = torch.tensor([[2, 3, -100, -100], [5, 6, 7, 8]])
    mask = inputs != 0

    def batch(start, end):
        return {
            "input_ids": inputs[start:end],
            "targets": targets[start:end],
            "attention_mask": mask[start:end],
            "valid_tokens": int((targets[start:end] != -100).sum()),
        }

    first_optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    second_optimizer = torch.optim.SGD(accumulated.parameters(), lr=0.1)
    whole = optimize(model, first_optimizer, [batch(0, 2)], torch.device("cpu"), "fp32", 1e6)
    partial = optimize(
        accumulated, second_optimizer, [batch(0, 1), batch(1, 2)], torch.device("cpu"), "fp32", 1e6
    )
    assert whole["valid_tokens"] == partial["valid_tokens"] == 6
    assert whole["loss"] == pytest.approx(partial["loss"], abs=1e-6)
    for before, after in zip(model.parameters(), accumulated.parameters()):
        torch.testing.assert_close(before, after, rtol=0, atol=1e-6)


def test_masked_targets_contribute_no_loss_or_gradient():
    logits = torch.randn(1, 4, 10, requires_grad=True)
    targets = torch.tensor([[2, -100, 3, -100]])
    value = loss_sum(logits, targets)
    value.backward()
    assert torch.count_nonzero(logits.grad[:, [1, 3]]) == 0
    changed = logits.detach().clone()
    changed[:, [1, 3]] = 1000
    torch.testing.assert_close(value.detach(), loss_sum(changed, targets))
