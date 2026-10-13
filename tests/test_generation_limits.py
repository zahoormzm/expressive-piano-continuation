import pytest

from music_project.generation.limits import parse_token_budget, resolve_token_budget


def test_auto_uses_context_remaining_after_actual_dense_prompt():
    assert resolve_token_budget("auto", 694, 2048) == 1354
    assert resolve_token_budget("auto", 694, 8192) == 7498
    assert resolve_token_budget("auto", 2045, 2048) == 3
    for prompt in [2046, 2048, 2050]:
        with pytest.raises(ValueError, match="context"):
            resolve_token_budget("auto", prompt, 2048)


def test_explicit_protocol_budget_is_preserved_and_cannot_overflow():
    assert parse_token_budget("auto") == "auto"
    assert parse_token_budget("384") == 384
    assert resolve_token_budget(384, 694, 2048) == 384
    assert resolve_token_budget(1354, 694, 2048) == 1354
    with pytest.raises(ValueError, match="context"):
        resolve_token_budget(1355, 694, 2048)
    for invalid in [0, 2, True, 3.5]:
        with pytest.raises(ValueError):
            resolve_token_budget(invalid, 694, 2048)
    for invalid in ["2", "bad"]:
        with pytest.raises(ValueError):
            parse_token_budget(invalid)
