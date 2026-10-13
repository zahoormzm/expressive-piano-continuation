"""Allocate continuation tokens within the selected model's context."""


def parse_token_budget(value):
    if value == "auto":
        return value
    tokens = int(value)
    if tokens < 3:
        raise ValueError("At least three new tokens are required")
    return tokens


def resolve_token_budget(requested, prompt_tokens, context_tokens):
    available = context_tokens - prompt_tokens
    if available < 3:
        raise ValueError("Prompt leaves no complete note within model context; shorten the prompt")
    if requested == "auto":
        return available
    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 3:
        raise ValueError("At least three new tokens or 'auto' are required")
    if requested > available:
        raise ValueError(
            "Prompt plus generation exceeds model context; shorten the prompt or token budget"
        )
    return requested
