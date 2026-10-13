"""Absolute-token syntax constraints shared by scratch and pretrained sampling."""

import math

import torch


class PianoGrammar:
    def __init__(self, vocabulary, prompt, start_ms, end_ms):
        self.vocabulary = vocabulary
        self.lookup = {
            tuple(token) if isinstance(token, list) else token: index
            for index, token in enumerate(vocabulary)
        }
        self.piano = [
            index
            for index, token in enumerate(vocabulary)
            if isinstance(token, (list, tuple))
            and token[0] == "piano"
            and 21 <= token[1] <= 108
            and token[2] > 0
        ]
        self.onsets = [
            (index, token[1])
            for index, token in enumerate(vocabulary)
            if isinstance(token, (list, tuple)) and token[0] == "onset"
        ]
        self.durations = [
            index
            for index, token in enumerate(vocabulary)
            if isinstance(token, (list, tuple)) and token[0] == "dur" and token[1] > 0
        ]
        self.phase, self.segment, self.last_onset = "note", 0, -1
        self.generated_notes, self.same_onset_notes = 0, 0
        self.pitches_at_onset, self.pending_pitch = set(), None
        self.start_ms, self.end_ms = start_ms, end_ms
        started = False
        for token_id in prompt:
            token = vocabulary[token_id]
            if token == "<S>":
                started = True
            elif started:
                self.consume(token_id, generated=False)
        if not started or self.phase != "note":
            raise ValueError("Prompt must start with BOS and end on a complete note group")

    def earliest(self):
        # A density ceiling prevents endlessly adding simultaneous notes.
        return max(
            self.start_ms,
            self.segment * 5000,
            self.last_onset + (10 if self.same_onset_notes >= 32 else 0),
        )

    def allowed_ids(self):
        if self.phase == "dur":
            return self.durations
        if self.phase == "onset":
            earliest = self.earliest()
            if self.pending_pitch in self.pitches_at_onset:
                earliest = max(earliest, self.last_onset + 10)
            return [
                index
                for index, onset in self.onsets
                if earliest <= self.segment * 5000 + onset < self.end_ms
            ]
        if self.phase == "ended":
            return []
        can_note = self.earliest() < min((self.segment + 1) * 5000, self.end_ms)
        allowed = list(self.piano) if can_note else []
        if self.last_onset + 10 >= min((self.segment + 1) * 5000, self.end_ms):
            allowed = [
                index for index in allowed if self.vocabulary[index][1] not in self.pitches_at_onset
            ]
        # Let a predicted clock transition reach the horizon, then stop.
        # Masking it out forces spurious notes into the last available onset.
        if self.segment * 5000 < self.end_ms:
            allowed.append(self.lookup["<T>"])
        if self.generated_notes:
            allowed.append(self.lookup["<E>"])
        return allowed

    def consume(self, token_id, generated=True):
        token = self.vocabulary[token_id]
        if self.phase == "note":
            if token == "<T>":
                self.segment += 1
            elif token == "<E>":
                self.phase = "ended"
            elif isinstance(token, (list, tuple)) and token[0] == "piano" and token[2] > 0:
                self.pending_pitch = token[1]
                self.phase = "onset"
            else:
                raise ValueError(f"Unexpected token at note boundary: {token}")
        elif self.phase == "onset":
            if not isinstance(token, (list, tuple)) or token[0] != "onset":
                raise ValueError("Pitch must be followed by onset")
            onset = self.segment * 5000 + token[1]
            if onset < self.last_onset:
                raise ValueError("Nonmonotonic onset")
            self.same_onset_notes = self.same_onset_notes + 1 if onset == self.last_onset else 1
            if onset != self.last_onset:
                self.pitches_at_onset = set()
            self.pitches_at_onset.add(self.pending_pitch)
            self.last_onset = onset
            self.phase = "dur"
        elif self.phase == "dur":
            if not isinstance(token, (list, tuple)) or token[0] != "dur" or token[1] <= 0:
                raise ValueError("Onset must be followed by positive duration")
            self.generated_notes += int(generated)
            self.phase = "note"
        else:
            raise ValueError("Tokens after EOS")


def sample_token(logits, allowed, temperature, min_p, generator):
    if not allowed:
        raise ValueError("No legal next token")
    if not math.isfinite(temperature) or temperature < 0 or not 0 <= min_p <= 1:
        raise ValueError("Finite nonnegative temperature and min_p in [0, 1] required")
    choices = torch.tensor(allowed, dtype=torch.long, device=logits.device)
    values = logits.float().index_select(0, choices)
    if not torch.isfinite(values).all():
        raise FloatingPointError("Nonfinite legal-token logits")
    if temperature == 0:
        return int(choices[values.argmax()].item())
    probabilities = torch.softmax(values / temperature, dim=-1)
    probabilities[probabilities < probabilities.max() * min_p] = 0
    probabilities /= probabilities.sum()
    chosen = torch.multinomial(probabilities, 1, generator=generator)
    return int(choices[chosen].item())


@torch.inference_mode()
def sample_sequence(
    predictor, vocabulary, prompt, start_ms, end_ms, budget, temperature, min_p, seed
):
    if budget < 3 or end_ms <= start_ms:
        raise ValueError("At least three new tokens and a positive continuation horizon required")
    grammar = PianoGrammar(vocabulary, prompt, start_ms, end_ms)
    generator = torch.Generator(device=predictor.device).manual_seed(seed)
    sequence = list(prompt)
    stop_reason = "token_budget"
    while len(sequence) - len(prompt) < budget:
        remaining = budget - (len(sequence) - len(prompt))
        if grammar.phase == "note" and remaining < 3:
            break
        allowed = grammar.allowed_ids()
        if not allowed:
            stop_reason = "time_horizon"
            break
        logits = predictor.next_logits(sequence)
        chosen = sample_token(logits, allowed, temperature, min_p, generator)
        sequence.append(chosen)
        grammar.consume(chosen)
        if grammar.phase == "ended":
            stop_reason = "sampled_eos"
            break
        if grammar.phase == "note" and grammar.segment * 5000 >= end_ms:
            stop_reason = "time_horizon"
            break
    if grammar.phase not in ("note", "ended"):
        raise RuntimeError("Generation ended inside a note group")
    if not grammar.generated_notes:
        raise ValueError("Model produced no continuation notes within the requested horizon")
    return sequence, {
        "stop_reason": stop_reason,
        "new_tokens": len(sequence) - len(prompt),
        "generated_note_groups": grammar.generated_notes,
        "sampled_eos": grammar.phase == "ended",
        "sampling": "shared syntax/range/density constraints, temperature and min_p",
        "constraints": {
            "pitch_range": [21, 108],
            "positive_velocity_and_duration": True,
            "monotonic_onsets": True,
            "max_simultaneous_note_groups": 32,
            "duplicate_pitch_at_same_onset": "forbidden",
            "new_note_onsets_ms": [start_ms, end_ms],
        },
    }
