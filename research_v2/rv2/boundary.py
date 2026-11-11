"""Answer boundary, answer-relevant token set, and scoring (PILOT_SPEC.md section 2).

Scoring is delegated to the HISTORICAL extractor (scripts/gsm8k_gate_check.py) so that outcomes
remain comparable with every stored result. The boundary regex is the extractor's own
CONTINUATION_DRIFT_RE object, so the boundary and the scored text can never disagree.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from . import PROJECT_ROOT  # noqa: F401  (sets sys.path for the historical modules)
from gsm8k_gate_check import (  # historical, unmodified
    CONTINUATION_DRIFT_RE,
    NUMBER_RE,
    clean_number,
    extract_candidate_answer,
    extract_ground_truth,
    numeric_match,
    truncate_at_continuation_drift,
)


@dataclass
class Boundary:
    boundary_char: int          # scored text = full_text[:boundary_char]
    boundary_type: str          # "drift_marker" | "eos" | "cap"
    n_tokens: int               # generated tokens incl. token 0 (prefill-produced)
    n_answer_tokens: int        # tokens k with start offset < boundary_char (incl. token 0 if so)
    answer_token_mask: list     # per generated token k: 1 if answer-relevant
    offsets_consistent: bool    # every decode(ids[:k]) is a prefix of decode(ids)
    marker_token_index: int     # index of the token in which the marker starts, -1 if none
    decision_char: int          # end of the number the extractor reads after the first "####"
                                # (strategy A); equals boundary_char when strategy A does not apply
    decision_type: str          # "hash_number" | "no_hash" (falls back to boundary_char)
    decision_token_mask: list   # per token k: 1 if start offset < decision_char
    n_decision_tokens: int

    def to_dict(self):
        return asdict(self)


def decode(tokenizer, ids) -> str:
    # Same decode call as DynamicQwen.generate / generate_with_override / random routing.
    return tokenizer.decode(ids, skip_special_tokens=True)


def token_start_offsets(tokenizer, ids):
    """offsets[k] = len(decode(ids[:k])): character offset at which token k starts in the full text.

    Byte-level BPE can split a multi-byte character across tokens; a prefix decode then ends in a
    replacement character and is not a prefix of the full decode. Such cases are flagged, not hidden.
    """
    full = decode(tokenizer, ids)
    offsets, consistent = [], True
    for k in range(len(ids)):
        prefix = decode(tokenizer, ids[:k])
        if not full.startswith(prefix):
            consistent = False
        offsets.append(len(prefix))
    return offsets, consistent, full


def find_boundary(tokenizer, ids, eos_token_id, max_new_tokens) -> Boundary:
    offsets, consistent, full = token_start_offsets(tokenizer, ids)
    m = CONTINUATION_DRIFT_RE.search(full)
    if m:
        bchar, btype = m.start(), "drift_marker"
    elif eos_token_id is not None and len(ids) > 0 and ids[-1] == eos_token_id:
        bchar, btype = len(full), "eos"
    else:
        bchar, btype = len(full), "cap"
        assert len(ids) == max_new_tokens, (len(ids), max_new_tokens)
    mask = [1 if off < bchar else 0 for off in offsets]
    # EOS (skipped special token) decodes to "" -> its offset == len(full) >= bchar -> excluded.
    marker_tok = -1
    if m:
        # last token whose start offset <= marker start is the token containing the marker's first char
        marker_tok = max(k for k, off in enumerate(offsets) if off <= bchar)
    # Decision boundary (sensitivity definition): mirrors extract_candidate_answer's strategy A, which
    # reads the first number after the first "####" in the drift-truncated text. Text after that number
    # cannot change the extracted answer. Without a usable "####" number, strategies B/C read the last
    # matching phrase/number, so the whole truncated text matters and decision_char = boundary_char.
    truncated = full[:bchar]
    dchar, dtype = bchar, "no_hash"
    if "####" in truncated:
        start = truncated.index("####") + 4
        tail = truncated.split("####")[1]
        nm = NUMBER_RE.search(tail)
        if nm and clean_number(nm.group()) is not None:
            dchar, dtype = start + nm.end(), "hash_number"
    dmask = [1 if off < dchar else 0 for off in offsets]
    return Boundary(bchar, btype, len(ids), sum(mask), mask, consistent, marker_tok, dchar, dtype, dmask, sum(dmask))


def score(problem: dict, full_text: str, boundary: Boundary) -> dict:
    """Historical scoring on the full text, plus a consistency check on the truncated text."""
    gt = extract_ground_truth(problem["answer"])
    cand, strategy = extract_candidate_answer(full_text)
    if strategy == "extraction_failed":
        status = "EXTRACTION_FAILED"
    elif numeric_match(cand, gt):
        status = "SOLVED"
    else:
        status = "WRONG"
    # The extractor truncates internally; scoring text[:boundary] must give the identical result.
    cand2, strategy2 = extract_candidate_answer(full_text[: boundary.boundary_char])
    assert truncate_at_continuation_drift(full_text) == full_text[: boundary.boundary_char]
    cand3, strategy3 = extract_candidate_answer(full_text[: boundary.decision_char])
    return {
        "status": status,
        "candidate": cand,
        "strategy": strategy,
        "ground_truth": gt,
        "truncated_extraction_identical": (cand2, strategy2) == (cand, strategy),
        # strategy A only: the answer is fully determined by text up to decision_char
        "decision_extraction_identical": (cand3, strategy3) == (cand, strategy),
    }


def would_stop_at_boundary(tokenizer, ids) -> bool:
    """Optional early-stopping predicate (NOT used in the pilot's primary protocol)."""
    return CONTINUATION_DRIFT_RE.search(decode(tokenizer, ids)) is not None
