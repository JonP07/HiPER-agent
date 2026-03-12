from dataclasses import dataclass
from typing import Tuple, Optional

import numpy as np
import torch


def _build_text_and_offsets(tokenizer, seq_ids) -> Tuple[str, list[Tuple[int, int]]]:
    pieces = tokenizer.batch_decode(
        [[int(i)] for i in seq_ids],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )
    offsets = []
    pos = 0
    for p in pieces:
        start = pos
        pos += len(p)
        offsets.append((start, pos))
    return "".join(pieces), offsets


def _char_to_token_start(offsets, s_char: int) -> Optional[int]:
    for i, (_, e) in enumerate(offsets):
        if e > s_char:
            return i
    return None


def _char_to_token_end(offsets, e_char: int) -> int:
    for i, (s, _) in enumerate(offsets):
        if s >= e_char:
            return i
    return len(offsets)


def _find_tag_char_span(text: str, open_tag: str, close_tags: list[str]) -> Optional[Tuple[int, int, int, int]]:
    s_open = text.find(open_tag)
    if s_open < 0:
        return None
    s_content = s_open + len(open_tag)
    s_close = None
    close_len = 0
    for tag in close_tags:
        idx = text.find(tag, s_content)
        if idx >= 0 and (s_close is None or idx < s_close):
            s_close = idx
            close_len = len(tag)
    if s_close is None:
        return None
    return s_open, s_content, s_close, close_len


def _span_from_tags_text(
    text: str,
    offsets,
    open_tag: str,
    close_tags: list[str],
    *,
    include_tags: bool = False,
) -> Optional[Tuple[int, int]]:
    sp = _find_tag_char_span(text, open_tag, close_tags)
    if sp is None:
        return None
    s_open, s_content, s_close, close_len = sp
    if include_tags:
        s_char = s_open
        e_char = s_close + close_len
    else:
        s_char = s_content
        e_char = s_close
    if s_char >= e_char:
        return None
    t_start = _char_to_token_start(offsets, s_char)
    if t_start is None:
        return None
    t_end = _char_to_token_end(offsets, e_char)
    if t_start >= t_end:
        return None
    return (t_start, t_end)


@dataclass
class HAEMaskConfig:
    # which switch values mean "do NOT start a new subgoal segment"
    keep_values: Tuple[str, ...] = ("KEEP",)
    # if tag extraction fails, fall back to training all valid response tokens as action
    fallback_action_to_full_response: bool = True
    # if switch tag missing, treat as KEEP by default
    default_switch_value: str = "KEEP"


@torch.no_grad()
def make_hae_masks_and_switch(
    batch,
    tokenizer,
    include_tags_mask: bool = False,
    cfg: HAEMaskConfig = HAEMaskConfig(),
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, np.ndarray]:
    """
    Returns:
      action_mask:  (N,L) bool
      subgoal_mask: (N,L) bool
      switch_mask:  (N,L) bool   (content inside <switch>...</switch>)
      is_new_subgoal: np.ndarray (N,) bool  True means start a new segment here
    """
    responses: torch.Tensor = batch.batch["responses"]          # (N,L) long
    response_mask: torch.Tensor = batch.batch["response_mask"]  # (N,L) 0/1
    device = responses.device
    response_mask = response_mask.to(torch.bool)

    N, L = responses.shape
    valid_lens = response_mask.long().sum(dim=1).cpu().numpy().astype(int)

    open_switch = "<switch>"
    close_switch_tags = ["</switch>", "</switch>\n"]
    open_subgoal = "<subgoal>"
    close_subgoal_tags = [
        "</subgoal>",
        "</subgoal>\n",
        "]</subgoal>",
        "]</subgoal>\n",
        ".</subgoal>",
        ".</subgoal>\n",
        ")</subgoal>",
        ")</subgoal>\n",
    ]
    open_action = "<action>"
    close_action_tags = [
        "</action>",
        "</action>\n",
        "]</action>",
        "]</action>\n",
        ".</action>",
        ".</action>\n",
        ")</action>",
        ")</action>\n",
    ]

    action_mask = torch.zeros((N, L), device=device, dtype=torch.bool)
    subgoal_mask = torch.zeros((N, L), device=device, dtype=torch.bool)
    switch_mask = torch.zeros((N, L), device=device, dtype=torch.bool)

    is_new_subgoal = np.zeros((N,), dtype=np.bool_)

    keep_set = {k.strip().upper() for k in cfg.keep_values}
    for i in range(N):
        vl = int(valid_lens[i])
        if vl <= 0:
            continue
        seq = responses[i, :vl]
        text, offsets = _build_text_and_offsets(tokenizer, seq.tolist())
        # pdb.set_trace()
        # ---------- ACTION mask (masking uses include_tags) ----------
        sp = _span_from_tags_text(
            text,
            offsets,
            open_action,
            close_action_tags,
            include_tags=include_tags_mask,
        )
        if sp is None:
            if cfg.fallback_action_to_full_response:
                action_mask[i, :vl] = True
        else:
            s, e = sp
            if s < e:
                action_mask[i, s:e] = True
        # pdb.set_trace()
        # ---------- SUBGOAL mask (masking uses include_tags) ----------
        sp = _span_from_tags_text(
            text,
            offsets,
            open_subgoal,
            close_subgoal_tags,
            include_tags=include_tags_mask,
        )
        if sp is not None:
            s, e = sp
            if s < e:
                subgoal_mask[i, s:e] = True

        # ---------- SWITCH: (1) content-only extraction, (2) include_tags for masking ----------
        # (1) content-only span for boundary decision
        sp_content = _find_tag_char_span(text, open_switch, close_switch_tags)

        # missing OR empty content => KEEP
        switch_text = cfg.default_switch_value.strip().upper()
        if sp_content is not None:
            _, s_c, s_e, _ = sp_content
            if s_c < s_e:  # non-empty content
                txt = text[s_c:s_e].strip().upper()
                if txt:
                    switch_text = txt

        # (2) mask span: include_tags controls whether tags are included in switch_mask
        sp_mask = _span_from_tags_text(
            text,
            offsets,
            open_switch,
            close_switch_tags,
            include_tags=include_tags_mask,
        )
        if sp_mask is not None:
            s_m, e_m = sp_mask
            if s_m < e_m:
                switch_mask[i, s_m:e_m] = True

        is_new_subgoal[i] = (switch_text not in keep_set)

    # ensure masks only on valid response tokens
    action_mask &= response_mask
    subgoal_mask &= response_mask
    switch_mask &= response_mask

    return action_mask, subgoal_mask, switch_mask, is_new_subgoal
