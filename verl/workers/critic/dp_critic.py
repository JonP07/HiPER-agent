# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Implement a multiprocess PPOCritic
"""

import itertools
import logging
import os

import torch
import torch.distributed
from flash_attn.bert_padding import index_first_axis, pad_input, rearrange, unpad_input
from torch import nn, optim
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

from verl import DataProto
from verl.trainer.ppo import core_algos
from verl.utils.debug import GPUMemoryLogger
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import get_reverse_idx, rearrange_micro_batches
from verl.utils.torch_functional import masked_mean
from verl.utils.ulysses import gather_outpus_and_unpad, ulysses_pad_and_slice_inputs
from verl.workers.critic import BasePPOCritic
from verl.utils.device import get_device_name, get_torch_device, is_npu_available, is_cuda_available


if is_cuda_available:
    from flash_attn.bert_padding import pad_input, unpad_input, rearrange, index_first_axis
elif is_npu_available:
    from transformers.integrations.npu_flash_attention import pad_input, unpad_input, rearrange, index_first_axis

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOCritic(BasePPOCritic):
    def __init__(self, config, critic_module: nn.Module, critic_optimizer: optim.Optimizer):
        super().__init__(config=config)
        self.critic_module = critic_module
        self.critic_optimizer = critic_optimizer
        self.use_remove_padding = self.config.model.get("use_remove_padding", False)
        print(f"Critic use_remove_padding={self.use_remove_padding}")

        self.ulysses_sequence_parallel_size = self.config.get("ulysses_sequence_parallel_size", 1)
        self.device_name = get_device_name()

    def _forward_micro_batch(self, micro_batch):
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch:
            for key in micro_batch["multi_modal_inputs"][0].keys():
                multi_modal_inputs[key] = torch.cat([inputs[key] for inputs in micro_batch["multi_modal_inputs"]], dim=0)

        with torch.autocast(device_type=self.device_name, dtype=torch.bfloat16):
            input_ids = micro_batch["input_ids"]
            batch, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)

            if self.use_remove_padding:
                input_ids_rmpad, indices, *_ = unpad_input(input_ids.unsqueeze(-1), attention_mask)  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices).transpose(0, 1).unsqueeze(1)  # (3, bsz, seqlen) -> (3, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices).transpose(0, 1)

                # pad and slice the inputs if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(input_ids_rmpad, position_ids_rmpad, sp_size=self.ulysses_sequence_parallel_size)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                output = self.critic_module(
                    input_ids=input_ids_rmpad,
                    attention_mask=None,
                    position_ids=position_ids_rmpad,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                values_rmpad = output.logits
                values_rmpad = values_rmpad.squeeze(0)  # (total_nnz)

                # gather output if sp > 1
                if self.ulysses_sequence_parallel_size > 1:
                    values_rmpad = gather_outpus_and_unpad(values_rmpad, gather_dim=0, unpad_dim=0, padding_size=pad_size)

                # pad it back
                values = pad_input(values_rmpad, indices=indices, batch=batch, seqlen=seqlen).squeeze(-1)
                # for debugging add breakpoint
                # import pdb; pdb.set_trace()
                # values = values[:, -response_length - 1 : -1]
            else:
                output = self.critic_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                )  # prevent model thinks we are generating
                values = output.logits
            # for debugging add breakpoint
            # import pdb; pdb.set_trace()
            # values shape (batch, seq_len) if one critic, (batch, seq_len, 2/3) if multi-head critics
            if values.dim() == 2:
                values = values[:, -response_length - 1 : -1].squeeze(-1)
                return values
            elif values.dim() == 3:
                values = values[:, -response_length - 1 : -1, :]
                if values.size(-1) == 2:
                    values_low = values[:, :, 0]
                    values_high = values[:, :, 1]
                    return values_low, values_high
                if values.size(-1) == 3:
                    values_low = values[:, :, 0]
                    values_high = values[:, :, 1]
                    values_term = values[:, :, 2]
                    return values_low, values_high, values_term
                raise ValueError(f"Unexpected values last-dim: {values.size(-1)}")
            else:
                raise ValueError(f"Unexpected values size: {values.size()}")

    def _optimizer_step(self):
        assert self.config.grad_clip is not None

        if isinstance(self.critic_module, FSDP):
            grad_norm = self.critic_module.clip_grad_norm_(self.config.grad_clip)
        elif isinstance(self.critic_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.critic_module.parameters(), max_norm=self.config.grad_clip)

        # if grad_norm is not finite, skip the update
        if not torch.isfinite(grad_norm):
            print(f"WARN: grad_norm is not finite: {grad_norm}")
            self.critic_optimizer.zero_grad()
        else:
            self.critic_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def compute_values(self, data: DataProto) -> torch.Tensor:
        self.critic_module.eval()
        micro_batch_size = data.meta_info["micro_batch_size"]
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        batch = data.select(batch_keys=select_keys).batch
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        use_three_value_heads = bool(self.config.get("use_three_heads_critic", False))
        use_two_value_heads = bool(self.config.get("use_two_heads_critic", False)) and not use_three_value_heads

        if has_multi_modal_inputs:
            num_micro_batches = data.batch.batch_size[0] // micro_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
        elif use_dynamic_bsz:
            # split using dynamic bsz
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, indices = rearrange_micro_batches(batch=batch, max_token_len=max_token_len)
        else:
            micro_batches = batch.split(micro_batch_size)

        values_lst = []
        values_high_lst = [] if use_two_value_heads or use_three_value_heads else None
        values_term_lst = [] if use_three_value_heads else None
        for micro_batch in micro_batches:
            if isinstance(micro_batch, DataProto):
                micro_batch = {**micro_batch.batch, **micro_batch.non_tensor_batch}

            with torch.no_grad():
                values = self._forward_micro_batch(micro_batch)
            if use_two_value_heads:
                values_low, values_high = values
                values_lst.append(values_low)
                if values_high_lst is None:
                    raise RuntimeError("values_high_lst is None but use_two_value_heads is True")
                values_high_lst.append(values_high)
            elif use_three_value_heads:
                values_low, values_high, values_term = values
                values_lst.append(values_low)
                if values_high_lst is None or values_term_lst is None:
                    raise RuntimeError("values_high_lst/values_term_lst is None but use_three_value_heads is True")
                values_high_lst.append(values_high)
                values_term_lst.append(values_term)
            else:
                values_lst.append(values)
        # for debugging add breakpoint
        # import pdb; pdb.set_trace()
        values = torch.concat(values_lst, dim=0)
        values_high = torch.concat(values_high_lst, dim=0) if (use_two_value_heads or use_three_value_heads) else None
        values_term = torch.concat(values_term_lst, dim=0) if use_three_value_heads else None
        responses = data.batch["responses"]
        attention_mask = data.batch["attention_mask"]
        response_length = responses.size(1)

        if use_dynamic_bsz:
            indices = list(itertools.chain.from_iterable(indices))
            assert len(indices) == values.size(0), f"{len(indices)} vs. {values.size()}"
            revert_indices = torch.tensor(get_reverse_idx(indices), dtype=torch.long)
            values = values[revert_indices]
            if use_two_value_heads or use_three_value_heads:
                values_high = values_high[revert_indices]
            if use_three_value_heads:
                values_term = values_term[revert_indices]
        values = values * attention_mask[:, -response_length - 1 : -1]
        if use_two_value_heads or use_three_value_heads:
            values_high = values_high * attention_mask[:, -response_length - 1 : -1]
            if use_three_value_heads:
                values_term = values_term * attention_mask[:, -response_length - 1 : -1]
                return values, values_high, values_term
            return values, values_high
        return values

    @GPUMemoryLogger(role="dp critic", logger=logger)
    def update_critic(self, data: DataProto):
        # make sure we are in training mode
        self.critic_module.train()
        metrics = {}
        use_three_value_heads = bool(self.config.get("use_three_heads_critic", False))
        use_two_value_heads = bool(self.config.get("use_two_heads_critic", False)) and not use_three_value_heads
        high_value_coef = self.config.get("high_value_coef", 1.0) if (use_two_value_heads or use_three_value_heads) else 0.0
        term_value_coef = self.config.get("term_value_coef", 1.0) if use_three_value_heads else 0.0
        if use_three_value_heads:
            select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "value_low", "value_high", "value_term", "returns_low", "returns_high", "returns_term"]
        elif use_two_value_heads:
            select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "value_low", "value_high", "returns_low", "returns_high"]
        else:
            select_keys = ["input_ids", "responses", "attention_mask", "position_ids", "values", "returns"]
        avail_keys = set(data.batch.keys()).union(set(data.non_tensor_batch.keys())) if isinstance(data, DataProto) else set()
        if use_two_value_heads and "value_high" in avail_keys:
            select_keys.append("value_high")
        if use_three_value_heads and "value_term" in avail_keys:
            select_keys.append("value_term")
        if "returns_high" in avail_keys:
            select_keys.append("returns_high")
        if "returns_low" in avail_keys:
            select_keys.append("returns_low")
        if "returns_term" in avail_keys:
            select_keys.append("returns_term")
        if "value_mask_low" in avail_keys:
            select_keys.append("value_mask_low")
        if "value_mask_high" in avail_keys:
            select_keys.append("value_mask_high")
        if "value_mask_term" in avail_keys:
            select_keys.append("value_mask_term")
        batch = data.select(batch_keys=select_keys).batch
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        if has_multi_modal_inputs:
            num_mini_batches = data.batch.batch_size[0] // self.config.ppo_mini_batch_size
            non_tensor_select_keys = ["multi_modal_inputs"]
            dataloader = data.select(select_keys, non_tensor_select_keys).chunk(num_mini_batches)
        else:
            dataloader = batch.split(self.config.ppo_mini_batch_size)

        for epoch in range(self.config.ppo_epochs):
            for batch_idx, data in enumerate(dataloader):
                # split batch into micro_batches
                mini_batch = data
                if has_multi_modal_inputs:
                    num_micro_batches = mini_batch.batch.batch_size[0] // self.config.ppo_micro_batch_size_per_gpu
                    micro_batches = data.select(select_keys, non_tensor_select_keys).chunk(num_micro_batches)
                elif self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = rearrange_micro_batches(batch=mini_batch, max_token_len=max_token_len)
                else:
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)
                    self.gradient_accumulation = self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu

                self.critic_optimizer.zero_grad()

                for data in micro_batches:
                    # Support all devices
                    if isinstance(data, DataProto):
                        data = {**data.batch.to(get_torch_device().current_device()), **data.non_tensor_batch}
                    else:
                        data = data.to(get_torch_device().current_device())  # critic device is cpu when using offload
                    responses = data["responses"]
                    attention_mask = data["attention_mask"]
                    if not use_two_value_heads and not use_three_value_heads:
                        values = data["values"]
                        returns = data["returns"]
                    response_length = responses.size(1)

                    response_mask = attention_mask[:, -response_length - 1 : -1]

                    vpreds = self._forward_micro_batch(data)
                    # import pdb; pdb.set_trace()

                    if use_two_value_heads:
                        vpred_low, vpred_high = vpreds
                        vpred_term = None
                    elif use_three_value_heads:
                        vpred_low, vpred_high, vpred_term = vpreds
                    else:
                        vpred_low = vpreds
                        vpred_high = None
                        vpred_term = None

                    # assert not torch.any(torch.isnan(vpreds)).item()
                    mask_low = response_mask
                    if "value_mask_low" in data:
                        mask_low = (data['value_mask_low'].bool()) & (response_mask.bool())

                    mask_high = None
                    has_high_targets = ("returns_high" in data) and ("value_mask_high" in data)
                    if has_high_targets:
                        mask_high = (data['value_mask_high'].bool()) & (response_mask.bool())
                    mask_term = None
                    has_term_targets = use_three_value_heads and ("returns_term" in data) and ("value_mask_term" in data)
                    if has_term_targets:
                        mask_term = (data["value_mask_term"].bool()) & (response_mask.bool())
                    if use_two_value_heads or use_three_value_heads:
                        values_low_old = data["value_low"]
                        returns_low = data["returns_low"]
                    else:
                        values_low_old = values
                        returns_low = returns

                    if mask_low.any():
                        vf_loss_low, vf_clipfrac_low = core_algos.compute_value_loss(
                            vpreds=vpred_low,
                            values=values_low_old,
                            returns=returns_low,
                            response_mask=mask_low,
                            cliprange_value=self.config.cliprange_value,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                    else:
                        vf_loss_low = torch.tensor(0.0, device=vpred_low.device)
                        vf_clipfrac_low = torch.tensor(0.0, device=vpred_low.device)
                    total_vf = vf_loss_low
                    # pdb.set_trace()
                    vf_high = None
                    vf_clipfrac_high = None
                    if has_high_targets and mask_high is not None and mask_high.any():
                        returns_high = data["returns_high"]
                        if (use_two_value_heads or use_three_value_heads) and ("value_high" in data):
                            values_high_old = data["value_high"]
                        else:
                            values_high_old = data.get("value_high", values_low_old)
                        vpred_for_high = vpred_high if (use_two_value_heads or use_three_value_heads) and vpred_high is not None else vpred_low
                        
                        vf_high, vf_clipfrac_high = core_algos.compute_value_loss(
                        vpreds=vpred_for_high,
                        values=values_high_old,
                        returns=returns_high,
                        response_mask=mask_high,
                        cliprange_value=self.config.cliprange_value,
                        loss_agg_mode=self.config.loss_agg_mode,
                    )
                        total_vf = total_vf + high_value_coef * vf_high
                    vf_term = None
                    vf_clipfrac_term = None
                    if has_term_targets and mask_term is not None and mask_term.any():
                        returns_term = data["returns_term"]
                        if use_three_value_heads and ("value_term" in data):
                            values_term_old = data["value_term"]
                        else:
                            values_term_old = data.get("value_term", values_low_old)
                        vpred_for_term = vpred_term if (use_three_value_heads and vpred_term is not None) else vpred_low
                        vf_term, vf_clipfrac_term = core_algos.compute_value_loss(
                            vpreds=vpred_for_term,
                            values=values_term_old,
                            returns=returns_term,
                            response_mask=mask_term,
                            cliprange_value=self.config.cliprange_value,
                            loss_agg_mode=self.config.loss_agg_mode,
                        )
                        total_vf = total_vf + term_value_coef * vf_term
                    # pdb.set_trace()
                    # vf_loss, vf_clipfrac = core_algos.compute_value_loss(
                    #     vpreds=vpreds,
                    #     values=values,
                    #     returns=returns,
                    #     response_mask=response_mask,
                    #     cliprange_value=self.config.cliprange_value,
                    #     loss_agg_mode=self.config.loss_agg_mode,
                    # )
                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = total_vf * (len(data) / self.config.ppo_mini_batch_size)
                    else:
                        loss = total_vf / self.gradient_accumulation
                    # pdb.set_trace()
                    loss.backward()

                    log = {
                    "critic/vf_loss": total_vf.detach().item(),
                    "critic/vf_loss_low": vf_loss_low.detach().item(),
                    "critic/vf_clipfrac_low": vf_clipfrac_low.detach().item(),
                    "critic/vpred_mean_low": (masked_mean(vpred_low, mask_low).detach().item() if mask_low.any() else 0.0),
                    "critic/mask_tokens_low": float(mask_low.sum().detach().item()),
                    "critic/use_two_heads_critic": float(use_two_value_heads),
                    "critic/use_three_heads_critic": float(use_three_value_heads),
                    }

                    if vf_high is not None:
                        vpred_for_high = vpred_high if ((use_two_value_heads or use_three_value_heads) and vpred_high is not None) else vpred_low
                        log.update(
                        {
                            "critic/vf_loss_high": vf_high.detach().item(),
                            "critic/vf_clipfrac_high": vf_clipfrac_high.detach().item() if vf_clipfrac_high is not None else 0.0,
                            "critic/vpred_mean_high": (masked_mean(vpred_for_high, mask_high).detach().item() if mask_high.any() else 0.0),
                            "critic/mask_tokens_high": float(mask_high.sum().detach().item()),
                            "critic/high_vf_coef": high_value_coef,
                        }
                        )
                    else:
                        log.update(
                            {
                                "critic/vf_loss_high": 0.0,
                                "critic/vf_clipfrac_high": 0.0,
                                "critic/vpred_mean_high": 0.0,
                                "critic/mask_tokens_high": 0.0,
                                "critic/high_vf_coef": high_value_coef,
                            }
                        )
                    if vf_term is not None:
                        vpred_for_term = vpred_term if (use_three_value_heads and vpred_term is not None) else vpred_low
                        log.update(
                            {
                                "critic/vf_loss_term": vf_term.detach().item(),
                                "critic/vf_clipfrac_term": vf_clipfrac_term.detach().item() if vf_clipfrac_term is not None else 0.0,
                                "critic/vpred_mean_term": (masked_mean(vpred_for_term, mask_term).detach().item() if mask_term.any() else 0.0),
                                "critic/mask_tokens_term": float(mask_term.sum().detach().item()),
                                "critic/term_vf_coef": term_value_coef,
                            }
                        )
                    else:
                        log.update(
                            {
                                "critic/vf_loss_term": 0.0,
                                "critic/vf_clipfrac_term": 0.0,
                                "critic/vpred_mean_term": 0.0,
                                "critic/mask_tokens_term": 0.0,
                                "critic/term_vf_coef": term_value_coef,
                            }
                        )

                    append_to_dict(metrics, log)

                grad_norm = self._optimizer_step()
                append_to_dict(
                    metrics,
                    {
                        "critic/grad_norm": grad_norm.detach().item(),
                    },
                )
        self.critic_optimizer.zero_grad()
        return metrics
