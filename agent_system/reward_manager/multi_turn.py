from verl import DataProto
import torch
import numpy as np

class MultiTurnRewardManager:
    """
    Reward manager for multi-turn training.
    """
    def __init__(self, tokenizer, num_examine, normalize_by_length=False) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.normalize_by_length = normalize_by_length

    def __call__(self, data: DataProto, return_dict=False):
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"], 'reward_extra_info': {}}
            else:
                return data.batch["rm_scores"]
        
        reward_tensor = torch.zeros_like(data.batch['responses'], dtype=torch.float32)

        for i in range(len(data)):
            item = data[i]  # DataProtoItem

            prompt_ids = item.batch['prompts']
            prompt_len = int(prompt_ids.shape[-1])

            attn = item.batch['attention_mask']
            # valid prompt length
            valid_prompt_len = int(attn[:prompt_len].sum().item())
            valid_resp_len = int(attn[prompt_len:].sum().item())


            # skip degenerate responses
            if valid_resp_len <= 0:
                continue
            
            valid_prompt_ids = prompt_ids[-valid_prompt_len:] if valid_prompt_len > 0 else prompt_ids[:0]
            resp_ids = item.batch["responses"]
            valid_resp_ids = resp_ids[:valid_resp_len]
            
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=False)
            resp_str = self.tokenizer.decode(valid_resp_ids, skip_special_tokens=False)

            r_t = item.non_tensor_batch.get('rewards', 0.0)
            done_t = item.non_tensor_batch.get('dones', False)

            score = r_t

            # if normalize by length
            if self.normalize_by_length and valid_resp_len > 0:
                score = score / valid_resp_len

            reward_tensor[i, valid_resp_len - 1] = torch.tensor(score, dtype=torch.float32)

        if return_dict:
            return {
                "reward_tensor": reward_tensor, 
                'reward_extra_info': {}
            }
        return reward_tensor