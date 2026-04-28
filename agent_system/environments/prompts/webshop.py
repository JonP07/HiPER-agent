# Copyright 2025 Nanyang Technological University (NTU), Singapore
# and the verl-agent (GiGPO) team.
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

# --------------------- WebShop --------------------- #
WEBSHOP_TEMPLATE_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. 
Your task is to: {task_description}.
Your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment.
Your task is to: {task_description}.
Prior to this step, you have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.
Your admissible actions of the current situation are: 
[
{available_actions}
].

Now it's your turn to take one action for the current step.
You should first reason step-by-step about the current situation, then think carefully which admissible action best advances the shopping goal. This reasoning process MUST be enclosed within <think> </think> tags. 
Once you've finished your reasoning, you should choose an admissible action for current step and present it within <action> </action> tags.
"""

WEBSHOP_TEMPLATE_OPTIONS_NO_HIS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. Your task is to: {task_description}.

You will complete the task by maintaining a SHORT-TERM SUB-GOAL at each step. A sub-goal is a small high-level objective that can typically be achieved in a few actions. A sub-goal is NOT the full task and NOT a low-level action.

At every step, you reconsider your current sub-goal based on the latest observation, and may continue it or switch to a new short-term sub-goal.

Your current observation is: {current_observation}.

Your current sub-goal is: {current_subgoal}.

Your admissible actions of the current situation are: 
[
{available_actions}
].

At EVERY step, you MUST output EXACTLY THREE blocks, in the order shown below:
1) A <switch> block          (KEEP or SWITCH)
2) A <subgoal> block         (the sub-goal to follow next)
3) An <action> block         (one admissible action)

NO other text, comments, or reasoning is allowed before, after, or between these blocks.

STRICT FORMAT REQUIREMENTS:
- <switch> MUST contain ONLY "KEEP" or "SWITCH".
- <subgoal> MUST appear at EVERY step:
    * If <switch>KEEP</switch>, you MUST copy the EXACT current sub-goal.
    * If <switch>SWITCH</switch>, you MUST write a NEW short sub-goal achievable in a few actions.
- <action> MUST contain EXACTLY ONE action verbatim copied AS IS from the admissible actions list.

NOW RESPOND IN THIS EXACT FORMAT (no extra text):

<switch>KEEP or SWITCH</switch>
<subgoal>the sub-goal you will follow next</subgoal>
<action>EXACTLY one ADMISSIBLE action</action>
"""

WEBSHOP_TEMPLATE_OPTIONS = """
You are an expert autonomous agent operating in the WebShop e‑commerce environment. Your task is to: {task_description}.

You will complete the task by maintaining a SHORT-TERM SUB-GOAL at each step. A sub-goal is a small high-level objective that can typically be achieved in a few actions. A sub-goal is NOT the full task and NOT a low-level action.

At every step, you reconsider your current sub-goal based on the latest observation, and may continue it or switch to a new short-term sub-goal.

You have already taken {step_count} step(s). Below are the most recent {history_length} observations and the corresponding actions you took: {action_history}
You are now at step {current_step} and your current observation is: {current_observation}.

Your current sub-goal is: {current_subgoal}.

Your admissible actions of the current situation are: 
[
{available_actions}
].

At EVERY step, you MUST output EXACTLY THREE blocks, in the order shown below:
1) A <switch> block          (KEEP or SWITCH)
2) A <subgoal> block         (the sub-goal to follow next)
3) An <action> block         (one admissible action)

NO other text, comments, or reasoning is permitted anywhere in the output.

STRICT FORMAT REQUIREMENTS:
- <switch> MUST contain ONLY "KEEP" or "SWITCH".
- <subgoal> MUST appear at EVERY step:
    * If you KEEP, copy the EXACT current sub-goal into <subgoal>.
    * If you SWITCH, write a NEW short sub-goal achievable in a few actions and NOT the entire task.
- <action> MUST contain EXACTLY ONE action verbatim copied AS IS from the admissible actions list.

NOW RESPOND IN THIS EXACT FORMAT (no extra text):

<switch>KEEP or SWITCH</switch>
<subgoal>the sub-goal you will follow next</subgoal>
<action>EXACTLY one ADMISSIBLE action</action>
"""