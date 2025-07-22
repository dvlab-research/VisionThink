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
FSDP PPO Trainer with Ray-based single controller.
This trainer supports model-agonistic model initialization with huggingface
"""

import os
import re
import uuid
import wandb
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from pprint import pprint
from typing import Type, Dict
from copy import deepcopy

import numpy as np
from codetiming import Timer
from omegaconf import OmegaConf, open_dict
from verl import DataProto
from verl.protocol import pad_dataproto_to_divisor, unpad_dataproto, DataProtoItem
from verl.single_controller.base import Worker
from verl.single_controller.ray import RayResourcePool, RayWorkerGroup, RayClassWithInitArgs
from verl.single_controller.ray.base import create_colocated_worker_cls
from verl.trainer.ppo import core_algos
from verl.trainer.constants import SYSTEM_PROMPT_MAP
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions, log_seqlen_unbalance
from verl.utils.checkpoint.checkpoint_manager import find_latest_ckpt_path
from verl.utils.dataset.rl_dataset import RLHFDataset, collate_fn
from verl.utils.dataset.multimodal_dataset import MultiModalDataset
from torch.utils.data import RandomSampler, SequentialSampler
from torchdata.stateful_dataloader import StatefulDataLoader

WorkerType = Type[Worker]

def dataprotoitem_to_dataproto(item: DataProtoItem) -> DataProto:
    """Convert a DataProtoItem to a DataProto object"""
    return DataProto.from_dict(
        tensors=item.batch,  # TensorDict is already in correct format
        non_tensors=item.non_tensor_batch,  # Dict is already in correct format 
        meta_info=item.meta_info
    )

class Role(Enum):
    """
    To create more roles dynamically, you can subclass Role and add new members
    """
    Actor = 0
    Rollout = 1
    ActorRollout = 2
    Critic = 3
    RefPolicy = 4
    RewardModel = 5
    ActorRolloutRef = 6


class AdvantageEstimator(str, Enum):
    """
    Using an enumeration class to avoid spelling errors in adv_estimator
    """
    GAE = 'gae'
    GRPO = 'grpo'
    REINFORCE_PLUS_PLUS = 'reinforce_plus_plus'
    REMAX = 'remax'
    RLOO = 'rloo'


@dataclass
class ResourcePoolManager:
    """
    Define a resource pool specification. Resource pool will be initialized first.
    Mapping
    """
    resource_pool_spec: dict[str, list[int]]
    mapping: dict[Role, str]
    resource_pool_dict: dict[str, RayResourcePool] = field(default_factory=dict)

    def create_resource_pool(self):
        for resource_pool_name, process_on_nodes in self.resource_pool_spec.items():
            # max_colocate_count means the number of WorkerGroups (i.e. processes) in each RayResourcePool
            # For FSDP backend, we recommend using max_colocate_count=1 that merge all WorkerGroups into one.
            # For Megatron backend, we recommend using max_colocate_count>1 that can utilize different WorkerGroup for differnt models
            resource_pool = RayResourcePool(process_on_nodes=process_on_nodes,
                                            use_gpu=True,
                                            max_colocate_count=1,
                                            name_prefix=resource_pool_name)
            self.resource_pool_dict[resource_pool_name] = resource_pool

    def get_resource_pool(self, role: Role) -> RayResourcePool:
        """Get the resource pool of the worker_cls"""
        return self.resource_pool_dict[self.mapping[role]]


import torch
from verl.utils.torch_functional import masked_mean


def apply_kl_penalty(data: DataProto, kl_ctrl: core_algos.AdaptiveKLController, kl_penalty='kl'):
    responses = data.batch['responses']
    response_length = responses.size(1)
    token_level_scores = data.batch['token_level_scores']
    batch_size = data.batch.batch_size[0]
    attention_mask = data.batch['attention_mask']
    response_mask = attention_mask[:, -response_length:]

    # compute kl between ref_policy and current policy
    if 'ref_log_prob' in data.batch.keys():
        kld = core_algos.kl_penalty(data.batch['old_log_probs'], data.batch['ref_log_prob'],
                                    kl_penalty=kl_penalty)  # (batch_size, response_length)
        kld = kld * response_mask
        beta = kl_ctrl.value
    else:
        beta = 0
        kld = torch.zeros_like(response_mask, dtype=torch.float32)

    token_level_rewards = token_level_scores - beta * kld

    current_kl = masked_mean(kld, mask=response_mask, axis=-1)  # average over sequence
    current_kl = torch.mean(current_kl, dim=0).item()

    # according to https://github.com/huggingface/trl/blob/951ca1841f29114b969b57b26c7d3e80a39f75a0/trl/trainer/ppo_trainer.py#L837
    kl_ctrl.update(current_kl=current_kl, n_steps=batch_size)
    data.batch['token_level_rewards'] = token_level_rewards

    metrics = {'critic/kl': current_kl, 'critic/kl_coeff': beta}

    return data, metrics


def compute_advantage(data: DataProto, adv_estimator, gamma=1.0, lam=1.0, num_repeat=1):
    # prepare response group
    # TODO: add other ways to estimate advantages
    if adv_estimator == AdvantageEstimator.GAE:
        values = data.batch['values']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        token_level_rewards = data.batch['token_level_rewards']
        advantages, returns = core_algos.compute_gae_advantage_return(token_level_rewards=token_level_rewards,
                                                                      values=values,
                                                                      eos_mask=response_mask,
                                                                      gamma=gamma,
                                                                      lam=lam)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.GRPO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_grpo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REINFORCE_PLUS_PLUS:
        token_level_rewards = data.batch['token_level_rewards']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_reinforce_plus_plus_outcome_advantage(
            token_level_rewards=token_level_rewards, eos_mask=response_mask, gamma=gamma)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.REMAX:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]

        reward_baselines = data.batch['reward_baselines']

        advantages, returns = core_algos.compute_remax_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                         reward_baselines=reward_baselines,
                                                                         eos_mask=response_mask)

        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    elif adv_estimator == AdvantageEstimator.RLOO:
        token_level_rewards = data.batch['token_level_rewards']
        index = data.non_tensor_batch['uid']
        responses = data.batch['responses']
        response_length = responses.size(-1)
        attention_mask = data.batch['attention_mask']
        response_mask = attention_mask[:, -response_length:]
        advantages, returns = core_algos.compute_rloo_outcome_advantage(token_level_rewards=token_level_rewards,
                                                                        eos_mask=response_mask,
                                                                        index=index)
        data.batch['advantages'] = advantages
        data.batch['returns'] = returns
    else:
        raise NotImplementedError
    return data


def reduce_metrics(metrics: dict):
    for key, val in metrics.items():
        metrics[key] = np.mean(val)
    return metrics


def _compute_response_info(batch):
    response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-response_length]
    response_mask = batch.batch['attention_mask'][:, -response_length:]

    prompt_length = prompt_mask.sum(-1).float()
    response_length = response_mask.sum(-1).float()  # (batch_size,)

    return dict(
        response_mask=response_mask,
        prompt_length=prompt_length,
        response_length=response_length,
    )


def compute_data_metrics(batch, use_critic=True):
    # TODO: add response length
    sequence_score = batch.batch['token_level_scores'].sum(-1)
    sequence_reward = batch.batch['token_level_rewards'].sum(-1)

    if 'format_scores' in batch.batch:
        format_reward = batch.batch['format_scores']
        accuracy_reward = batch.batch['acc_scores']
    else:
        format_reward = torch.tensor(0.0)
        accuracy_reward = torch.tensor(0.0)

    advantages = batch.batch['advantages']
    returns = batch.batch['returns']

    max_response_length = batch.batch['responses'].shape[-1]

    prompt_mask = batch.batch['attention_mask'][:, :-max_response_length].bool()
    response_mask = batch.batch['attention_mask'][:, -max_response_length:].bool()

    max_prompt_length = prompt_mask.size(-1)

    response_info = _compute_response_info(batch)
    prompt_length = response_info['prompt_length']
    response_length = response_info['response_length']
    # update response_length for multi_turn_tool_call
    if 'multi_turn_response_mask' in batch.batch:
        response_length = batch.batch['multi_turn_response_mask'].sum(dim=1).float()

    valid_adv = torch.masked_select(advantages, response_mask)
    valid_returns = torch.masked_select(returns, response_mask)

    if use_critic:
        values = batch.batch['values']
        valid_values = torch.masked_select(values, response_mask)
        return_diff_var = torch.var(valid_returns - valid_values)
        return_var = torch.var(valid_returns)

    metrics = {
        # score
        'critic/score/mean':
            torch.mean(sequence_score).detach().item(),
        'critic/score/max':
            torch.max(sequence_score).detach().item(),
        'critic/score/min':
            torch.min(sequence_score).detach().item(),
        # reward
        'critic/rewards/mean/format':
            torch.mean(format_reward).detach().item(),
        'critic/rewards/mean/accuracy':
            torch.mean(accuracy_reward).detach().item(),
        'critic/rewards/mean':
            torch.mean(sequence_reward).detach().item(),
        'critic/rewards/var':
            torch.var(sequence_reward).detach().item(),
        'critic/rewards/max':
            torch.max(sequence_reward).detach().item(),
        'critic/rewards/min':
            torch.min(sequence_reward).detach().item(),
        # adv
        'critic/advantages/mean':
            torch.mean(valid_adv).detach().item(),
        'critic/advantages/max':
            torch.max(valid_adv).detach().item(),
        'critic/advantages/min':
            torch.min(valid_adv).detach().item(),
        # returns
        'critic/returns/mean':
            torch.mean(valid_returns).detach().item(),
        'critic/returns/max':
            torch.max(valid_returns).detach().item(),
        'critic/returns/min':
            torch.min(valid_returns).detach().item(),
        **({
            # values
            'critic/values/mean': torch.mean(valid_values).detach().item(),
            'critic/values/max': torch.max(valid_values).detach().item(),
            'critic/values/min': torch.min(valid_values).detach().item(),
            # vf explained var
            'critic/vf_explained_var': (1.0 - return_diff_var / (return_var + 1e-5)).detach().item(),
        } if use_critic else {}),

        # response length
        'response_length/mean':
            torch.mean(response_length).detach().item(),
        'response_length/max':
            torch.max(response_length).detach().item(),
        'response_length/min':
            torch.min(response_length).detach().item(),
        'response_length/clip_ratio':
            torch.mean(torch.eq(response_length, max_response_length).float()).detach().item(),
        # prompt length
        'prompt_length/mean':
            torch.mean(prompt_length).detach().item(),
        'prompt_length/max':
            torch.max(prompt_length).detach().item(),
        'prompt_length/min':
            torch.min(prompt_length).detach().item(),
        'prompt_length/clip_ratio':
            torch.mean(torch.eq(prompt_length, max_prompt_length).float()).detach().item(),
    }
    return metrics


def compute_timing_metrics(batch, timing_raw):
    response_info = _compute_response_info(batch)
    num_prompt_tokens = torch.sum(response_info['prompt_length']).item()
    num_response_tokens = torch.sum(response_info['response_length']).item()
    num_overall_tokens = num_prompt_tokens + num_response_tokens

    num_tokens_of_section = {
        'gen': num_response_tokens,
        **{
            name: num_overall_tokens for name in ['ref', 'values', 'adv', 'update_critic', 'update_actor']
        },
    }

    return {
        **{
            f'timing_s/{name}': value for name, value in timing_raw.items()
        },
        **{
            f'timing_per_token_ms/{name}': timing_raw[name] * 1000 / num_tokens_of_section[name] for name in set(num_tokens_of_section.keys(
            )) & set(timing_raw.keys())
        },
    }


@contextmanager
def _timer(name: str, timing_raw: Dict[str, float]):
    with Timer(name=name, logger=None) as timer:
        yield
    timing_raw[name] = timer.last


class RayPPOTrainer(object):
    """
    Note that this trainer runs on the driver process on a single CPU/GPU node.
    """

    # TODO: support each role have individual ray_worker_group_cls,
    # i.e., support different backend of different role
    def __init__(self,
                 config,
                 tokenizer,
                 role_worker_mapping: dict[Role, WorkerType],
                 resource_pool_manager: ResourcePoolManager,
                 ray_worker_group_cls: RayWorkerGroup = RayWorkerGroup,
                 processor=None,
                 reward_fn=None,
                 val_reward_fn=None):

        # assert torch.cuda.is_available(), 'cuda must be available on driver'

        self.tokenizer = tokenizer
        self.processor = processor
        self.config = config
        self.reward_fn = reward_fn
        self.val_reward_fn = val_reward_fn
        self.tool_call = self.config.data.tool_call

        self.hybrid_engine = config.actor_rollout_ref.hybrid_engine
        assert self.hybrid_engine, 'Currently, only support hybrid engine'

        if self.hybrid_engine:
            assert Role.ActorRollout in role_worker_mapping, f'{role_worker_mapping.keys()=}'

        self.role_worker_mapping = role_worker_mapping
        self.resource_pool_manager = resource_pool_manager
        self.use_reference_policy = Role.RefPolicy in role_worker_mapping
        self.use_rm = Role.RewardModel in role_worker_mapping
        self.ray_worker_group_cls = ray_worker_group_cls

        # define KL control
        if self.use_reference_policy:
            if config.algorithm.kl_ctrl.type == 'fixed':
                self.kl_ctrl = core_algos.FixedKLController(kl_coef=config.algorithm.kl_ctrl.kl_coef)
            elif config.algorithm.kl_ctrl.type == 'adaptive':
                assert config.algorithm.kl_ctrl.horizon > 0, f'horizon must be larger than 0. Got {config.critic.kl_ctrl.horizon}'
                self.kl_ctrl = core_algos.AdaptiveKLController(init_kl_coef=config.algorithm.kl_ctrl.kl_coef,
                                                               target_kl=config.algorithm.kl_ctrl.target_kl,
                                                               horizon=config.algorithm.kl_ctrl.horizon)
            else:
                raise NotImplementedError
        else:
            self.kl_ctrl = core_algos.FixedKLController(kl_coef=0.)

        if self.config.algorithm.adv_estimator == AdvantageEstimator.GAE:
            self.use_critic = True
        elif self.config.algorithm.adv_estimator in [
                AdvantageEstimator.GRPO, AdvantageEstimator.REINFORCE_PLUS_PLUS, AdvantageEstimator.REMAX,
                AdvantageEstimator.RLOO
        ]:
            self.use_critic = False
        else:
            raise NotImplementedError

        self._validate_config()
        self._create_dataloader()

    def _validate_config(self):
        config = self.config
        # number of GPUs total
        n_gpus = config.trainer.n_gpus_per_node * config.trainer.nnodes

        # 1. Check total batch size for data correctness
        real_train_batch_size = config.data.train_batch_size * config.actor_rollout_ref.rollout.n
        assert real_train_batch_size % n_gpus == 0, \
            f"real_train_batch_size ({real_train_batch_size}) must be divisible by total n_gpus ({n_gpus})."

        # A helper function to check "micro_batch_size" vs "micro_batch_size_per_gpu"
        # We throw an error if the user sets both. The new convention is "..._micro_batch_size_per_gpu".
        def check_mutually_exclusive(mbs, mbs_per_gpu, name: str):
            if mbs is None and mbs_per_gpu is None:
                raise ValueError(f"[{name}] Please set at least one of '{name}.micro_batch_size' or "
                                 f"'{name}.micro_batch_size_per_gpu'.")

            if mbs is not None and mbs_per_gpu is not None:
                raise ValueError(f"[{name}] You have set both '{name}.micro_batch_size' AND "
                                 f"'{name}.micro_batch_size_per_gpu'. Please remove '{name}.micro_batch_size' "
                                 f"because only '*_micro_batch_size_per_gpu' is supported (the former is deprecated).")

        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            # actor: ppo_micro_batch_size vs. ppo_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.actor.ppo_micro_batch_size,
                                     config.actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.actor")

            # reference: log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.ref.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.ref")

            #  The rollout section also has log_prob_micro_batch_size vs. log_prob_micro_batch_size_per_gpu
            check_mutually_exclusive(config.actor_rollout_ref.rollout.log_prob_micro_batch_size,
                                     config.actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu,
                                     "actor_rollout_ref.rollout")

        if self.use_critic and not config.critic.use_dynamic_bsz:
            # Check for critic micro-batch size conflicts
            check_mutually_exclusive(config.critic.ppo_micro_batch_size, config.critic.ppo_micro_batch_size_per_gpu,
                                     "critic")

        # Check for reward model micro-batch size conflicts
        if config.reward_model.enable and not config.reward_model.use_dynamic_bsz:
            check_mutually_exclusive(config.reward_model.micro_batch_size, config.reward_model.micro_batch_size_per_gpu,
                                     "reward_model")

        # Actor
        # if NOT dynamic_bsz, we must ensure:
        #    ppo_mini_batch_size is divisible by ppo_micro_batch_size
        #    ppo_micro_batch_size * sequence_parallel_size >= n_gpus
        if not config.actor_rollout_ref.actor.use_dynamic_bsz:
            sp_size = config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1)
            if config.actor_rollout_ref.actor.ppo_micro_batch_size is not None:
                assert config.actor_rollout_ref.actor.ppo_mini_batch_size % config.actor_rollout_ref.actor.ppo_micro_batch_size == 0
                assert config.actor_rollout_ref.actor.ppo_micro_batch_size * sp_size >= n_gpus

        # critic
        if self.use_critic and not config.critic.use_dynamic_bsz:
            sp_size = config.critic.get('ulysses_sequence_parallel_size', 1)
            if config.critic.ppo_micro_batch_size is not None:
                assert config.critic.ppo_mini_batch_size % config.critic.ppo_micro_batch_size == 0
                assert config.critic.ppo_micro_batch_size * sp_size >= n_gpus

        # Check if use_remove_padding is enabled when using sequence parallelism for fsdp
        if config.actor_rollout_ref.actor.strategy == 'fsdp':
            if config.actor_rollout_ref.actor.get('ulysses_sequence_parallel_size', 1) > 1 or \
                    config.actor_rollout_ref.ref.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.actor_rollout_ref.model.use_remove_padding, \
                    "When using sequence parallelism for actor/ref policy, you must enable `use_remove_padding`."

        if self.use_critic and config.critic.strategy == 'fsdp':
            if config.critic.get('ulysses_sequence_parallel_size', 1) > 1:
                assert config.critic.model.use_remove_padding, \
                    "When using sequence parallelism for critic, you must enable `use_remove_padding`."

        if config.data.get('val_batch_size', None) is not None:
            print(
                f"WARNING: val_batch_size is deprecated. Validation datasets are sent to inference engines as a whole batch, which will schedule the memory themselves."
            )

        print("[validate_config] All configuration checks passed successfully!")

    def _create_dataloader(self):
        # TODO: we have to make sure the batch size is divisible by the dp size
        system_prompt_type = self.config.data.get('system_prompt', False)
        system_prompt = SYSTEM_PROMPT_MAP[system_prompt_type] if system_prompt_type else None
        if isinstance(self.config.data.train_files, str):
            # assert self.config.data.train_files.endswith('.yaml')
            self.train_dataset = MultiModalDataset(
                data_path=self.config.data.train_files,
                tokenizer=self.tokenizer,
                processor=self.processor,
                prompt_key=self.config.data.prompt_key,
                image_key=self.config.data.get('image_key', 'images'),
                max_prompt_length=self.config.data.max_prompt_length,
                filter_prompts=True,
                return_raw_chat=self.config.data.get('return_raw_chat', False),
                truncation='error',
                system_prompt=system_prompt,
                max_pixels=self.config.data.max_pixels,
                min_pixels=self.config.data.min_pixels,
                mask_blank=self.config.data.mask_blank,
                use_3drope=self.config.trainer.use_3drope,
                prompt_type=self.config.data.prompt_type,
                general_qa_reward_fn=self.config.data.get("general_qa_reward_fn", "general_qa_gpt"),
                tool_call=self.config.data.tool_call,
                use_tgt_size=self.config.data.use_tgt_size,
                use_raw_image=self.config.actor_rollout_ref.rollout.use_raw_image,
            )
        else:
            self.train_dataset = RLHFDataset(parquet_files=self.config.data.train_files,
                                            tokenizer=self.tokenizer,
                                            processor=self.processor,
                                            prompt_key=self.config.data.prompt_key,
                                            image_key=self.config.data.get('image_key', 'images'),
                                            max_prompt_length=self.config.data.max_prompt_length,
                                            filter_prompts=True,
                                            return_raw_chat=self.config.data.get('return_raw_chat', False),
                                            truncation='error',
                                            system_prompt=system_prompt,)
        # use sampler for better ckpt resume
        if self.config.data.shuffle:
            train_dataloader_generator = torch.Generator()
            train_dataloader_generator.manual_seed(self.config.data.get('seed', 1))
            sampler = RandomSampler(data_source=self.train_dataset, generator=train_dataloader_generator)
        else:
            sampler = SequentialSampler(data_source=self.train_dataset)

        train_batch_size = self.config.data.train_batch_size
        if self.config.trainer.rejection_sample:
            train_batch_size *= self.config.trainer.rejection_sample_multiplier
            train_batch_size = int(train_batch_size)

        self.train_dataloader = StatefulDataLoader(dataset=self.train_dataset,
                                                   batch_size=train_batch_size,
                                                   num_workers=8,
                                                   drop_last=True,
                                                   collate_fn=collate_fn,
                                                   sampler=sampler)
        assert len(self.train_dataloader) >= 1
        print(f'Size of train dataloader: {len(self.train_dataloader)}')

        if self.config.data.val_files != 'None':
            if isinstance(self.config.data.val_files, str):
                # assert self.config.data.val_files.endswith('.yaml')
                self.val_dataset = MultiModalDataset(
                    data_path=self.config.data.val_files,
                    tokenizer=self.tokenizer,
                    processor=self.processor,
                    prompt_key=self.config.data.prompt_key,
                    image_key=self.config.data.get('image_key', 'images'),
                    max_prompt_length=self.config.data.max_prompt_length,
                    filter_prompts=True,
                    return_raw_chat=self.config.data.get('return_raw_chat', False),
                    truncation='error',
                    system_prompt=system_prompt,
                    max_pixels=self.config.data.max_pixels,
                    min_pixels=self.config.data.min_pixels,
                    mask_blank=self.config.data.mask_blank,
                    use_3drope=self.config.trainer.use_3drope,
                    prompt_type=self.config.data.prompt_type,
                    general_qa_reward_fn=self.config.data.get("general_qa_reward_fn", "v1"),
                    tool_call=self.config.data.tool_call,
                    use_tgt_size=self.config.data.use_tgt_size,
                    use_raw_image=self.config.actor_rollout_ref.rollout.use_raw_image,
                )
            else:
                self.val_dataset = RLHFDataset(parquet_files=self.config.data.val_files,
                                            tokenizer=self.tokenizer,
                                            processor=self.processor,
                                            prompt_key=self.config.data.prompt_key,
                                            image_key=self.config.data.get('image_key', 'images'),
                                            max_prompt_length=self.config.data.max_prompt_length,
                                            filter_prompts=True,
                                            return_raw_chat=self.config.data.get('return_raw_chat', False),
                                            truncation='error',
                                            system_prompt=system_prompt,)
            self.val_dataloader = StatefulDataLoader(
                dataset=self.val_dataset,
                batch_size=512,
                num_workers=8,
                shuffle=False,
                drop_last=True,
                collate_fn=collate_fn)
            

        # inject total_training_steps to actor/critic optim_config. This is hacky.
        total_training_steps = len(self.train_dataloader) * self.config.trainer.total_epochs

        if self.config.trainer.total_training_steps is not None:
            total_training_steps = self.config.trainer.total_training_steps

        self.total_training_steps = total_training_steps
        print(f'Total training steps: {self.total_training_steps}')

        OmegaConf.set_struct(self.config, True)
        with open_dict(self.config):
            self.config.actor_rollout_ref.actor.optim.total_training_steps = total_training_steps
            self.config.critic.optim.total_training_steps = total_training_steps

    def _maybe_log_train_generations_to_wandb(self, batch, reward_tensor):
        """Log a table of train samples to wandb"""
        generations_to_log = self.config.trainer.train_generations_to_log_to_wandb

        
        if generations_to_log == 0:
            return
        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return
        input_ids = batch.batch['prompts'][:generations_to_log]
        response_ids = batch.batch['responses'][:generations_to_log]
        doc_ids = batch.non_tensor_batch['doc_id'][:generations_to_log]
        inputs = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
        outputs = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        ground_truth = batch.non_tensor_batch['ground_truth'].tolist()[:generations_to_log]
        scores = reward_tensor.max(dim=-1)[0][:generations_to_log]

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(doc_ids, inputs, outputs, ground_truth, scores))
        samples.sort(key=lambda x: str(x[0]))  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)
        # Create column names for all samples
        columns = ["step"] + sum([[f"doc_id_{i+1}", f"input_{i+1}", f"output_{i+1}", f"ground_truth_{i+1}", f"score_{i+1}"] for i in range(generations_to_log)], [])
        if not hasattr(self, 'train_table'):
            # Initialize the table on first call
            self.train_table = wandb.Table(columns=columns)

        new_table = wandb.Table(columns=columns, data=self.train_table.data)
        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)
        if generations_to_log > len(samples):
            for _ in range(generations_to_log - len(samples)):
                row_data.extend(['', '', '', torch.tensor(0)])
        new_table.add_data(*row_data)
        # Update reference and log
        wandb.log({"train/generations": new_table}, step=self.global_steps)
        self.train_table = new_table

    def _maybe_log_val_generations_to_wandb(self, doc_ids, inputs, outputs, ground_truth, scores):
        """Log a table of validation samples to wandb"""

        generations_to_log = self.config.trainer.val_generations_to_log_to_wandb

        if generations_to_log == 0:
            return

        if generations_to_log > 0 and 'wandb' not in self.config.trainer.logger:
            print(
                'WARNING: `val_generations_to_log_to_wandb` is set to a positive value, but no wandb logger is found. ')
            return

        # Create tuples of (input, output, score) and sort by input text
        samples = list(zip(doc_ids, inputs, outputs, ground_truth, scores))
        samples.sort(key=lambda x: x[0])  # Sort by input text

        # Use fixed random seed for deterministic shuffling
        rng = np.random.RandomState(42)
        rng.shuffle(samples)

        # Take first N samples after shuffling
        samples = samples[:generations_to_log]

        # Create column names for all samples
        columns = ["step"] + sum([[f"doc_id_{i+1}", f"input_{i+1}", f"output_{i+1}", f"ground_truth_{i+1}", f"score_{i+1}"] for i in range(len(samples))], [])
        if not hasattr(self, 'validation_table'):
            # Initialize the table on first call
            self.validation_table = wandb.Table(columns=columns)

        # Create a new table with same columns and existing data
        # Workaround for https://github.com/wandb/wandb/issues/2981#issuecomment-1997445737
        new_table = wandb.Table(columns=columns, data=self.validation_table.data)

        # Add new row with all data
        row_data = []
        row_data.append(self.global_steps)
        for sample in samples:
            row_data.extend(sample)

        new_table.add_data(*row_data)

        # Update reference and log
        wandb.log({"val/generations": new_table}, step=self.global_steps)
        self.validation_table = new_table

    def _maybe_log_tool_call_metrics(self, batch, metrics):
        response_ids = batch.batch['responses']
        accuracy_scores = batch.batch['acc_scores']
        multi_modal_data = batch.non_tensor_batch['multi_modal_data']
        responses = self.tokenizer.batch_decode(response_ids, skip_special_tokens=True)
        apply_tool_call_list = []
        success_tool_call_list = []
        for response, multi_modal_item in zip(responses, multi_modal_data):
            if self.config.actor_rollout_ref.rollout.name == "vllm_multi_turn_tool_call":
                tool_call_pattern = re.compile(r'<tool_call>(.*?)</tool_call>', re.DOTALL)
                apply_tool_call = len(tool_call_pattern.findall(response)) > 0
            elif self.config.actor_rollout_ref.rollout.name == "vllm_multi_turn_resize_image":
                apply_tool_call = "<resize>" in response
            else:
                raise NotImplementedError
            success_tool_call = len(multi_modal_item['image']) > 1  # Successfully calling RESIZE tool wiil append a new image to 'multi_modal_data'
            apply_tool_call_list.append(apply_tool_call)
            success_tool_call_list.append(success_tool_call)
        use_tool_correct_answer_num = (accuracy_scores[success_tool_call_list] == 1).sum().item()
        if batch.meta_info.get('validate', False):  # Validation
            metrics['val/tool_call/success_tool_call_ratio_per_batch'] = sum(success_tool_call_list) / len(success_tool_call_list)
            metrics['val/tool_call/success_tool_call_rate'] = sum(success_tool_call_list) / (sum(apply_tool_call_list) + 1e-4)
            metrics['val/tool_call/success_tool_call_correct_answer_ratio'] = use_tool_correct_answer_num / (sum(success_tool_call_list) + 1e-4)
        else:   # Train
            num_rollouts = self.config.actor_rollout_ref.rollout.n
            sample_level_success_tool_call_list = [True if any(success_tool_call_list[i : i + num_rollouts]) else False for i in range(0, len(success_tool_call_list), num_rollouts)] 
            metrics['tool_call/effective_batch_success_tool_call_ratio'] = sum(success_tool_call_list) / len(success_tool_call_list)
            metrics['tool_call/effective_sample_success_tool_call_ratio'] = sum(sample_level_success_tool_call_list) / len(sample_level_success_tool_call_list)
            metrics['tool_call/success_tool_call_rate'] = sum(success_tool_call_list) / (sum(apply_tool_call_list) + 1e-4)
            metrics['tool_call/success_tool_call_correct_answer_ratio'] = use_tool_correct_answer_num / (sum(success_tool_call_list) + 1e-4)

            # log different scores combination ratio
            group_accuracy_reward_tensor = accuracy_scores.view(-1, num_rollouts)
            num_samples = group_accuracy_reward_tensor.shape[0]
            group_success_tool_call_mask = torch.tensor(success_tool_call_list).view(-1, num_rollouts)
            use_tool_correct_count = ((group_accuracy_reward_tensor == self.config.data.acc_reward_weight) & group_success_tool_call_mask).sum(dim=1)  # (bs,)
            direct_answer_correct_count = ((group_accuracy_reward_tensor == self.config.data.acc_reward_weight) & (~group_success_tool_call_mask)).sum(dim=1)  # (bs,)
            wrong_answer_count = (group_accuracy_reward_tensor == 0).sum(dim=1) # (bs,)

            # Obtain mask
            use_tool_correct_mask = use_tool_correct_count > 0
            direct_answer_correct_mask = direct_answer_correct_count > 0
            wrong_answer_mask = wrong_answer_count > 0

            # Due to rejection sample, here we will not encounter all_correct | all_wrong cases
            wrong_answer_use_tool_correct_num = (use_tool_correct_mask & wrong_answer_mask & (~direct_answer_correct_mask)).sum().item()  # [0, 1 - penalty]
            wrong_answer_direct_answer_correct_num = (direct_answer_correct_mask & wrong_answer_mask & (~use_tool_correct_mask)).sum().item()  # [0, 1]
            use_tool_correct_direct_answer_correct_num = (use_tool_correct_mask & direct_answer_correct_mask & (~wrong_answer_mask)).sum().item()  # [1 - penalty, 1]
            wrong_answer_use_tool_correct_direct_answer_correct_num = (use_tool_correct_mask & direct_answer_correct_mask & wrong_answer_mask).sum().item() # [0, 1 - penalty, 1]
            metrics['tool_call/wrong_answer_use_tool_correct_ratio'] = wrong_answer_use_tool_correct_num / num_samples
            metrics['tool_call/wrong_answer_direct_answer_correct_ratio'] = wrong_answer_direct_answer_correct_num / num_samples
            metrics['tool_call/use_tool_correct_direct_answer_correct_ratio'] = use_tool_correct_direct_answer_correct_num / num_samples
            metrics['tool_call/wrong_answer_use_tool_correct_direct_answer_correct_ratio'] = wrong_answer_use_tool_correct_direct_answer_correct_num / num_samples

        return metrics

    def _validate(self):
        reward_tensor_lst = []
        data_source_lst = []
        success_tool_call_count_lst = []
        if self.config.reward_model.log_rewards_separately:
            format_reward_tensor_lst = []
            acc_reward_tensor_lst = []

        # Lists to collect samples for the table
        sample_inputs = []
        sample_outputs = []
        sample_ground_truth = []
        sample_scores = []

        batch = None
        for test_data in self.val_dataloader:
            test_batch = DataProto.from_single_dict(test_data)

            test_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(test_batch.batch))],
                                                    dtype=object)

            # Store original inputs
            input_ids = test_batch.batch['input_ids']
            input_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in input_ids]
            sample_inputs.extend(input_texts)
            sample_ground_truth.extend(test_batch.non_tensor_batch['ground_truth']) # FIXME: Currently do not support ['reward_model']['ground_truth']

            if 'multi_modal_data' in test_batch.non_tensor_batch.keys():
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                )
            else:
                test_gen_batch = test_batch.pop(
                    batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                    non_tensor_batch_keys=['raw_prompt_ids'],
                )

            test_gen_batch.meta_info = {
                'eos_token_id': self.tokenizer.eos_token_id,
                'pad_token_id': self.tokenizer.pad_token_id,
                'recompute_log_prob': False,
                'do_sample': False,
                'validate': True,
            }

            # pad to be divisible by dp_size
            test_gen_batch_padded, pad_size = pad_dataproto_to_divisor(test_gen_batch, self.actor_rollout_wg.world_size)
            test_output_gen_batch_padded = self.actor_rollout_wg.generate_sequences(test_gen_batch_padded)
            # unpad
            test_output_gen_batch = unpad_dataproto(test_output_gen_batch_padded, pad_size=pad_size)
            print('validation generation end')

            # Store generated outputs
            output_ids = test_output_gen_batch.batch['responses']
            output_texts = [self.tokenizer.decode(ids, skip_special_tokens=True) for ids in output_ids]
            sample_outputs.extend(output_texts)
            test_batch = test_batch.union(test_output_gen_batch)
            if self.use_rm and self.config.data.get("val_general_qa_reward_fn", None) is None:
                reward_score_proto = self.rm_wg.compute_rm_score(test_batch)
                test_batch = test_batch.union(reward_score_proto)
            # evaluate using reward_function
            reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids = self.val_reward_fn(test_batch)
            if self.config.reward_model.log_rewards_separately:
                format_reward_tensor_lst.append(format_reward_tensor)
                acc_reward_tensor_lst.append(acc_reward_tensor)

            # Store scores
            scores = reward_tensor.sum(-1).cpu().tolist()
            sample_scores.extend(scores)
            reward_tensor_lst.append(reward_tensor)
            data_source_lst.append(test_batch.non_tensor_batch.get('data_source', ['unknown'] * reward_tensor.shape[0]))
            # Store tool call states
            multi_modal_data = test_batch.non_tensor_batch['multi_modal_data']
            success_tool_call_count_lst.extend([1 if len(item['image']) > 1 else 0 for item in multi_modal_data])
            if batch is None:
                batch = test_batch
            else:
                batch = DataProto.concat([batch, test_batch])

        self._maybe_log_val_generations_to_wandb(doc_ids=test_data['doc_id'], inputs=sample_inputs, outputs=sample_outputs, ground_truth=sample_ground_truth, scores=sample_scores)

        reward_tensor = torch.cat(reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
        if self.config.reward_model.log_rewards_separately:
            format_reward_tensor = torch.cat(format_reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            acc_reward_tensor = torch.cat(acc_reward_tensor_lst, dim=0).sum(-1).cpu()  # (batch_size,)
            if self.tool_call:
                penalized_acc_reward_tensor = reward_tensor - format_reward_tensor

        data_sources = np.concatenate(data_source_lst, axis=0)

        # evaluate test_score based on data source
        data_source_reward = {}
        for i in range(reward_tensor.shape[0]):
            data_source = data_sources[i]
            if data_source not in data_source_reward:
                data_source_reward[data_source] = []
            data_source_reward[data_source].append(reward_tensor[i].item())

        metric_dict = {}
        for data_source, rewards in data_source_reward.items():
            metric_dict[f'val/test_score/{data_source}/total'] = np.mean(rewards)

        if self.config.reward_model.log_rewards_separately:
            data_source_format_reward = {}
            data_source_acc_reward = {}
            if self.tool_call:
                data_source_tool_call_count = {}
                if self.config.data.supervise_tool_label:
                    data_source_necessary_tool_call_count = {}
                else:
                    data_source_use_tool_correct_answer_count = {}

            for i in range(reward_tensor.shape[0]):
                data_source = data_sources[i]
                if data_source not in data_source_format_reward:
                    data_source_format_reward[data_source] = []
                data_source_format_reward[data_source].append(format_reward_tensor[i].item())
                if data_source not in data_source_acc_reward:
                    data_source_acc_reward[data_source] = []
                data_source_acc_reward[data_source].append(acc_reward_tensor[i].item())
                if self.tool_call:
                    if data_source not in data_source_tool_call_count:
                        data_source_tool_call_count[data_source] = []
                    data_source_tool_call_count[data_source].append(success_tool_call_count_lst[i])
                    if self.config.data.supervise_tool_label:
                        if data_source not in data_source_necessary_tool_call_count:
                            data_source_necessary_tool_call_count[data_source] = []
                        data_source_necessary_tool_call_count[data_source].append(not (batch[i].non_tensor_batch['use_tool'] ^ (success_tool_call_count_lst[i] > 0)))
                    else:
                        if data_source not in data_source_use_tool_correct_answer_count:
                            data_source_use_tool_correct_answer_count[data_source] = []
                        data_source_use_tool_correct_answer_count[data_source].append((penalized_acc_reward_tensor[i] == 1 - self.config.data.tool_call_penalty))
            for data_source, rewards in data_source_format_reward.items():
                metric_dict[f'val/test_score/{data_source}/format_reward'] = np.mean(rewards)
            for data_source, rewards in data_source_acc_reward.items():
                metric_dict[f'val/test_score/{data_source}/acc_reward'] = np.mean(rewards)
            if self.tool_call:
                for data_source, tool_call_count in data_source_tool_call_count.items():
                    metric_dict[f'val/test_score/{data_source}/success_tool_call_ratio'] = np.mean(tool_call_count)
                if self.config.data.supervise_tool_label:
                    for data_source, necessary_tool_call_count in data_source_necessary_tool_call_count.items():
                        metric_dict[f'val/test_score/{data_source}/necessary_tool_call_ratio'] = np.mean(necessary_tool_call_count)
                else:
                    for data_source, use_tool_correct_answer_count in data_source_use_tool_correct_answer_count.items():
                        metric_dict[f'val/test_score/{data_source}/use_tool_correct_answer_ratio'] = np.mean(use_tool_correct_answer_count)
        if self.tool_call:
            batch.meta_info['validate'] = True
            batch.batch['acc_scores'] = acc_reward_tensor
            metric_dict = self._maybe_log_tool_call_metrics(batch=batch, metrics=metric_dict)
        return metric_dict

    def init_workers(self):
        """Init resource pool and worker group"""
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()}

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.ActorRollout],
                                                     config=self.config.actor_rollout_ref,
                                                     role='actor_rollout')
            self.resource_pool_to_cls[resource_pool]['actor_rollout'] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(cls=self.role_worker_mapping[Role.Critic], config=self.config.critic)
            self.resource_pool_to_cls[resource_pool]['critic'] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RefPolicy],
                                                  config=self.config.actor_rollout_ref,
                                                  role='ref')
            self.resource_pool_to_cls[resource_pool]['ref'] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model)
            self.resource_pool_to_cls[resource_pool]['rm'] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`. Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        self.wg_dicts = []
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(resource_pool=resource_pool, ray_cls_with_init=worker_dict_cls)
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)
            # keep the referece of WorkerDict to support ray >= 2.31. Ref: https://github.com/ray-project/ray/pull/45699
            self.wg_dicts.append(wg_dict)

        if self.use_critic:
            self.critic_wg = all_wg['critic']
            self.critic_wg.init_model()

        if self.use_reference_policy:
            self.ref_policy_wg = all_wg['ref']
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg['rm']
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg['actor_rollout']
        self.actor_rollout_wg.init_model()

    def _save_checkpoint(self):
        # path: given_path + `/global_step_{global_steps}` + `/actor`
        local_global_step_folder = os.path.join(self.config.trainer.default_local_dir,
                                                f'global_step_{self.global_steps}')
        actor_local_path = os.path.join(local_global_step_folder, 'actor')

        actor_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
            self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'actor')
        self.actor_rollout_wg.save_checkpoint(actor_local_path,
                                              actor_remote_path,
                                              self.global_steps,
                                              remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        if self.use_critic:
            critic_local_path = os.path.join(local_global_step_folder, 'critic')
            critic_remote_path = None if self.config.trainer.default_hdfs_dir is None else os.path.join(
                self.config.trainer.default_hdfs_dir, f'global_step_{self.global_steps}', 'critic')
            self.critic_wg.save_checkpoint(critic_local_path,
                                           critic_remote_path,
                                           self.global_steps,
                                           remove_previous_ckpt=self.config.trainer.remove_previous_ckpt_in_save)

        # save dataloader
        dataloader_local_path = os.path.join(local_global_step_folder, 'data.pt')
        dataloader_state_dict = self.train_dataloader.state_dict()
        torch.save(dataloader_state_dict, dataloader_local_path)

        # latest checkpointed iteration tracker (for atomic usage)
        local_latest_checkpointed_iteration = os.path.join(self.config.trainer.default_local_dir,
                                                           'latest_checkpointed_iteration.txt')
        with open(local_latest_checkpointed_iteration, 'w') as f:
            f.write(str(self.global_steps))

    def _load_checkpoint(self):
        if self.config.trainer.resume_mode == 'disable':
            return 0

        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            NotImplementedError('load from hdfs is not implemented yet')
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        if self.config.trainer.resume_mode == 'auto':
            if global_step_folder is None:
                print('Training from scratch')
                return 0
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(self.config.trainer.resume_mode, str), "resume ckpt must be str type"
                assert 'global_step_' in self.config.trainer.resume_mode, "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        print(f'Load from checkpoint folder: {global_step_folder}')
        # set global step
        self.global_steps = int(global_step_folder.split('global_step_')[-1])

        print(f'Setting global step to {self.global_steps}')
        print(f'Resuming from {global_step_folder}')

        actor_path = os.path.join(global_step_folder, 'actor')
        critic_path = os.path.join(global_step_folder, 'critic')
        # load actor
        self.actor_rollout_wg.load_checkpoint(actor_path,
                                              del_local_after_load=self.config.trainer.del_local_ckpt_after_load)
        # load critic
        if self.use_critic:
            self.critic_wg.load_checkpoint(critic_path,
                                           del_local_after_load=self.config.trainer.del_local_ckpt_after_load)

        # load dataloader,
        # TODO: from remote not implemented yet
        dataloader_local_path = os.path.join(global_step_folder, 'data.pt')
        if os.path.exists(dataloader_local_path):
            dataloader_state_dict = torch.load(dataloader_local_path)
            self.train_dataloader.load_state_dict(dataloader_state_dict)
        else:
            print(f"Warning: No dataloader state found at {dataloader_local_path}, will start from scratch")

    def _balance_batch(self, batch: DataProto, metrics, logging_prefix='global_seqlen'):
        """Reorder the data on single controller such that each dp rank gets similar total tokens"""
        attention_mask = batch.batch['attention_mask']
        batch_size = attention_mask.shape[0]
        global_seqlen_lst = batch.batch['attention_mask'].view(batch_size, -1).sum(-1).tolist()  # (train_batch_size,)
        world_size = self.actor_rollout_wg.world_size
        global_partition_lst = get_seqlen_balanced_partitions(global_seqlen_lst,
                                                              k_partitions=world_size,
                                                              equal_size=True)
        # reorder based on index. The data will be automatically equally partitioned by dispatch function
        global_idx = torch.tensor([j for partition in global_partition_lst for j in partition])
        batch.reorder(global_idx)
        global_balance_stats = log_seqlen_unbalance(seqlen_list=global_seqlen_lst,
                                                    partitions=global_partition_lst,
                                                    prefix=logging_prefix)
        metrics.update(global_balance_stats)

    def fit(self):
        """
        The training loop of PPO.
        The driver process only need to call the compute functions of the worker group through RPC to construct the PPO dataflow.
        The light-weight advantage computation is done on the driver process.
        """
        from verl.utils.tracking import Tracking
        from omegaconf import OmegaConf

        logger = Tracking(project_name=self.config.trainer.project_name,
                          experiment_name=self.config.trainer.experiment_name,
                          default_backend=self.config.trainer.logger,
                          config=OmegaConf.to_container(self.config, resolve=True))

        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

        # perform validation before training
        # currently, we only support validation using the reward_function.
        if self.val_reward_fn is not None and self.config.trainer.get('val_before_train', True) and self.config.trainer.test_freq != -1:
            val_metrics = self._validate()
            pprint(f'Initial validation metrics: {val_metrics}')
            logger.log(data=val_metrics, step=self.global_steps)
            if self.config.trainer.get('val_only', False):
                return

        # we start from step 1
        self.global_steps += 1

        batch = None
        num_prompt_in_batch = 0
        num_gen_batches = 0
        solve_none = 0
        solve_all = 0
        solve_acc_none = 0
        solve_acc_all = 0
        solve_format_none = 0
        solve_format_all = 0
        success_tool_call_per_batch = 0
        success_tool_call_per_sample = 0
        totals = 0
        invalid_num = 0
        effective_num = 0

        for epoch in range(self.config.trainer.total_epochs):
            for batch_dict in self.train_dataloader:
                metrics = {}
                timing_raw = {}

                # batch: DataProto = DataProto.from_single_dict(batch_dict)
                new_batch: DataProto = DataProto.from_single_dict(batch_dict)
                num_gen_batches += 1
                # pop those keys for generation
                if 'multi_modal_data' in new_batch.non_tensor_batch.keys():
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids', 'multi_modal_data'],
                    )
                else:
                    gen_batch = new_batch.pop(
                        batch_keys=['input_ids', 'attention_mask', 'position_ids'],
                        non_tensor_batch_keys=['raw_prompt_ids'],
                    )

                with _timer('step', timing_raw):
                    # generate a batch
                    with _timer('gen', timing_raw):
                        gen_batch_output = self.actor_rollout_wg.generate_sequences(gen_batch)

                    if self.config.algorithm.adv_estimator == AdvantageEstimator.REMAX:
                        with _timer('gen_max', timing_raw):
                            gen_baseline_batch = deepcopy(gen_batch)
                            gen_baseline_batch.meta_info['do_sample'] = False
                            gen_baseline_output = self.actor_rollout_wg.generate_sequences(gen_baseline_batch)

                            new_batch = new_batch.union(gen_baseline_output)
                            reward_baseline_tensor = self.reward_fn(new_batch)
                            reward_baseline_tensor = reward_baseline_tensor.sum(dim=-1)

                            new_batch.pop(batch_keys=list(gen_baseline_output.batch.keys()))

                            new_batch.batch['reward_baselines'] = reward_baseline_tensor

                            del gen_baseline_batch, gen_baseline_output

                    new_batch.non_tensor_batch['uid'] = np.array([str(uuid.uuid4()) for _ in range(len(new_batch.batch))],
                                                             dtype=object)
                    # repeat to align with repeated responses in rollout
                    new_batch = new_batch.repeat(repeat_times=self.config.actor_rollout_ref.rollout.n, interleave=True)
                    new_batch = new_batch.union(gen_batch_output)

                    # compute global_valid tokens
                    new_batch.meta_info['global_token_num'] = torch.sum(new_batch.batch['attention_mask'], dim=-1).tolist()

                    with _timer('adv', timing_raw):
                        # compute scores. Support both model and function-based.
                        # We first compute the scores using reward model. Then, we call reward_fn to combine
                        # the results from reward model and rule-based results.
                        if self.use_rm:
                            # Compute reward model score
                            reward_score_proto = self.rm_wg.compute_rm_score(new_batch)
                            new_batch = new_batch.union(reward_score_proto)

                        reward_tensor, acc_reward_tensor, format_reward_tensor, invalid_uids = self.reward_fn(new_batch)
                        if self.config.trainer.soft_tool_call_penalty_ratio > 0:
                            assert self.config.data.tool_call_penalty > 0, f"Soft tool call penalty was triggered, but found tool_call_penalty is 0."
                            max_response_length = reward_tensor.shape[-1]
                            penalized_accuracy_tensor = reward_tensor - format_reward_tensor    # Only keep accuracy scores with penalty
                            penalized_accuracy_tensor_ = penalized_accuracy_tensor.view(-1, self.config.actor_rollout_ref.rollout.n, max_response_length)    # (bs, n, max_response_length)
                            correct_item_mask = (penalized_accuracy_tensor_ > 0).any(dim=2)
                            correct_direct_answer = (penalized_accuracy_tensor_ == self.config.data.acc_reward_weight).any(dim=2)
                            correct_item_num = correct_item_mask.sum(dim=1)
                            correct_direct_answer_num = correct_direct_answer.sum(dim=1)
                            ratio = correct_direct_answer_num.float() / (correct_item_num.float() + 1e-4)   # Add 1e-4 to avoid zero-division

                            # Penalty
                            penalty_tensor_mask = (ratio < self.config.trainer.soft_tool_call_penalty_ratio) & (ratio > 0)
                            if self.config.trainer.soft_tool_call_filter:
                                all_correct_mask = (penalized_accuracy_tensor_.max(dim=-1)[0] != 0).all(dim=1)  # (bs,)，
                                any_wrong_mask = (penalized_accuracy_tensor_.max(dim=-1)[0] == 0).any(dim=1) # (bs,)，
                                penalty_tensor_mask = (penalty_tensor_mask & all_correct_mask) | any_wrong_mask  # (bs,)
                            penalty_tensor_mask_expand = penalty_tensor_mask[:, None, None]  # (bs, 1, 1)
                            correct_direct_answer_item_mask = penalized_accuracy_tensor_ == self.config.data.acc_reward_weight

                            # Direct Answer Penalty
                            if self.config.data.direct_answer_penalty > 0:
                                modify_penalty_direct_answer_item_mask = penalty_tensor_mask_expand & correct_direct_answer_item_mask
                                penalized_accuracy_tensor_[modify_penalty_direct_answer_item_mask] = self.config.data.acc_reward_weight - self.config.data.direct_answer_penalty
                            reward_tensor = penalized_accuracy_tensor_.view(-1, max_response_length) + format_reward_tensor   # Add format scores

                        invalid_num += len(invalid_uids)

                        new_batch.batch['token_level_scores'] = reward_tensor
                        new_batch.batch['acc_scores'] = acc_reward_tensor.sum(-1)
                        new_batch.batch['format_scores'] = format_reward_tensor.sum(-1)

                        uids = new_batch.non_tensor_batch['uid']
                        unique_uids = np.unique(uids)
                        valid_mask = torch.ones(len(uids), dtype=torch.bool)
                            
                        for uid in unique_uids:
                            uid_mask = uids == uid

                            if uid in invalid_uids:

                                valid_mask[uid_mask] = False
                                continue

                            uid_rewards = reward_tensor[uid_mask].sum(-1)  # Sum rewards for each sequence

                            if uid_rewards.std().item() < 1e-4: # epsilon
                                valid_mask[uid_mask] = False

                            if (uid_rewards == 0).all():
                                solve_none += 1
                            elif (uid_rewards == self.config.data.acc_reward_weight + self.config.data.format_reward_weight).all():
                                solve_all += 1

                            uid_acc_rewards = acc_reward_tensor[uid_mask].sum(-1)
                            uid_format_rewards = format_reward_tensor[uid_mask].sum(-1)
                            # Check if all rewards are 0 or all are 1 for this uid
                            if (uid_acc_rewards == 0).all():
                                solve_acc_none += 1
                            elif (uid_acc_rewards == self.config.data.acc_reward_weight).all():
                                solve_acc_all += 1

                            if (uid_format_rewards == 0).all():
                                solve_format_none += 1
                            elif (uid_format_rewards == self.config.data.format_reward_weight).all():
                                solve_format_all += 1

                            totals += 1

                        effective_num += valid_mask.sum().item() // self.config.actor_rollout_ref.rollout.n

                        if self.config.trainer.rejection_sample:
                            # If no valid samples remain, skip this batch and get a new one
                            if not valid_mask.any():
                                continue

                            # Filter batch to keep only valid samples
                            new_batch = new_batch[valid_mask]
                            new_batch = dataprotoitem_to_dataproto(new_batch)

                            assert new_batch.batch['input_ids'].shape[0] % self.config.actor_rollout_ref.rollout.n == 0
                            num_prompt_in_batch += new_batch.batch['input_ids'].shape[0] // self.config.actor_rollout_ref.rollout.n

                            if batch is None:
                                batch = new_batch
                            else:
                                batch = DataProto.concat([batch, new_batch])

                            prompt_bsz = self.config.data.train_batch_size
                            if num_prompt_in_batch < prompt_bsz:
                                print(f'{num_prompt_in_batch=} < {prompt_bsz=}')
                                max_num_gen_batches = self.config.algorithm.max_num_gen_batches
                                if max_num_gen_batches <= 0 or num_gen_batches < max_num_gen_batches:
                                    print(f'{num_gen_batches=}. Keep generating...')
                                    continue
                                else:
                                    raise ValueError(
                                        f'{num_gen_batches=} >= {max_num_gen_batches=}. Generated too many. Please check your data.'
                                    )
                            else:
                                # Align the batch
                                traj_bsz = self.config.data.train_batch_size * self.config.actor_rollout_ref.rollout.n
                                batch = batch[:traj_bsz]
                                batch = dataprotoitem_to_dataproto(batch)
                        else:
                            batch = new_batch

                        # balance the number of valid tokens on each dp rank.
                        # Note that this breaks the order of data inside the batch.
                        # Please take care when you implement group based adv computation such as GRPO and rloo
                        if self.tool_call:
                            metrics = self._maybe_log_tool_call_metrics(batch, metrics)
                        self._balance_batch(batch, metrics=metrics)

                        metrics['batch/solve_none'] = solve_none / totals
                        metrics['batch/solve_all'] = solve_all / totals
                        metrics['batch/solve_acc_none'] = solve_acc_none / totals
                        metrics['batch/solve_acc_all'] = solve_acc_all / totals
                        metrics['batch/solve_format_none'] = solve_format_none / totals
                        metrics['batch/solve_format_all'] = solve_format_all / totals
                        metrics['batch/invalid_ratio'] = invalid_num / totals
                        metrics['batch/effective_ratio'] = effective_num / totals
                        metrics['batch/success_tool_call_per_batch'] = success_tool_call_per_batch / (totals * self.config.actor_rollout_ref.rollout.n)
                        metrics['batch/success_tool_call_per_sample'] = success_tool_call_per_sample / totals

                        # compute rewards. apply_kl_penalty if available
                        if not self.config.actor_rollout_ref.actor.get('use_kl_loss', False):
                            batch, kl_metrics = apply_kl_penalty(batch,
                                                                 kl_ctrl=self.kl_ctrl,
                                                                 kl_penalty=self.config.algorithm.kl_penalty)
                            metrics.update(kl_metrics)
                        else:
                            batch.batch['token_level_rewards'] = batch.batch['token_level_scores']

                        # compute advantages, executed on the driver process
                        batch = compute_advantage(batch,
                                                  adv_estimator=self.config.algorithm.adv_estimator,
                                                  gamma=self.config.algorithm.gamma,
                                                  lam=self.config.algorithm.lam,
                                                  num_repeat=self.config.actor_rollout_ref.rollout.n)

                    # recompute old_log_probs
                    with _timer('old_log_prob', timing_raw):
                        old_log_prob = self.actor_rollout_wg.compute_log_prob(batch)
                        batch = batch.union(old_log_prob)

                    if self.use_reference_policy:
                        # compute reference log_prob
                        with _timer('ref', timing_raw):
                            ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                            batch = batch.union(ref_log_prob)

                    # compute values
                    if self.use_critic:
                        with _timer('values', timing_raw):
                            values = self.critic_wg.compute_values(batch)
                            batch = batch.union(values)

                    # update critic
                    if self.use_critic:
                        with _timer('update_critic', timing_raw):
                            critic_output = self.critic_wg.update_critic(batch)
                        critic_output_metrics = reduce_metrics(critic_output.meta_info['metrics'])
                        metrics.update(critic_output_metrics)

                    # implement critic warmup
                    if self.config.trainer.critic_warmup <= self.global_steps:

                        # print(f"e1 batch.batch_size[0]: {batch.batch.batch_size[0]}")

                        # update actor
                        with _timer('update_actor', timing_raw):
                            actor_output = self.actor_rollout_wg.update_actor(batch)
                        actor_output_metrics = reduce_metrics(actor_output.meta_info['metrics'])
                        metrics.update(actor_output_metrics)

                    # log training rollouts
                    if self.config.trainer.log_training_rollouts_freq > 0 and self.global_steps % self.config.trainer.log_training_rollouts_freq == 0:
                        self._maybe_log_train_generations_to_wandb(batch, batch.batch['token_level_scores'])

                    # validate
                    if self.val_reward_fn is not None and self.config.trainer.test_freq > 0 and \
                        self.global_steps % self.config.trainer.test_freq == 0:
                        with _timer('testing', timing_raw):
                            val_metrics: dict = self._validate()
                        metrics.update(val_metrics)

                    if self.config.trainer.save_freq > 0 and \
                            self.global_steps % self.config.trainer.save_freq == 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()

                # collect metrics
                metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
                metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))

                metrics["train/num_gen_batches"] = num_gen_batches
                batch = None
                num_prompt_in_batch = 0
                num_gen_batches = 0
                solve_none = 0
                solve_all = 0
                solve_acc_none = 0
                solve_acc_all = 0
                solve_format_none = 0
                solve_format_all = 0
                success_tool_call_per_batch = 0
                success_tool_call_per_sample = 0
                totals = 0
                invalid_num = 0
                effective_num = 0

                # TODO: make a canonical logger that supports various backend
                logger.log(data=metrics, step=self.global_steps)

                self.global_steps += 1

                if self.global_steps >= self.total_training_steps:

                    # perform validation after training
                    if self.val_reward_fn is not None:
                        val_metrics = self._validate()
                        pprint(f'Final validation metrics: {val_metrics}')
                        logger.log(data=val_metrics, step=self.global_steps)
                    if self.config.trainer.save_freq > 0 and \
                            (self.global_steps - 1) % self.config.trainer.save_freq != 0:
                        with _timer('save_checkpoint', timing_raw):
                            self._save_checkpoint()
                    return
