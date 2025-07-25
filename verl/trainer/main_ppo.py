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
Note that we don't combine the main with ray_trainer as ray_trainer is used by other main.
"""
from verl.trainer.ppo.ray_trainer import RayPPOTrainer
from verl.trainer.constants import SYSTEM_PROMPT_MAP

import ray
import hydra


@hydra.main(config_path='config', config_name='ppo_trainer', version_base=None)
def main(config):
    run_ppo(config)


def run_ppo(config, compute_score=None):
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={'env_vars': {'TOKENIZERS_PARALLELISM': 'true', 'NCCL_DEBUG': 'WARN'}})

    ray.get(main_task.remote(config, compute_score))


@ray.remote(num_cpus=1)  # please make sure main_task is not scheduled on head
def main_task(config, compute_score=None):
    from verl.utils.fs import copy_to_local
    # print initial config
    from pprint import pprint
    from omegaconf import OmegaConf
    pprint(OmegaConf.to_container(config, resolve=True))  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    # download the checkpoint from hdfs
    local_path = copy_to_local(config.actor_rollout_ref.model.path)

    # instantiate tokenizer
    from verl.utils import hf_tokenizer, hf_processor
    tokenizer = hf_tokenizer(local_path)
    processor = hf_processor(local_path, use_fast=True)  # used for multimodal LLM, could be none

    # define worker classes
    if config.actor_rollout_ref.actor.strategy == 'fsdp':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.fsdp_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray import RayWorkerGroup
        ray_worker_group_cls = RayWorkerGroup

    elif config.actor_rollout_ref.actor.strategy == 'megatron':
        assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
        from verl.workers.megatron_workers import ActorRolloutRefWorker, CriticWorker
        from verl.single_controller.ray.megatron import NVMegatronRayWorkerGroup
        ray_worker_group_cls = NVMegatronRayWorkerGroup

    else:
        raise NotImplementedError

    from verl.trainer.ppo.ray_trainer import ResourcePoolManager, Role

    role_worker_mapping = {
        Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
        Role.Critic: ray.remote(CriticWorker),
        Role.RefPolicy: ray.remote(ActorRolloutRefWorker)
    }

    global_pool_id = 'global_pool'
    resource_pool_spec = {
        global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
    }
    mapping = {
        Role.ActorRollout: global_pool_id,
        Role.Critic: global_pool_id,
        Role.RefPolicy: global_pool_id,
    }

    # we should adopt a multi-source reward function here
    # - for rule-based rm, we directly call a reward score
    # - for model-based rm, we call a model
    # - for code related prompt, we send to a sandbox if there are test cases
    # - finally, we combine all the rewards together
    # - The reward type depends on the tag of the data
    if config.reward_model.enable:
        if config.reward_model.strategy == 'fsdp':
            from verl.workers.fsdp_workers import RewardModelWorker
        elif config.reward_model.strategy == 'megatron':
            from verl.workers.megatron_workers import RewardModelWorker
        elif config.reward_model.strategy == 'verifier':
            from verl.workers.reward_model.verifier import RewardModelWorker
            # For Qwen3 Judge
        else:
            raise NotImplementedError
        role_worker_mapping[Role.RewardModel] = ray.remote(RewardModelWorker)
        mapping[Role.RewardModel] = global_pool_id

    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    if reward_manager_name == 'naive':
        from verl.workers.reward_manager import NaiveRewardManager
        reward_manager_cls = NaiveRewardManager
    elif reward_manager_name == 'naive_multithreads':
        from verl.workers.reward_manager import NaiveMultiThreadsRewardManager
        reward_manager_cls = NaiveMultiThreadsRewardManager
    elif reward_manager_name == 'naive_multithreads_tool':
        from verl.workers.reward_manager import NaiveMultiThreadsToolRewardManager
        reward_manager_cls = NaiveMultiThreadsToolRewardManager
    elif reward_manager_name == 'verifier_reward_manager':
        from verl.workers.reward_manager import VerifierRewardManager
        reward_manager_cls = VerifierRewardManager
    else:
        raise NotImplementedError

    system_prompt_type = config.data.get('system_prompt', False)
    system_prompt = SYSTEM_PROMPT_MAP[system_prompt_type] if system_prompt_type else None
    extra_info = {
        "acc_reward_weight": config.data.get("acc_reward_weight", 1.0),
        "format_reward_weight": config.data.get("format_reward_weight", 1.0),
        "use_tool_reward_weight": config.data.get("use_tool_reward_weight", 0.0),
        "tool_call_penalty": config.data.get("tool_call_penalty", 0.1),
        "extract_answer_tags": config.data.get("extract_answer_tags", "split"),
        "general_qa_reward_fn": config.data.get("general_qa_reward_fn", "general_qa_gpt"),
        "val_general_qa_reward_fn": config.data.get("val_general_qa_reward_fn", None),
        "gpt_extract_answer": config.data.get("gpt_extract_answer", False),
        "model_system_prompt": system_prompt,
        "supervise_tool_label": config.data.get("supervise_tool_label", False),
        "strict_tool_call_penalty": config.data.get("strict_tool_call_penalty", False),
    }
    reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=0, compute_score=compute_score, mode="train", extra_info=extra_info, gpt_threads=config.data.get("gpt_threads", 100))
    if config.data.get("val_general_qa_reward_fn", None) == "general_qa_gpt":
        from verl.workers.reward_manager import NaiveMultiThreadsRewardManager
        val_reward_manager_cls = NaiveMultiThreadsRewardManager
        val_reward_fn = val_reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, mode="val", extra_info=extra_info)
    elif config.data.get("val_general_qa_reward_fn", None) == "general_qa_tool":
        from verl.workers.reward_manager import NaiveMultiThreadsToolRewardManager
        val_reward_manager_cls = NaiveMultiThreadsToolRewardManager
        val_reward_fn = val_reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, mode="val", extra_info=extra_info)
    else:
        val_reward_fn = reward_manager_cls(tokenizer=tokenizer, num_examine=1, compute_score=compute_score, mode="val", extra_info=extra_info)

    resource_pool_manager = ResourcePoolManager(resource_pool_spec=resource_pool_spec, mapping=mapping)

    trainer = RayPPOTrainer(config=config,
                            tokenizer=tokenizer,
                            processor=processor,
                            role_worker_mapping=role_worker_mapping,
                            resource_pool_manager=resource_pool_manager,
                            ray_worker_group_cls=ray_worker_group_cls,
                            reward_fn=reward_fn,
                            val_reward_fn=val_reward_fn)
    trainer.init_workers()
    trainer.fit()


if __name__ == '__main__':
    main()
