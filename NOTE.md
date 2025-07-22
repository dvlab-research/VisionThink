Thanks for your interest in **VisionThink**.  
On this page, we list the **main modified files** for your convenience in checking and integrating into your project.

---

### 📁 Dataset

- `verl/utils/dataset/multimodal_dataset.py`

---

### 🚀 Rollout

- `verl/trainer/ppo/ray_trainer.py`  
  *Main logic for training.*

- `verl/workers/rollout/vllm_rollout/vllm_rollout_spmd.py`  
  *Multi-turn rollout logic.*

---

### 🧠 Reward Manager

- `verl/workers/reward_manager/naive_multithreads.py`  
  *GPT-based judge for General VQA.*

- `verl/workers/reward_manager/naive_multithreads_tool.py`  
  *GPT-based judge for General VQA and Efficient Reasoning VLM.*

- `verl/workers/reward_manager/verifier_reward_manager.py`  
  *Open-source LLM (e.g., Qwen) as judge.*

---

### 🎯 Reward Function

- `verl/utils/reward_score/general_qa_gpt.py`  
  *GPT-based scoring for General VQA.*

- `verl/utils/reward_score/general_qa_tool.py`  
  *GPT-based scoring for General VQA and Efficient Reasoning VLM.*

- `verl/workers/reward_model/verifier/verifier.py`  
  *Scoring with open-source verifier models like Qwen.*
