# run on 8xH100
# make sure your current working directory is the root of the project
# export HIP_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
# export ROCR_VISIBLE_DEVICES=$HIP_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
# export NCCL_DEBUG=INFO
# export NCCL_P2P_DISABLE=1   
# export NCCL_IB_DISABLE=1    
# export NCCL_SOCKET_IFNAME=eth0 
# export NCCL_SHM_DISABLE=1
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export WANDB_MODE=offline

export RAY_IGNORE_UNHANDLED_ERRORS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
# export VLLM_ATTENTION_BACKEND=XFORMERS
export GLOO_SOCKET_IFNAME=eth0  # 或其它非 lo 的网络接口: eth0
export NCCL_DEBUG=INFO
export GLOO_DEBUG=1
export API_PARALLEL=60
# export VLLM_USE_V1=0

set -x

ulimit -n 65535
experiment_name='Qwen3-8B-grpo-xm-1k-1129'
export TRIAL_NAME=$experiment_name

# PROJECT_DIR="$(pwd)"
CONFIG_PATH=/mnt/data2/liyonghui/verl0715/verl/examples/sglang_multiturn/config
CONFIG_NAME='doctor_multiturn_grpo_w_interaction'
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-224}    # 512、448、384
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
OFFLOAD=${OFFLOAD:-False}

# Algorithm
adv_estimator=grpo  # gae, reinforce_plus_plus , grpo, remax
use_kl_in_reward=False

# Data Config
# train_files=/mnt/data2/liyonghui/data/gsm8k_interaction2/train.parquet
# val_files=/mnt/data2/liyonghui/data/gsm8k_interaction2/test.parquet
# train_files=/mnt/data2/liyonghui/data/json_data_en_align_think_generated_rl_parquet_5000/train.parquet
# val_files=/mnt/data2/liyonghui/data/json_data_en_align_think_generated_rl_parquet_5000/valid.parquet
train_files=/mnt/data2/liyonghui/Patient-zero/项目结题/traindata/呼吸科_1000_new_251129.jsonl
val_files=/mnt/data2/liyonghui/Patient-zero/项目结题/traindata/呼吸科_200_new_251129.jsonl
train_batch_size=256   #896     # 1024、512、448、384
max_prompt_length=512   # 1024
max_response_length=512    # $((1024 * 3))   # 1024 * 3
filter_overlong_prompts=True
data_truncation='error'
model_path=/mnt/data2/liyonghui/models/Llama-3.1-8B-Instruct
# model_path=/mnt/data2/liyonghui/sft/marge_ckpts/Qwen2.5-7B-Instruct-SFT-MEDDG-think-ep15-eval

# /mnt/data2/liyonghui/verl/checkpoints/sft_meddialog/qwen_7b_meddialog/global_step_22

# Experience Config
experience_enable=False
# experience_embedding_model=/mnt/data2/liyonghui/ckpts/jina-embeddings-v2-base-zh
experience_file_path=/mnt/data2/liyonghui/verl0715/verl/verl/trainer/experience/${experiment_name}.jsonl
experience_retrieval_top_k=2
experience_rerank_top_n=30
experience_use_action_suggestion=True
experience_use_novelty_filter=True
experience_reward_coefficient=0.5
experience_novelty_threshold=0.9
experience_high_reward_std_factor=1.0
experience_high_reward_boundary=0.7
data_return_raw_chat=True

# Interaction Config
multi_turn_enable=True
interaction_config_path=/mnt/data2/liyonghui/verl0715/verl/examples/sglang_multiturn/config/interaction_config/doctor_interaction_config.yaml
tool_config_path=/mnt/data2/liyonghui/verl0715/verl/examples/sglang_multiturn/config/tool_config/doctor_tool_config.yaml
max_user_turns=1
max_assistant_turns=1

# Actor / Rollout (Policy Model)
actor_optim_lr=1e-6
actor_use_remove_padding=False
# actor_ppo_mini_batch_size=256   # 256、128  
# actor_ppo_micro_batch_size_per_gpu=18   # 16、8  - change
# actor_fsdp_config_param_offload=False
# actor_fsdp_config_optimizer_offload=False   
actor_use_kl_loss=False
actor_enable_gradient_checkpointing=True   # True
actor_enable_activation_offloading=True
# actor_log_prob_micro_batch_size_per_gpu=8  # 8
actor_tensor_model_parallel_size=1   # 2
actor_rollout_name=sglang
actor_rollout_mode=sync
actor_gpu_memory_utilization=0.7
actor_kl_loss_coef=0.001
actor_kl_loss_type=low_var_kl
actor_entropy_coeff=0
actor_rollout_n=8
actor_fsdp_config_model_dtype=bfloat16


# Critic (Reward Model)
critic_optim_lr=1e-5
critic_use_remove_padding=True
critic_enable_gradient_checkpointing=True   # True
critic_ppo_micro_batch_size_per_gpu=16   # 32、16   - change
critic_fsdp_config_param_offload=False
critic_fsdp_config_optimizer_offload=False
critic_warmup=0


# Reward
reward_manager=parallel


# logger
logger=['console','wandb']    # ['console','wandb']
project_name='verl_doctor_rl'
n_gpus_per_node=7
nnodes=1 
save_freq=4
test_freq=2
total_epochs=10   # 15


python3 -m verl.trainer.main_ppo \
    --config-path="$CONFIG_PATH" \
    --config-name="$CONFIG_NAME" \
    algorithm.adv_estimator=$adv_estimator \
    data.train_batch_size=$TRAIN_BATCH_SIZE \
    data.max_prompt_length=$max_prompt_length \
    data.max_response_length=$max_response_length \
    data.filter_overlong_prompts=$filter_overlong_prompts \
    data.truncation=$data_truncation \
    data.return_raw_chat=$data_return_raw_chat\
    experience.enable=$experience_enable \
    experience.retrieval_top_k=$experience_retrieval_top_k \
    experience.file_path=$experience_file_path \
    experience.reward_coefficient=$experience_reward_coefficient \
    +experience.rerank_top_n=$experience_rerank_top_n \
    experience.use_novelty_filter=$experience_use_novelty_filter \
    experience.novelty_threshold=$experience_novelty_threshold \
    experience.use_action_suggestion=$experience_use_action_suggestion \
    experience.high_reward_std_factor=$experience_high_reward_std_factor \
    experience.high_reward_boundary=$experience_high_reward_boundary \
    actor_rollout_ref.model.path=$model_path \
    actor_rollout_ref.model.use_remove_padding=$actor_use_remove_padding \
    actor_rollout_ref.model.enable_gradient_checkpointing=$actor_enable_gradient_checkpointing \
    +actor_rollout_ref.model.enable_activation_offloading=$actor_enable_activation_offloading \
    actor_rollout_ref.actor.optim.lr=$actor_optim_lr \
    actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.actor.use_kl_loss=$actor_use_kl_loss \
    actor_rollout_ref.actor.kl_loss_coef=$actor_kl_loss_coef \
    actor_rollout_ref.actor.kl_loss_type=$actor_kl_loss_type \
    actor_rollout_ref.actor.entropy_coeff=$actor_entropy_coeff \
    actor_rollout_ref.actor.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=$OFFLOAD \
    +actor_rollout_ref.actor.fsdp_config.model_dtype=$actor_fsdp_config_model_dtype \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.rollout.tensor_model_parallel_size=$actor_tensor_model_parallel_size \
    actor_rollout_ref.rollout.name=$actor_rollout_name \
    actor_rollout_ref.rollout.mode=$actor_rollout_mode \
    actor_rollout_ref.rollout.gpu_memory_utilization=$actor_gpu_memory_utilization \
    actor_rollout_ref.rollout.n=$actor_rollout_n \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=$MICRO_BATCH_SIZE \
    actor_rollout_ref.ref.fsdp_config.param_offload=$OFFLOAD \
    actor_rollout_ref.rollout.trace.token2text=True \
    actor_rollout_ref.rollout.trace.backend=weave \
    reward_model.reward_manager=$reward_manager \
    critic.optim.lr=$critic_optim_lr \
    critic.model.use_remove_padding=$critic_use_remove_padding \
    critic.model.path=$model_path \
    critic.model.enable_gradient_checkpointing=$critic_enable_gradient_checkpointing \
    critic.ppo_micro_batch_size_per_gpu=$critic_ppo_micro_batch_size_per_gpu \
    critic.model.fsdp_config.param_offload=$critic_fsdp_config_param_offload \
    critic.model.fsdp_config.optimizer_offload=$critic_fsdp_config_optimizer_offload \
    algorithm.use_kl_in_reward=$use_kl_in_reward \
    trainer.critic_warmup=$critic_warmup \
    trainer.logger=$logger \
    trainer.project_name=$project_name \
    trainer.experiment_name=$experiment_name \
    trainer.n_gpus_per_node=$n_gpus_per_node \
    trainer.nnodes=$nnodes \
    trainer.save_freq=$save_freq \
    trainer.test_freq=$test_freq \
    data.train_files=$train_files \
    data.val_files=$val_files \
    actor_rollout_ref.rollout.multi_turn.enable=$multi_turn_enable \
    actor_rollout_ref.rollout.multi_turn.max_user_turns=$max_user_turns \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=$max_assistant_turns \
    actor_rollout_ref.rollout.multi_turn.interaction_config_path=$interaction_config_path \
    trainer.total_epochs=$total_epochs \
    "$@"

