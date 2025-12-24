export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6
export HYDRA_FULL_ERROR=1
export VERL_LOGGING_LEVEL=DEBUG
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export RAY_IGNORE_UNHANDLED_ERRORS=1
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
export GLOO_SOCKET_IFNAME=eth0 
export NCCL_DEBUG=INFO
export GLOO_DEBUG=1
export API_PARALLEL=60

set -x

ulimit -n 65535
experiment_name='Qwen3-8B-grpo-xm-1k-1129'
export TRIAL_NAME=$experiment_name

CONFIG_PATH=/mnt/data2/liyonghui/verl0715/verl/examples/sglang_multiturn/config
CONFIG_NAME='doctor_multiturn_grpo_w_interaction'
TRAIN_BATCH_SIZE=${TRAIN_BATCH_SIZE:-224}
MICRO_BATCH_SIZE=${MICRO_BATCH_SIZE:-8}
OFFLOAD=${OFFLOAD:-False}

# Algorithm
adv_estimator=grpo
use_kl_in_reward=False

# Data Config
train_files=./traindata/train.jsonl
val_files=./traindata/valid.jsonl
max_prompt_length=512  
max_response_length=512   
filter_overlong_prompts=True
data_truncation='error'
model_path=Qwen3-8B

# Experience Config
experience_enable=False
experience_embedding_model=jina-embeddings-v2-base-zh
experience_file_path=./experience/${experiment_name}.jsonl
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
interaction_config_path=verl/examples/sglang_multiturn/config/interaction_config/doctor_interaction_config.yaml
tool_config_path=verl/examples/sglang_multiturn/config/tool_config/doctor_tool_config.yaml
max_user_turns=1
max_assistant_turns=1

# Actor / Rollout (Policy Model)
actor_optim_lr=1e-6
actor_use_remove_padding=False
actor_use_kl_loss=False
actor_enable_gradient_checkpointing=True
actor_enable_activation_offloading=True
actor_tensor_model_parallel_size=1
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
critic_enable_gradient_checkpointing=True 
critic_ppo_micro_batch_size_per_gpu=16 
critic_fsdp_config_param_offload=False
critic_fsdp_config_optimizer_offload=False
critic_warmup=0

# Reward
reward_manager=parallel

# logger
logger=['console','wandb'] 
project_name='verl_doctor_rl'
n_gpus_per_node=7
nnodes=1 
save_freq=4
test_freq=2
total_epochs=10


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

