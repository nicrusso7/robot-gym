import os

import tensorflow as tf
from absl import logging
import gin
from gibson2.utils.utils import parse_config

from robot_gym.agents.stanford.sac import wrapper
from robot_gym.agents.stanford.sac.stanford_sac_agent import train_eval


def train(log_dir, env_config):
    base_config = parse_config("base_config.yaml")
    tf.compat.v1.enable_resource_variables()
    logging.set_verbosity(logging.INFO)
    gin.parse_config_files_and_bindings(base_config["gin_file"], base_config["gin_param"])

    os.environ['CUDA_VISIBLE_DEVICES'] = str(base_config["gpu_c"])

    conv_1d_layer_params = [(32, 8, 4), (64, 4, 2), (64, 3, 1)]
    conv_2d_layer_params = [(32, (8, 8), 4), (64, (4, 4), 2), (64, (3, 3), 2)]
    encoder_fc_layers = [256]
    actor_fc_layers = [256]
    critic_obs_fc_layers = [256]
    critic_action_fc_layers = [256]
    critic_joint_fc_layers = [256]

    for k, v in base_config.items():
        print(k, v)
    print('conv_1d_layer_params', conv_1d_layer_params)
    print('conv_2d_layer_params', conv_2d_layer_params)
    print('encoder_fc_layers', encoder_fc_layers)
    print('actor_fc_layers', actor_fc_layers)
    print('critic_obs_fc_layers', critic_obs_fc_layers)
    print('critic_action_fc_layers', critic_action_fc_layers)
    print('critic_joint_fc_layers', critic_joint_fc_layers)

    train_eval(
        root_dir=log_dir,
        gpu=base_config["gpu_g"],
        env_load_fn=lambda model_id, mode, device_idx: wrapper.load(
            config_file=env_config,
            model_id=model_id,
            env_mode=mode,
            action_timestep=base_config["action_timestep"],
            physics_timestep=base_config["physics_timestep"],
            device_idx=device_idx,
        ),
        model_ids=base_config["model_ids"],
        eval_env_mode=base_config["env_mode"],
        num_iterations=base_config["num_iterations"],
        conv_1d_layer_params=conv_1d_layer_params,
        conv_2d_layer_params=conv_2d_layer_params,
        encoder_fc_layers=encoder_fc_layers,
        actor_fc_layers=actor_fc_layers,
        critic_obs_fc_layers=critic_obs_fc_layers,
        critic_action_fc_layers=critic_action_fc_layers,
        critic_joint_fc_layers=critic_joint_fc_layers,
        initial_collect_steps=base_config["initial_collect_steps"],
        collect_steps_per_iteration=base_config["collect_steps_per_iteration"],
        num_parallel_environments=base_config["num_parallel_environments"],
        replay_buffer_capacity=base_config["replay_buffer_capacity"],
        train_steps_per_iteration=base_config["train_steps_per_iteration"],
        batch_size=base_config["batch_size"],
        actor_learning_rate=base_config["actor_learning_rate"],
        critic_learning_rate=base_config["critic_learning_rate"],
        alpha_learning_rate=base_config["alpha_learning_rate"],
        gamma=base_config["gamma"],
        num_eval_episodes=base_config["num_eval_episodes"],
        eval_interval=base_config["eval_interval"],
        eval_only=base_config["eval_only"],
        num_parallel_environments_eval=base_config["num_parallel_environments_eval"],
        model_ids_eval=base_config["model_ids_eval"],
    )
