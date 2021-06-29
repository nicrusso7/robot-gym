import os

import tensorflow as tf
from absl import logging
import gin
from gibson2.utils.utils import parse_config
from robot_gym.gym.envs.reach import reach_env

from robot_gym.gym.envs import reach

from robot_gym.controllers.mpc import mpc_controller
from robot_gym.model.robots.ghost import ghost

from robot_gym.agents.stanford.sac import wrapper
from robot_gym.agents.stanford.sac.stanford_sac_agent import train_eval


def train(log_dir,
          env_class,
          robot_model,
          mark,
          controller_class,
          config,
          render,
          debug,
          policy):
    base_config = parse_config("base_config.yaml")
    tf.compat.v1.enable_resource_variables()
    logging.set_verbosity(logging.INFO)
    # TODO read gin files (if any) from config
    gin.parse_config_files_and_bindings(None, None)

    os.environ['CUDA_VISIBLE_DEVICES'] = str(base_config["gpu_c"])
    os.environ['NUMEXPR_MAX_THREADS'] = "20"

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
        env_class=env_class,
        robot_model=robot_model,
        mark=mark,
        controller_class=controller_class,
        config=config,
        render=render,
        debug=debug,
        policy=policy,
        # TODO gpu index configuration parsing from base conf.
        # gpu=base_config["gpu_g"],
        env_load_fn=lambda _env_cls, _robot_model, _mark, _ctrl_class, _config, _render, _debug, _policy: wrapper.load(
            _env_cls,
            _robot_model,
            _mark,
            _ctrl_class,
            _config,
            _render,
            _debug,
            _policy
        ),
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
        num_parallel_environments_eval=base_config["num_parallel_environments_eval"]
    )


if __name__ == '__main__':
    train("~/dev/robotics/playground/sac", reach_env.ReachEnv, ghost.Ghost, "1", mpc_controller.MPCController,
          os.path.join(os.path.dirname(reach.__file__), "reach.yaml"), True, True, True)
