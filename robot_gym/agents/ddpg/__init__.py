import gym
from gym.envs.registration import registry


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


register(
    id='GoTo-v0',
    entry_point='robot_gym.gym.envs.go_to.go_env:GoEnv',
    reward_threshold=700,
)
