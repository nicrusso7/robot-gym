import gym
from gym.envs.registration import registry


def register(id, *args, **kvargs):
    if id in registry.env_specs:
        return
    else:
        return gym.envs.registration.register(id, *args, **kvargs)


def getList():
    btenvs = ['- ' + spec.id for spec in gym.envs.registry.all() if spec.id.find('Bullet') >= 0]
    return btenvs


register(
    id='GoTo-v0',
    entry_point='robot_gym.gym.envs.go_to.go_env:GoEnv',
    max_episode_steps=1500,
    reward_threshold=5.0,
)

register(
    id='Gallop-v0',
    entry_point='robot_gym.gym.envs.gallop_env:GallopEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='TurnOnSpot-v0',
    entry_point='robot_gym.gym.envs.turn_env:TurnEnv',
    max_episode_steps=1000,
    reward_threshold=5.0,
)

register(
    id='Standup-v0',
    entry_point='robot_gym.gym.envs.standup_env:StandupEnv',
    max_episode_steps=500,
    reward_threshold=5.0,
)

register(
    id='RexPoses-v0',
    entry_point='robot_gym.gym.envs.poses_env:PosesEnv',
    max_episode_steps=500,
    reward_threshold=5.0,
)
