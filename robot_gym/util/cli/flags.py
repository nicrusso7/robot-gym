ENV_ID_TO_POLICY = {
    'go_mpc': ('robot_gym/policies/go_to/ppo', 'model.ckpt-5250000')
}

ENV_ID_TO_ENV = {
    'go': 'GoEnv'
}

TERRAIN_TYPE = {
    'valley': 'png',
    'maze': 'png',
    'hills': 'csv',
    'random': 'random',
    'plane': 'plane'
}

SUPPORTED_CONTROLLERS = ['mpc']

SUPPORTED_ROBOTS = ['ghost', 'k3lso']

SUPPORTED_AGENTS = ['ppo', 'ddpg']
