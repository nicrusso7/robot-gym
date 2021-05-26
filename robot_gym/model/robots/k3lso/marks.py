MARK_LIST = ['1']

MARK_PARAMS = {
    '1': {
        'num_motors': 12,
        'num_legs': 4,
        'urdf_name': "robots/k3lso.urdf",
        'motor_names': [
            "FR_hip_joint",
            "FR_upper_joint",
            "FR_lower_joint",
            "FL_hip_joint",
            "FL_upper_joint",
            "FL_lower_joint",
            "RR_hip_joint",
            "RR_upper_joint",
            "RR_lower_joint",
            "RL_hip_joint",
            "RL_upper_joint",
            "RL_lower_joint",
        ],
        'hardware': {
            'camera': {
                'default': 0,
                'cams': [
                    {
                        "name": "front",
                        "position": (0., 0., 0.25),
                        "target": (0.5, 0., 0.)
                    }
                ]
            }
        }
    }
}
