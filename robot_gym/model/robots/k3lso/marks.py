MARK_LIST = ['1']

MARK_PARAMS = {
    '1': {
        'num_motors': 12,
        'num_legs': 4,
        'urdf_name': "robots/k3lso.urdf",
        'motor_names': [
            "torso_to_abduct_fr_j",
            "abduct_fr_to_thigh_fr_j",
            "thigh_fr_to_knee_fr_j",
            "torso_to_abduct_fl_j",
            "abduct_fl_to_thigh_fl_j",
            "thigh_fl_to_knee_fl_j",
            "torso_to_abduct_hr_j",
            "abduct_hr_to_thigh_hr_j",
            "thigh_hr_to_knee_hr_j",
            "torso_to_abduct_hl_j",
            "abduct_hl_to_thigh_hl_j",
            "thigh_hl_to_knee_hl_j",
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
