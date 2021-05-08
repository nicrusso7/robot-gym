import numpy as np


def get_Rx(x):
    return np.asmatrix([[1, 0, 0, 0],
                        [0, np.cos(x), -np.sin(x), 0],
                        [0, np.sin(x), np.cos(x), 0],
                        [0, 0, 0, 1]])


def get_Ry(y):
    return np.asmatrix([[np.cos(y), 0, np.sin(y), 0],
                        [0, 1, 0, 0],
                        [-np.sin(y), 0, np.cos(y), 0],
                        [0, 0, 0, 1]])


def get_Rz(z):
    return np.asmatrix([[np.cos(z), -np.sin(z), 0, 0],
                        [np.sin(z), np.cos(z), 0, 0],
                        [0, 0, 1, 0],
                        [0, 0, 0, 1]])


def get_Rxyz(x, y, z):
    if x != 0 or y != 0 or z != 0:
        R = get_Rx(x) * get_Ry(y) * get_Rz(z)
        return R
    else:
        return np.identity(4)


def get_RT(orientation, position):
    roll = orientation[0]
    pitch = orientation[1]
    yaw = orientation[2]
    x0 = position[0]
    y0 = position[1]
    z0 = position[2]

    translation = np.asmatrix([[1, 0, 0, x0],
                               [0, 1, 0, y0],
                               [0, 0, 1, z0],
                               [0, 0, 0, 1]])
    rotation = get_Rxyz(roll, pitch, yaw)
    return rotation * translation


def transform(coord, rotation, translation):
    vector = np.array([[coord[0]],
                       [coord[1]],
                       [coord[2]],
                       [1]])

    transform_vector = get_RT(rotation, translation) * vector
    return np.array([transform_vector[0, 0], transform_vector[1, 0], transform_vector[2, 0]])


def check_domain(domain):
    if domain > 1 or domain < -1:
        if domain > 1:
            domain = 0.99
        else:
            domain = -0.99
    return domain


def solve_IK(coord, hip, leg, foot, right_side):
    domain = (coord[1] ** 2 + (-coord[2]) ** 2 - hip ** 2 + (-coord[0]) ** 2 - leg ** 2 - foot ** 2) / (
            2 * foot * leg)
    domain = check_domain(domain)
    gamma = np.arctan2(-np.sqrt(1 - domain ** 2), domain)
    sqrt_value = coord[1] ** 2 + (-coord[2]) ** 2 - hip ** 2
    if sqrt_value < 0.0:
        sqrt_value = 0.0
    alpha = np.arctan2(-coord[0], np.sqrt(sqrt_value)) - np.arctan2(foot * np.sin(gamma),
                                                                    leg + foot * np.cos(gamma))
    hip_val = hip
    if right_side:
        hip_val = -hip
    theta = -np.arctan2(coord[2], coord[1]) - np.arctan2(np.sqrt(sqrt_value), hip_val)
    angles = np.array([theta, alpha, gamma])
    return angles
