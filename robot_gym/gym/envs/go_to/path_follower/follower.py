from robot_gym.gym.envs.go_to.path_follower.geometry_ref import CameraWindow, ReferencePoint


class Follower:
    def __init__(self, nb_cam_points, start_xy, start_yaw):
        self.prev_pos = ((0., 0.), 0.)
        self.pos = ((0., 0.), 0.)

        self.prev_vel = ((0., 0.), 0.)
        self.vel = ((0., 0.), 0.)

        self.cam_window: CameraWindow = None
        self.num_cam_pts = nb_cam_points

        self.track_ref_point: ReferencePoint = None
        self.cam_target_point: ReferencePoint = None  # POV Camera target point
        self.cam_pos_point: ReferencePoint = None  # POV Camera position
        # maximum distance from the line before signaling episode done
        self.max_track_err = 0.1

        self._position_on_track = 0.

        self.reset(start_xy, start_yaw)

    def reward(self, path):
        reward = 0.
        # Path distance error
        track_err = path.distance_from_point(self.pos[0])
        track_err_norm = track_err * (1.0 / self.max_track_err)

        self._position_on_track += path.length_along_track(self.prev_pos[0], self.pos[0])

        # Track progress
        checkpoint_reward = 1000. / path.num_checkpoints
        if self._position_on_track - path.progress < 0.4:
            checkpoints_reached = path.update_progress(self._position_on_track)
            reward += checkpoints_reached * checkpoint_reward * (1.0 - track_err_norm) ** 2

        # Time penalty
        reward -= 0.15

        if abs(self._position_on_track - path.progress) > 0.5:
            # progress distance limit penalty
            reward = -100
        elif track_err > self.max_track_err:
            # path distance penalty
            reward = -100.

        return reward

    def reset(self, xy, yaw):
        h = 0.160  # camera_window_height
        wt = 0.270  # camera_window_top_width
        wb = 0.120  # camera_window_bottom_width
        d = 0.112  # camera_window_distance
        win_points = [(d + h, wt / 2), (d + h, -wt / 2), (d, -wb / 2), (d, wb / 2)]

        self.cam_window = CameraWindow(win_points)
        self.cam_window.move(xy, yaw)

        tref_pt_x = 0.112  # track_ref_point_x
        self.track_ref_point = ReferencePoint(xy_shift=(tref_pt_x, 0.))
        self.track_ref_point.move(xy, yaw)

        cam_target_pt_x = 0.185  # camera_target_point_x
        self.cam_target_point = ReferencePoint(xy_shift=(cam_target_pt_x, 0.))
        self.cam_target_point.move(xy, yaw)

        cam_pos_pt_x = 0.060  # camera_position_point_x
        self.cam_pos_point = ReferencePoint(xy_shift=(cam_pos_pt_x, 0.))
        self.cam_pos_point.move(xy, yaw)

    def update_position_velocity(self, current_pos, current_orient, robot_id, pybullet_client):
        new_xy, new_yaw = self.format_position(current_pos, current_orient)
        self.cam_window.move(new_xy, new_yaw)
        self.track_ref_point.move(new_xy, new_yaw)
        self.cam_target_point.move(new_xy, new_yaw)
        self.cam_pos_point.move(new_xy, new_yaw)
        self.prev_pos = self.pos
        self.prev_vel = self.vel
        self.pos = new_xy, new_yaw
        self.vel = self.get_velocity(pybullet_client, robot_id)

    @staticmethod
    def format_position(position, orientation):
        x, y, _ = position
        _, _, yaw = orientation
        return (x, y), yaw

    @staticmethod
    def get_velocity(pb_client, robot):
        linear, angular = pb_client.getBaseVelocity(robot)
        vx, vy, _ = linear
        _, _, wz = angular
        return (vx, vy), wz

    def get_info(self):
        (x, y), yaw = self.pos
        return {"x": x,
                "y": y,
                "yaw": yaw}
