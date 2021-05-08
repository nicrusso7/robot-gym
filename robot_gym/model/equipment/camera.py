import pybullet as p


def parse_cams(marks, mark, equip):
    for cam in marks.MARK_PARAMS[mark]["hardware"]["camera"]["cams"]:
        if "cams" in equip:
            equip["cams"].append(Camera(cam["name"], cam["position"], cam["target"]))
        else:
            equip["cams"] = [Camera(cam["name"], cam["position"], cam["target"])]
    equip["default_cam"] = marks.MARK_PARAMS[mark]["hardware"]["camera"]["default"]
    return equip


class Camera:

    def __init__(self, name, position, target):
        self._name = name
        self._position = position
        self._target = target

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, value):
        self._target = value

    def get_camera_image(self, pb_client):
        """
        Render virtual camera image.
        :return: Camera image
        """
        cam_x, cam_y, cam_z = self._position
        target_x, target_y, target_z = self._target
        vm = pb_client.computeViewMatrix(cameraEyePosition=[cam_x, cam_y, cam_z],
                                         cameraTargetPosition=[target_x, target_y, target_z],
                                         cameraUpVector=[0.0, 0.0, 1.0])
        pm = pb_client.computeProjectionMatrixFOV(fov=49,
                                                  aspect=320 / 240,
                                                  nearVal=0.0001,
                                                  farVal=1)
        w, h, rgb, deth, seg = pb_client.getCameraImage(width=320,
                                                        height=240,
                                                        viewMatrix=vm,
                                                        projectionMatrix=pm,
                                                        renderer=p.ER_BULLET_HARDWARE_OPENGL)
        # RGB Array shape (240, 320, 3)
        # rgb = np.array(rgb)
        # rgb = rgb[:, :, :3]
        return w, h, rgb, deth, seg
