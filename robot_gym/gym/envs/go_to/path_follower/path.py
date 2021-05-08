"""
Credits: nplan

https://github.com/nplan/gym-line-follower
"""
import json
import random

from shapely.geometry import LineString, MultiPoint, Point
import numpy as np
from shapely.ops import nearest_points

from robot_gym.gym.envs.go_to.path_follower.line_interpolation import interpolate_points
from robot_gym.gym.envs.go_to.path_planner import potential_field_planner


class Path:
    """
    The robot follows a Path instance. This class contains methods for randomly generating, rendering and
    calculating relative follower distance, speed and direction.
    """

    def __init__(self, pts, num_checkpoints=100, render_params=None, debug=False):
        l = LineString(pts).length
        n = int(l / 1e-2)  # Get number of points for 1 cm spacing
        if debug:
            print("Path points:")
            print(pts)
        self.pts = interpolate_points(np.array(pts), n)  # interpolate points to get the right spacing
        # print("Path interpolate points:")
        # print(self.pts)
        self.x = self.pts[:, 0]
        self.y = self.pts[:, 1]

        self.render_params = render_params

        self.mpt = MultiPoint(self.pts)
        self.string = LineString(self.pts)

        # Find starting point and angle
        self.start_xy = self.x[0], self.y[0]
        self.start_angle = self.angle_at_index(0)

        # Get length
        self.length = self.string.length

        # Progress tracking setup
        self.progress = 0.
        self.progress_idx = 0
        self.num_checkpoints = num_checkpoints
        self.checkpoints = [i * (self.length / self.num_checkpoints) for i in range(1, self.num_checkpoints + 1)]
        self.next_checkpoint_idx = 0
        self.done = False

    @classmethod
    def generate(cls, *args, **kwargs):
        """
        Generate random path.
        :return: Path instance
        """
        # randomize target
        x = round(random.uniform(-1.5, 1.5), 2)
        y = round(random.uniform(-1.5, 1.5), 2)
        print(f"Target: {x},{y}")

        pts = potential_field_planner.get_path(x, y, [], [])
        return cls(pts, *args, **kwargs)

    @classmethod
    def from_file(cls, path, *args, **kwargs):
        with open(path, "r") as f:
            d = json.load(f)
        points = d["points"]
        points.append(points[0])  # Close the loop
        points = interpolate_points(points, 1000)
        return cls(points, *args, **kwargs)

    def _render(self, w=3., h=2., ppm=1500, line_thickness=0.015, save=None, line_color="black", line_opacity=0.8):
        """
        Render track using open-cv
        :param w: canvas width in meters
        :param h: canvas height in meters
        :param ppm: pixel per meter
        :param line_thickness: line thickness in meters
        :param save: path to save
        :param line_color: string or BGR tuple
                           options: [black, red, green, blue]
        :param background: string or BGR tuple
                           options: [wood, wood_2, concrete, brick, checkerboard, white, gray]
        :param line_opacity: opacity of line in range 0, 1 where 0 is fully transparent
        :return: rendered track image array
        """
        import cv2
        w_res = int(round(w * ppm))
        h_res = int(round(h * ppm))

        background_bgr = (255, 255, 255)
        if background_bgr:
            bg = np.ones((h_res, w_res, 3), dtype=np.uint8)
            bg[:, :, 0] *= background_bgr[0]
            bg[:, :, 1] *= background_bgr[1]
            bg[:, :, 2] *= background_bgr[2]

        h_res = int(round(h * ppm))
        t_res = int(round(line_thickness * ppm))

        if isinstance(line_color, str):
            line_color = line_color.lower()
            if line_color == "black":
                line_bgr = (0, 0, 0)
            elif line_color == "red":
                line_bgr = (0, 0, 255)
            elif line_color == "green":
                line_bgr = (0, 128, 0)
            elif line_color == "blue":
                line_bgr = (255, 0, 0)
            else:
                raise ValueError("Invalid color string.")
        elif isinstance(line_color, tuple):
            line_bgr = line_color
        else:
            raise ValueError("Invalid line_color.")

        line = bg.copy()

        for i in range(len(self.pts) - 1):
            x1, y1 = self.pts[i]
            x1_img = int(round((x1 + w / 2) * ppm, ndigits=0))
            y1_img = int(round(h_res - (y1 + h / 2) * ppm, ndigits=0))

            x2, y2 = self.pts[i + 1]
            x2_img = int(round((x2 + w / 2) * ppm, ndigits=0))
            y2_img = int(round(h_res - (y2 + h / 2) * ppm, ndigits=0))

            cv2.line(line, (x1_img, y1_img), (x2_img, y2_img), color=line_bgr, thickness=t_res,
                     lineType=cv2.LINE_AA)

        alpha = line_opacity
        out = cv2.addWeighted(line, alpha, bg, 1 - alpha, 0)

        if save is not None:
            cv2.imwrite(save, out)
        return out

    def render(self, *args, **kwargs):
        if self.render_params:
            return self._render(*args, **kwargs, **self.render_params)
        else:
            return self._render(*args, **kwargs)

    def distance_from_point(self, pt):
        """
        Calculate minimal distance of a position from track.
        :param pt: position. [x, y] or shapely.geometry.Point instance
        :return: minimal absolute distance to track, float
        """
        if not isinstance(pt, Point):
            pt = Point(pt)
        return pt.distance(self.mpt)

    def vector_at_index(self, idx):
        """
        Return normalized track direction vector at desired index.
        :param idx: index of track point
        :return: unit direction vector
        """
        x, y = self.x, self.y

        # Handle indexing last track point
        if idx < len(self.pts) - 2:
            vect = np.array([x[idx + 1] - x[idx], y[idx + 1] - y[idx]])
        else:
            vect = np.array([x[0] - x[idx], y[0] - y[idx]])

        # Find track angle
        norm = np.linalg.norm(vect)
        vect = (vect / norm) if norm > 0.0 else np.array([1., 0])  # normalize vector to unit length
        return vect

    def angle_at_index(self, idx):
        """
        Calculate track angle at desired index. Angle is calculated from x-axis, CCW is positive. Angle is returned in
        radians in range [0, 2pi]
        :param idx: index of track point
        :return: angle in radians, range [0, 2pi]
        """
        vect = self.vector_at_index(idx)
        x_vect = np.array([1, 0])
        dot = np.dot(vect, x_vect)
        det = np.linalg.det([x_vect, vect])
        track_ang = np.arctan2(det, dot)
        if track_ang < 0.:
            track_ang += 2 * np.pi
        return track_ang

    def nearest_point(self, pt):
        """
        Determine point on track that is nearest to provided point.
        :param pt: point to search nearest track point for, Point instance or coordinate array [x, y]
        :return: nearest track point coordinates [x, y]
        """
        if not isinstance(pt, Point):
            pt = Point(pt)
        nearest = nearest_points(pt, self.mpt)[1]
        return nearest.x, nearest.y

    def nearest_angle(self, pt):
        """
        Calculate track angle at the point on track nearest to provided point-
        :param pt: point to search nearest track point for, Point instance or coordinate array [x, y]
        :return: angle, float
        """
        near_x, near_y = self.nearest_point(pt)
        near_idx = np.where(self.x == near_x)[0][0]
        return self.angle_at_index(near_idx)

    def nearest_vector(self, pt):
        """
        Calculate track angle at the point on track nearest to provided point.
        :param pt: point to search nearest track point for, Point instance or coordinate array [x, y]
        :return: unit track direction vector
        """
        near_x, near_y = self.nearest_point(pt)
        near_idx = np.where(self.x == near_x)[0][0]
        return self.vector_at_index(near_idx)

    def length_between_idx(self, idx1, idx2, shortest=True):
        """
        Calculate length of track segment between two point indexes. Direction is determined based on index order.
        :param idx1: first index
        :param idx2: second index
        :param shortest: True to return shortest path, False to return longest
        :return: segment length, float, positive or negative based on direction
        """
        if idx1 == idx2:
            return 0.
        if idx1 < idx2:
            first = idx1
            second = idx2
        else:
            first = idx2
            second = idx1
        string_1 = LineString(self.pts[first:second + 1])
        string_2 = LineString(np.concatenate((self.pts[0:first + 1], self.pts[second:])))
        len_1 = string_1.length
        len_2 = string_2.length

        if len_1 < len_2:
            if idx1 < idx2:
                if shortest:
                    return len_1
                else:
                    return -len_2
            else:
                if shortest:
                    return -len_1
                else:
                    return len_2
        else:
            if idx1 < idx2:
                if shortest:
                    return -len_2
                else:
                    return len_1
            else:
                if shortest:
                    return len_2
                else:
                    return -len_1

    def length_along_track(self, pt1, pt2):
        """
        Calculate length along track between two points near to track. Returns the shortest possible path.
        Order of argument points is arbitrary.
        :param pt1: first point
        :param pt2: second point
        :return: length, float, positive if in direction of track, negative otherwise
        """
        near_1 = self.nearest_point(pt1)
        near_2 = self.nearest_point(pt2)

        idx_1 = np.where(self.x == near_1[0])[0][0]
        idx_2 = np.where(self.x == near_2[0])[0][0]
        return self.length_between_idx(idx_1, idx_2, shortest=True)

    def position_along(self, pt):
        """
        Calculate position along track from start of track.
        :param pt:
        :return: position in range [0, track length]
        """
        near = self.nearest_point(pt)
        idx = np.where(self.x == near[0])[0][0]
        return (idx / len(self.pts)) * self.length

    def update_progress(self, position):
        """
        Update track progress and return passed checkpoints.
        :param position: position along track in meters from starting point
        :return: number of checkpoints passed
        """
        if self.done:
            return 0
        if position > self.progress:
            self.progress = position
            self.progress_idx = int(round((self.progress / self.length) * len(self.pts)))
        ret = 0
        while self.progress >= self.checkpoints[self.next_checkpoint_idx]:
            self.next_checkpoint_idx += 1
            ret += 1
            if self.next_checkpoint_idx >= self.num_checkpoints - 1:
                self.done = True
                break
        return ret


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    # t = Track.generate(2.0, hw_ratio=0.7, seed=4125,
    #                    spikeyness=0.2, nb_checkpoints=500)

    # img = t.render()
    # plt.imshow(img)
    # plt.show()

    for i in range(9):
        t = Path.generate(2.0, hw_ratio=0.7, seed=None,
                          spikeyness=0.2, nb_checkpoints=500)
        img = t.render(ppm=1000)
        plt.subplot(3, 3, i + 1)
        plt.imshow(img)
        plt.axis("off")
    # plt.tight_layout()
    plt.savefig("track_generator.png", dpi=300)
    plt.show()
