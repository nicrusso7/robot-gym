import pybullet_data as pd
import robot_gym.util.pybullet_data as rpd
import pybullet as p
import random

from robot_gym.util.cli.flags import TERRAIN_TYPE


ID_TO_FILENAME = {
    'valley': "heightmaps/wm_height_out.png",
    'maze': "heightmaps/Maze.png"
}

ROBOT_INIT_POSITION_OFFSET = {
    'png_valley': .64,
    'plane': .0,
    'csv_hills': 1.77,
    'png_maze': .0,
    'random': .0
}


class Terrain:

    def __init__(self, terrain_type, terrain_id, columns=256, rows=256):
        self.terrain_type = terrain_type
        self.terrain_id = terrain_id
        self.columns = columns
        self.rows = rows
        self.terrain_shape = None
        self.id = None

    def generate_terrain(self, pybullet_client, height_perturbation_range=0.06):
        pybullet_client.setAdditionalSearchPath(pd.getDataPath())
        pybullet_client.configureDebugVisualizer(pybullet_client.COV_ENABLE_RENDERING, 0)
        # if self.id is not None:
        #     pybullet_client.removeBody(self.id)

        if self.terrain_type == 'plane':
            self.id = pybullet_client.loadURDF("plane.urdf")
            pybullet_client.changeVisualShape(self.id, -1, rgbaColor=[1, 1, 1, 1])
            # planeShape = p.createCollisionShape(shapeType = p.GEOM_PLANE)
            # ground_id  = p.createMultiBody(0, planeShape)
        else:
            if self.terrain_type == 'random':
                terrain_data = [0] * self.columns * self.rows
                for j in range(int(self.columns / 2)):
                    for i in range(int(self.rows / 2)):
                        height = random.uniform(0, height_perturbation_range)
                        terrain_data[2 * i + 2 * j * self.rows] = height
                        terrain_data[2 * i + 1 + 2 * j * self.rows] = height
                        terrain_data[2 * i + (2 * j + 1) * self.rows] = height
                        terrain_data[2 * i + 1 + (2 * j + 1) * self.rows] = height
                terrain_shape = pybullet_client.createCollisionShape(
                    shapeType=pybullet_client.GEOM_HEIGHTFIELD,
                    meshScale=[.05, .05, 1],
                    heightfieldTextureScaling=(self.rows - 1) / 2,
                    heightfieldData=terrain_data,
                    numHeightfieldRows=self.rows,
                    numHeightfieldColumns=self.columns)
                terrain = pybullet_client.createMultiBody(0, terrain_shape)
                pybullet_client.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])

            elif self.terrain_type == 'csv':
                terrain_shape = pybullet_client.createCollisionShape(
                    shapeType=pybullet_client.GEOM_HEIGHTFIELD,
                    meshScale=[.5, .5, .5],
                    fileName="heightmaps/ground0.txt",
                    heightfieldTextureScaling=128)
                terrain = pybullet_client.createMultiBody(0, terrain_shape)
                textureId = pybullet_client.loadTexture(f"{rpd.getDataPath()}/world/terrains/grass.png")
                pybullet_client.changeVisualShape(terrain, -1, textureUniqueId=textureId)
                pybullet_client.resetBasePositionAndOrientation(terrain, [1, 0, 2], [0, 0, 0, 1])

            elif self.terrain_type == 'png':
                terrain_shape = pybullet_client.createCollisionShape(
                    shapeType=pybullet_client.GEOM_HEIGHTFIELD,
                    meshScale=[.1, .1, 24 if self.terrain_id == "valley" else 1],
                    fileName=ID_TO_FILENAME[self.terrain_id])
                terrain = pybullet_client.createMultiBody(0, terrain_shape)
                if self.terrain_id == "valley":
                    textureId = pybullet_client.loadTexture("heightmaps/gimp_overlay_out.png")
                    pybullet_client.changeVisualShape(terrain, -1, textureUniqueId=textureId)
                    pybullet_client.resetBasePositionAndOrientation(terrain, [0, 0, 2], [0, 0, 0, 1])
                else:
                    pybullet_client.resetBasePositionAndOrientation(terrain, [0, 0, 0], [0, 0, 0, 1])
                pybullet_client.changeVisualShape(terrain, -1, rgbaColor=[1, 1, 1, 1])
            else:
                raise Exception(f"{self.terrain_type} is not a valid terrain type!")

            self.terrain_shape = terrain_shape
            self.id = terrain
            pybullet_client.configureDebugVisualizer(pybullet_client.COV_ENABLE_RENDERING, 1)

    def update_terrain(self, height_perturbation_range=0.05):
        if self.terrain_type == TERRAIN_TYPE['random']:
            terrain_data = [0] * self.columns * self.rows
            for j in range(int(self.columns / 2)):
                for i in range(int(self.rows / 2)):
                    height = random.uniform(0, height_perturbation_range)
                    terrain_data[2 * i + 2 * j * self.rows] = height
                    terrain_data[2 * i + 1 + 2 * j * self.rows] = height
                    terrain_data[2 * i + (2 * j + 1) * self.rows] = height
                    terrain_data[2 * i + 1 + (2 * j + 1) * self.rows] = height
            # GEOM_CONCAVE_INTERNAL_EDGE may help avoid getting stuck at an internal (shared) edge of
            # the triangle/heightfield. GEOM_CONCAVE_INTERNAL_EDGE is a bit slower to build though.
            flags = p.GEOM_CONCAVE_INTERNAL_EDGE
            # flags = 0
            self.terrain_shape = p.createCollisionShape(
                shapeType=p.GEOM_HEIGHTFIELD,
                flags=flags,
                meshScale=[.05, .05, 1],
                heightfieldTextureScaling=(self.rows - 1) / 2,
                heightfieldData=terrain_data,
                numHeightfieldRows=self.rows,
                numHeightfieldColumns=self.columns,
                replaceHeightfieldIndex=self.terrain_shape)

    @staticmethod
    def setup_ui_params(pybullet_client):
        plane = pybullet_client.addUserDebugParameter("Plane", 0, -1, 0)
        heightfield = pybullet_client.addUserDebugParameter("Heightfield", 0, -1, 0)
        # update_heightfield = pybullet_client.addUserDebugParameter("Change heightfield", 0, -1, 0)
        # maze = pybullet_client.addUserDebugParameter("Maze", 0, -1, 0)
        valley = pybullet_client.addUserDebugParameter("Valley", 0, -1, 0)
        hills = pybullet_client.addUserDebugParameter("Hills", 0, -1, 0)
        ui = {
            "plane": plane,
            "heightfield": heightfield,
            # "update_heightfield": update_heightfield,
            # "maze": maze,
            "valley": valley,
            "hills": hills
        }
        return ui

    def parse_ui_input(self, ui, pybullet_client):
        reset_sim = False
        terrain_type = None
        terrain_id = None
        if pybullet_client.readUserDebugParameter(ui["plane"]):
            terrain_type = "plane"
            terrain_id = None
            reset_sim = True
        elif pybullet_client.readUserDebugParameter(ui["heightfield"]):
            terrain_type = "random"
            terrain_id = None
            reset_sim = True
        # elif pybullet_client.readUserDebugParameter(ui["update_heightfield"]):
        #     self.update_terrain()
        # elif pybullet_client.readUserDebugParameter(ui["maze"]):
        #     terrain_type = "png"
        #     terrain_id = "maze"
        #     reset_sim = True
        elif pybullet_client.readUserDebugParameter(ui["valley"]):
            terrain_type = "png"
            terrain_id = "valley"
            reset_sim = True
        elif pybullet_client.readUserDebugParameter(ui["hills"]):
            terrain_type = "csv"
            terrain_id = "hills"
            reset_sim = True
        else:
            terrain_type = self.terrain_type
            terrain_id = self.terrain_id
            reset_sim = False

        return reset_sim, terrain_id, terrain_type

    def get_terrain_z_offset(self):
        if self.terrain_type in ROBOT_INIT_POSITION_OFFSET.keys():
            return ROBOT_INIT_POSITION_OFFSET[self.terrain_type]
        return ROBOT_INIT_POSITION_OFFSET[f"{self.terrain_type}_{self.terrain_id}"]
