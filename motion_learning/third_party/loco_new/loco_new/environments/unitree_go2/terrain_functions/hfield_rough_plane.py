import numpy as np
import scipy.interpolate
import mujoco

from loco_new.environments import observation_indices as obs_idx


class HfieldRoughPlaneTerrainGeneration:
    def __init__(self, env,
                 inner_platform_size=3.0,
                 random_min_height=-0.05, random_max_height=0.05, random_step=0.005, random_downsampled_scale=0.2,
                 measured_points_x=[-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                 measured_points_y=[-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]
                ):
        self.env = env
        self.inner_platform_size = inner_platform_size
        self.random_min_height = random_min_height
        self.random_max_height = random_max_height
        self.random_step = random_step
        self.random_downsampled_scale = random_downsampled_scale
        self.measured_points_x = measured_points_x
        self.measured_points_y = measured_points_y

        self.xml_file_name = "hfield.xml"
        self.hfield_length = 320
        self.hfield_half_length_in_meters = 4
        self.max_possible_height = 30.0
        self.center_height = 0.0
        self.robot_height_over_ground = self.env.initial_drop_height

        self.one_meter_length = int(self.hfield_length / (self.hfield_half_length_in_meters * 2))
        self.hfield_half_length = self.hfield_length // 2
        self.mujoco_height_scaling = self.max_possible_height
        self.base_height_field = np.zeros((self.hfield_length, self.hfield_length))

        grid_x, grid_y = np.meshgrid(np.array(self.measured_points_x), np.array(self.measured_points_y), indexing='ij')
        self.nr_sampled_heights = np.prod(grid_x.shape)
        self.base_height_points = np.zeros((self.nr_sampled_heights, 2))
        self.base_height_points[:, 0] = grid_x.flatten()
        self.base_height_points[:, 1] = grid_y.flatten()

        obs_idx.update_nr_height_samples(self.nr_sampled_heights)


    def get_height_samples(self):
        global_height_points = self.base_height_points.copy()
        rotation_angle = self.env.orientation_euler[2]
        global_height_points[:, 0] = self.base_height_points[:, 0] * np.cos(rotation_angle) - self.base_height_points[:, 1] * np.sin(rotation_angle)
        global_height_points[:, 1] = self.base_height_points[:, 0] * np.sin(rotation_angle) + self.base_height_points[:, 1] * np.cos(rotation_angle) 
        global_height_points += self.env.data.qpos[:2]

        global_height_points = (global_height_points * self.one_meter_length).astype(np.int32)
        px = global_height_points[:, 0].reshape(-1)
        py = global_height_points[:, 1].reshape(-1)
        px += self.hfield_half_length
        py += self.hfield_half_length
        px = np.clip(px, 0, self.hfield_length-2)
        py = np.clip(py, 0, self.hfield_length-2)

        heights1 = self.current_height_field_data[py, px]
        heights2 = self.current_height_field_data[py+1, px]
        heights3 = self.current_height_field_data[py, px+1]
        heights = np.minimum(heights1, heights2)
        heights = np.minimum(heights, heights3)

        self.sampled_heights = self.env.data.qpos[2] - heights * self.mujoco_height_scaling
        self.robot_height_over_ground = self.sampled_heights[self.nr_sampled_heights // 2]

        return np.clip(self.sampled_heights, -self.max_possible_height, self.max_possible_height)


    def step(self, obs, reward, absorbing, info):
        if 3.5 < np.abs(self.env.data.qpos[0]) < 4.0 or 3.5 < np.abs(self.env.data.qpos[1]) < 4.0:
            self.env.data.qpos, self.env.data.qvel = self.env.initial_state_function.setup()
    

    def info(self, info):
        return info
    

    def sample(self):
        isaac_height_field = self.random_uniform_terrain()
        new_height_field_data = self.isaac_hf_to_mujoco_hf(isaac_height_field)

        if np.max(new_height_field_data) >= 0.9999 or np.min(new_height_field_data) < 0.0:
            raise ValueError(f"Invalid height field. Max height: {np.max(new_height_field_data)}. Min height: {np.min(new_height_field_data)}.")

        self.env.model.hfield_data = new_height_field_data
        self.current_height_field_data = new_height_field_data.reshape(self.hfield_length, self.hfield_length)
        self.center_height = new_height_field_data[self.hfield_half_length * self.hfield_length + self.hfield_half_length] * self.mujoco_height_scaling
        if self.env.viewer:
            mujoco.mjr_uploadHField(self.env.viewer.model, self.env.viewer.context, 0)
    

    def isaac_hf_to_mujoco_hf(self, isaac_hf):
        hf = isaac_hf + np.abs(np.min(isaac_hf))
        hf /= self.mujoco_height_scaling
        return hf.reshape(-1)


    def random_uniform_terrain(self):
        min_height = self.random_min_height
        max_height = self.random_max_height
        step = self.random_step
        heights_range = np.arange(min_height, max_height + step, step)
        add_height_field_downsampled = np.random.choice(heights_range, size=(int(self.hfield_length * self.random_downsampled_scale), int(self.hfield_length * self.random_downsampled_scale)))
        x = np.linspace(0, self.hfield_length, int(self.hfield_length * self.random_downsampled_scale))
        y = np.linspace(0, self.hfield_length, int(self.hfield_length * self.random_downsampled_scale))
        interpolator = scipy.interpolate.RegularGridInterpolator((x, y), add_height_field_downsampled, method='linear')
        x_upsampled = np.linspace(0, self.hfield_length, self.hfield_length)
        y_upsampled = np.linspace(0, self.hfield_length, self.hfield_length)
        x_upsampled_grid, y_upsampled_grid = np.meshgrid(x_upsampled, y_upsampled, indexing='ij')
        points = np.array([x_upsampled_grid.ravel(), y_upsampled_grid.ravel()]).T
        add_height_field = interpolator(points).reshape((self.hfield_length, self.hfield_length))
        platform_size = int(self.inner_platform_size * self.one_meter_length)
        x1 = self.hfield_half_length - (platform_size // 6)
        y1 = self.hfield_half_length - (platform_size // 6)
        x2 = self.hfield_half_length + (platform_size // 6)
        y2 = self.hfield_half_length + (platform_size // 6)
        add_height_field[x1:x2, y1:y2] = 0
        height_field_raw = self.base_height_field + add_height_field
        return height_field_raw
