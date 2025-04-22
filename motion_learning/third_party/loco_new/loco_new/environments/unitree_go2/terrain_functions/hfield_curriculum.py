from pathlib import Path
import numpy as np
import scipy.interpolate
import mujoco

from loco_new.environments import observation_indices as obs_idx


class HFieldCurriculumTerrainGeneration:
    def __init__(self, env,
                 terrain_type_probs=[0.05, 0.05, 0.1, 0.25, 0.35, 0.2], max_difficulty_level=9,
                 inner_platform_size=3, max_slope=0.4, base_step_height=0.05, additional_max_step_height=0.18,
                 random_min_height=-0.05, random_max_height=0.05, random_step=0.005, random_downsampled_scale=0.2,
                 stairs_step_width=0.31, obstacles_base_height=0.05, obstacles_additional_factor=0.2,
                 obstacles_nr_rectangles=20, obstacles_rectangle_min_size=1.0, obstacles_rectangle_max_size=2.0,
                 measured_points_x=[-0.8, -0.7, -0.6, -0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8],
                 measured_points_y=[-0.5, -0.4, -0.3, -0.2, -0.1, 0., 0.1, 0.2, 0.3, 0.4, 0.5]):
        self.env = env
        self.terrain_type_probs = terrain_type_probs
        self.max_difficulty_level = max_difficulty_level
        self.inner_platform_size = inner_platform_size
        self.max_slope = max_slope
        self.base_step_height = base_step_height
        self.additional_max_step_height = additional_max_step_height
        self.random_min_height = random_min_height
        self.random_max_height = random_max_height
        self.random_step = random_step
        self.random_downsampled_scale = random_downsampled_scale
        self.stairs_step_width = stairs_step_width
        self.obstacles_base_height = obstacles_base_height
        self.obstacles_additional_factor = obstacles_additional_factor
        self.obstacles_nr_rectangles = obstacles_nr_rectangles
        self.obstacles_rectangle_min_size = obstacles_rectangle_min_size
        self.obstacles_rectangle_max_size = obstacles_rectangle_max_size
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
        self.current_difficulty_levels = np.zeros_like(self.terrain_type_probs)
        self.current_terrain_type = 0
        self.farthest_distance_to_center = 0.0
        self.command_magnitude = 0.0
        self.nr_commands = 0

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
        self.farthest_distance_to_center = max(self.farthest_distance_to_center, np.linalg.norm(self.env.data.qpos[:2]))
        self.command_magnitude += np.linalg.norm(np.array([self.env.goal_x_velocity, self.env.goal_y_velocity]))
        self.nr_commands += 1

        if 3.5 < np.abs(self.env.data.qpos[0]) < 4.0 or 3.5 < np.abs(self.env.data.qpos[1]) < 4.0:
            self.got_to_the_edge = True
            self.env.data.qpos, self.env.data.qvel = self.env.initial_state_function.setup()
    

    def info(self, info):
        for i in range(len(self.current_difficulty_levels)):
            info[f"env_info/terrain/terrain_{i}_difficulty"] = self.current_difficulty_levels[i]
        info["env_info/terrain/average_difficulty"] = np.mean(self.current_difficulty_levels)
        return info
    

    def update_curriculum(self):
        if self.env.total_timesteps != 0:
            if self.got_to_the_edge:
                self.current_difficulty_levels[self.current_terrain_type] += 1
                if self.current_difficulty_levels[self.current_terrain_type] > self.max_difficulty_level:
                    self.current_difficulty_levels[self.current_terrain_type] = np.random.randint(0, self.max_difficulty_level)
            else:
                if self.nr_commands > 0 and self.farthest_distance_to_center < (self.command_magnitude / self.nr_commands) * 0.5 * self.env.episode_length_in_seconds:
                    self.current_difficulty_levels[self.current_terrain_type] = max(0, self.current_difficulty_levels[self.current_terrain_type] - 1)
        
        self.got_to_the_edge = False
        self.farthest_distance_to_center = 0.0
        self.command_magnitude = 0.0
        self.nr_commands = 0

    def sample(self):
        using_terrains_txt = False
        if self.env.mode == "test" and Path("terrain.txt").is_file():
            with open("terrain.txt", "r") as f:
                terrains = f.readlines()
            if len(terrains) == 2:
                self.current_terrain_type = int(terrains[0])
                difficulty_factor = float(terrains[1])
                using_terrains_txt = True

        if not using_terrains_txt:
            self.update_curriculum()
            self.current_terrain_type = np.random.choice(len(self.terrain_type_probs), p=self.terrain_type_probs)
            difficulty_factor = self.current_difficulty_levels[self.current_terrain_type] / self.max_difficulty_level

        if self.current_terrain_type == 0:
            slope = self.max_slope * difficulty_factor
            isaac_height_field = self.pyramid_sloped_terrain(slope)

        elif self.current_terrain_type == 1:
            slope = -self.max_slope * difficulty_factor
            isaac_height_field = self.pyramid_sloped_terrain(slope)

        elif self.current_terrain_type == 2:
            slope = self.max_slope * difficulty_factor
            isaac_height_field = self.pyramid_sloped_terrain(slope)
            isaac_height_field = self.random_uniform_terrain(isaac_height_field)

        elif self.current_terrain_type == 3:
            step_height = self.base_step_height + self.additional_max_step_height * difficulty_factor
            isaac_height_field = self.pyramid_stairs_terrain(step_height)

        elif self.current_terrain_type == 4:
            step_height = self.base_step_height + self.additional_max_step_height * difficulty_factor
            step_height *= -1
            isaac_height_field = self.pyramid_stairs_terrain(step_height)

        elif self.current_terrain_type == 5:
            discrete_obstacles_height = self.obstacles_base_height + self.obstacles_additional_factor * difficulty_factor
            isaac_height_field = self.discrete_obstacles_terrain(discrete_obstacles_height)

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


    def pyramid_sloped_terrain(self, slope):
        x = np.arange(0, self.hfield_length)
        y = np.arange(0, self.hfield_length)
        center_x = self.hfield_half_length
        center_y = self.hfield_half_length
        xx, yy = np.meshgrid(x, y, sparse=True)
        xx = (center_x - np.abs(center_x-xx)) / center_x
        yy = (center_y - np.abs(center_y-yy)) / center_y
        xx = xx.reshape(self.hfield_length, 1)
        yy = yy.reshape(1, self.hfield_length)
        max_height = slope * self.hfield_half_length_in_meters
        height_field_raw = self.base_height_field + (max_height * xx * yy)
        platform_size = int(self.inner_platform_size * self.one_meter_length)
        x1 = self.hfield_half_length - (platform_size // 4)
        y1 = self.hfield_half_length - (platform_size // 4)
        min_h = min(height_field_raw[x1, y1], 0)
        max_h = max(height_field_raw[x1, y1], 0)
        height_field_raw = np.clip(height_field_raw, min_h, max_h)
        return height_field_raw
    

    def random_uniform_terrain(self, base_height_field):
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
        height_field_raw = base_height_field + add_height_field
        return height_field_raw
    

    def pyramid_stairs_terrain(self, step_height):
        step_width = int(self.stairs_step_width * self.one_meter_length)
        platform_size = int(self.inner_platform_size * self.one_meter_length)
        height, start_x, start_y, stop_x, stop_y = 0, 0, 0, self.hfield_length, self.hfield_length
        height_field_raw = self.base_height_field.copy()
        while (stop_x - start_x) > platform_size and (stop_y - start_y) > platform_size:
            start_x += step_width
            stop_x -= step_width
            start_y += step_width
            stop_y -= step_width
            height += step_height 
            height_field_raw[start_x: stop_x, start_y: stop_y] = height
        return height_field_raw
    

    def discrete_obstacles_terrain(self, discrete_obstacles_height):
        max_height = discrete_obstacles_height
        min_size = int(self.obstacles_rectangle_min_size * self.one_meter_length)
        max_size = int(self.obstacles_rectangle_max_size * self.one_meter_length)
        platform_size = int(self.inner_platform_size * self.one_meter_length)
        (i, j) = self.base_height_field.shape
        height_range = [-max_height, -max_height / 2, max_height / 2, max_height]
        width_range = range(min_size, max_size, 4)
        length_range = range(min_size, max_size, 4)
        height_field_raw = self.base_height_field.copy()
        for _ in range(self.obstacles_nr_rectangles):
            width = np.random.choice(width_range)
            length = np.random.choice(length_range)
            start_i = np.random.choice(range(0, i-width, 4))
            start_j = np.random.choice(range(0, j-length, 4))
            height_field_raw[start_i:start_i+width, start_j:start_j+length] = np.random.choice(height_range)
        x1 = (self.hfield_length - platform_size) // 2
        x2 = (self.hfield_length + platform_size) // 2
        y1 = (self.hfield_length - platform_size) // 2
        y2 = (self.hfield_length + platform_size) // 2
        height_field_raw[x1:x2, y1:y2] = 0
        return height_field_raw
