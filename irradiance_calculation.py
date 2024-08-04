import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from line_profiler import LineProfiler




def pd_integrate_voxel_info(point_grid, irradiance_vals, voxel_dim, voxel_size=2.0):
    """
    Integrate the point cloud data into a voxel grid and calculate the irradiance values for each voxel.
    point_grid: np.array, shape=(N, 6), dtype=float32, the point cloud data with normal vectors.
    irradiance_vals: np.array, shape=(N,), dtype=float32, the irradiance values for each point.
    """
    voxel_grid = {}
    bbox_min = np.min(point_grid[:, :3], axis=0)
    
    def compute_intensity_for_face(normals, face_normal):
        ratio = np.sum(normals @ face_normal > 0) / len(normals)
        dot_products = normals[normals @ face_normal > 0] @ face_normal
        if len(dot_products) > 0:
            intensity = np.mean(dot_products) * ratio
        else:
            intensity = 0
        return intensity


    for i in range(point_grid.shape[0]):
        voxel_ids = ((point_grid[i, :3] - bbox_min) / voxel_size).astype(int)
        voxel_idx = voxel_ids[0] + voxel_ids[1] * voxel_dim[0] + voxel_ids[2] * voxel_dim[0] * voxel_dim[1]
        if voxel_idx not in voxel_grid:
            voxel_grid[voxel_idx] = {'normals': [], 'irradiance': 0}
        voxel_grid[voxel_idx]['normals'].append(point_grid[i, 3:6])
        voxel_grid[voxel_idx]['irradiance'] += irradiance_vals[i]

    data = []
    data.append({
        'voxel_idx': np.iinfo(np.uint32).max,
        'irradiance': 0,
        'up_intensity': 0,
        'down_intensity': 0,
        'left_intensity': 0,
        'right_intensity': 0,
        'front_intensity': 0,
        'back_intensity': 0
    })
    
    for voxel_idx, voxel_data in voxel_grid.items():
        normals = np.array(voxel_data['normals'])
        up_intensity = compute_intensity_for_face(normals, np.array([0, 0, 1])) + 0.1
        down_intensity = compute_intensity_for_face(normals, np.array([0, 0, -1])) + 0.1
        left_intensity = compute_intensity_for_face(normals, np.array([-1, 0, 0])) + 0.1
        right_intensity = compute_intensity_for_face(normals, np.array([1, 0, 0])) + 0.1
        front_intensity = compute_intensity_for_face(normals, np.array([0, -1, 0])) + 0.1
        back_intensity = compute_intensity_for_face(normals, np.array([0, 1, 0])) + 0.1

        intensity_sum = (up_intensity + down_intensity + left_intensity + right_intensity +
                         front_intensity + back_intensity)

        if intensity_sum > 0:
            up_intensity /= intensity_sum
            down_intensity /= intensity_sum
            left_intensity /= intensity_sum
            right_intensity /= intensity_sum
            front_intensity /= intensity_sum
            back_intensity /= intensity_sum

        data.append({
            'voxel_idx': voxel_idx,
            'irradiance': voxel_data['irradiance'],
            'up_intensity': up_intensity,
            'down_intensity': down_intensity,
            'left_intensity': left_intensity,
            'right_intensity': right_intensity,
            'front_intensity': front_intensity,
            'back_intensity': back_intensity
        })

    df = pd.DataFrame(data)
    df.set_index('voxel_idx', inplace=True)
    return df


def batch_update_grid_point_irradiance(point_grid, voxel_grid, irradiance, index_map, azimuth_map, elevation_map, num_samples):
    """
    update the irradiance values for each point in the point grid.
    point_grid: np.array, shape=(N, 6), dtype=float32, the point cloud data with normal vectors.
    voxel_grid: pd.DataFrame, the voxel grid data.
    irradiance: np.array, shape=(N,), dtype=float32, the irradiance values for each point.
    index_map: np.array, shape=(N*num_samples), dtype=int32, the voxel index for each point.
    azimuth_map: np.array, shape=(N*num_samples), dtype=float32, the azimuth angle for each point.
    elevation_map: np.array, shape=(N*num_samples), dtype=float32, the elevation angle for each point.
    num_samples: int, the number of samples for each point.
    """
    num_points = point_grid.shape[0]
    updated_irradiance = irradiance.copy()

    batch_size = 1000
    for i in tqdm(range(0, num_points, batch_size), desc="Processing points"):
        batch_start = i
        batch_end = min(i + batch_size, num_points)

        # azimuth_point = azimuth_map[batch_start*num_samples:batch_end*num_samples]
        # elevation_point = elevation_map[batch_start*num_samples:batch_end*num_samples]
        # azimuth_valid = np.radians(azimuth_point)
        # elevation_valid = np.radians(elevation_point)
        azimuth_valid = azimuth_map[batch_start*num_samples:batch_end*num_samples]
        elevation_valid = elevation_map[batch_start*num_samples:batch_end*num_samples]
        directions_x = np.cos(elevation_valid) * np.cos(azimuth_valid)
        directions_y = np.cos(elevation_valid) * np.sin(azimuth_valid)
        directions_z = np.sin(elevation_valid)
        

        voxel_indexes = index_map[batch_start*num_samples:batch_end*num_samples]
        relevant_voxels = voxel_grid.reindex(voxel_indexes, fill_value=0)

        pixels_irradiance = relevant_voxels['irradiance'].values

        up_intensities = relevant_voxels['up_intensity'].values
        down_intensities = relevant_voxels['down_intensity'].values
        left_intensities = relevant_voxels['left_intensity'].values
        right_intensities = relevant_voxels['right_intensity'].values
        front_intensities = relevant_voxels['front_intensity'].values
        back_intensities = relevant_voxels['back_intensity'].values
        
        x_mask = np.where(directions_x > 0, right_intensities, left_intensities)
        y_mask = np.where(directions_y > 0, back_intensities, front_intensities)
        z_mask = np.where(directions_z > 0, up_intensities, down_intensities)
        
        contributions = x_mask + y_mask + z_mask
        updated_irradiance[batch_start:batch_end] += np.sum(contributions * pixels_irradiance/num_samples * 0.1)
        

    return updated_irradiance


if __name__=="__main__":
    # Load the point cloud data
    point_grid_path = "./out/build/grid_points.dat"
    index_map_path = "./out/build/index_map.dat"
    shadow_map_path = "./out/build/shadow_map.dat"

    azimuth_map_path = "./out/build/azimuth_map.dat"
    elevation_map_path = "./out/build/elevation_map.dat"
    voxel_dim = (587,590,50)
    voxel_size = 2.0
    num_samples = 360*90

    index_map = np.memmap(index_map_path, dtype=np.uint32, mode='r')
    azimuth_map = np.memmap(azimuth_map_path, dtype=np.float16, mode='r')
    elevation_map = np.memmap(elevation_map_path, dtype=np.float16, mode='r')
    point_grid = np.loadtxt(point_grid_path)

    irradiance = np.random.uniform(0, 1000, point_grid.shape[0])

    first_voxel_grid = pd_integrate_voxel_info(point_grid, irradiance, voxel_dim, voxel_size)

    lp = LineProfiler()
    lp_wrapper = lp(batch_update_grid_point_irradiance)
    lp_wrapper(point_grid, first_voxel_grid, irradiance, index_map, azimuth_map, elevation_map, num_samples)
    lp.print_stats()

    # updated_irradiance_1 = batch_update_grid_point_irradiance(point_grid, first_voxel_grid, irradiance, index_map, azimuth_map, elevation_map)
    # updated_voxel_grid = pd_integrate_voxel_info(point_grid, updated_irradiance_1, voxel_dim, voxel_size)
    # updated_irradiance_2 = batch_update_grid_point_irradiance(point_grid, updated_voxel_grid, updated_irradiance_1, index_map, azimuth_map, elevation_map)