import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from line_profiler import LineProfiler
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pvlib




def pd_integrate_voxel_info(point_grid, irradiance_vals, voxel_dim, voxel_size=2.0):
    """
    Integrate the point cloud data into a voxel grid and calculate the irradiance values for each voxel.
    point_grid: np.array, shape=(N, 6), dtype=float32, the point cloud data with normal vectors.
    irradiance_vals: np.array, shape=(N, M), dtype=float32, the irradiance values for each point in M timesteps.
    """
    voxel_grid = {}
    bbox_min = np.min(point_grid[:, :3], axis=0)
    
    def compute_intensity_for_face(normals, face_normal, albedos):
        ratio = np.sum(normals @ face_normal > 0) / len(normals)
        dot_products = normals[normals @ face_normal > 0] @ face_normal
        
        if len(dot_products) > 0:
            intensity = np.mean(dot_products) * ratio
            mean_albedo = np.mean(albedos[normals @ face_normal > 0])
        else:
            intensity = 0
            mean_albedo = 0
        
        
        return intensity, mean_albedo


    for i in range(point_grid.shape[0]):
        voxel_ids = ((point_grid[i, :3] - bbox_min) / voxel_size).astype(int)
        voxel_idx = voxel_ids[0] + voxel_ids[1] * voxel_dim[0] + voxel_ids[2] * voxel_dim[0] * voxel_dim[1]
        if voxel_idx not in voxel_grid:
            voxel_grid[voxel_idx] = {'normals': [], 'irradiance': 0, 'albedos':[]}
        voxel_grid[voxel_idx]['normals'].append(point_grid[i, 3:6])
        voxel_grid[voxel_idx]['irradiance'] += irradiance_vals[i]
        voxel_grid[voxel_idx]['albedos'].append(point_grid[i, 6])

    data = []
    data.append({
        'voxel_idx': np.iinfo(np.uint32).max,
        'irradiance': np.zeros_like(irradiance_vals[0]),
        'up_intensity': 0,
        'down_intensity': 0,
        'left_intensity': 0,
        'right_intensity': 0,
        'front_intensity': 0,
        'back_intensity': 0
    })
    
    for voxel_idx, voxel_data in voxel_grid.items():
        normals = np.array(voxel_data['normals'])
        albedos = np.array(voxel_data['albedos'])
        up_intensity, up_mean_albedo = compute_intensity_for_face(normals, np.array([0, 0, 1]), albedos)
        down_intensity, down_mean_albedo = compute_intensity_for_face(normals, np.array([0, 0, -1]), albedos)
        left_intensity, left_mean_albedo = compute_intensity_for_face(normals, np.array([-1, 0, 0]), albedos)
        right_intensity, right_mean_albedo = compute_intensity_for_face(normals, np.array([1, 0, 0]), albedos)
        front_intensity, front_mean_albedo = compute_intensity_for_face(normals, np.array([0, -1, 0]), albedos)
        back_intensity, back_mean_albedo = compute_intensity_for_face(normals, np.array([0, 1, 0]), albedos)

        intensity_sum = (up_intensity + down_intensity + left_intensity + right_intensity +
                         front_intensity + back_intensity)

        offset =  1e-6
        if intensity_sum > 0:
            up_intensity = up_mean_albedo * (up_intensity +  offset)/intensity_sum
            down_intensity = down_mean_albedo * (down_intensity + offset)/intensity_sum
            left_intensity = left_mean_albedo * (left_intensity + offset)/intensity_sum
            right_intensity = right_mean_albedo * (right_intensity + offset)/intensity_sum
            front_intensity = front_mean_albedo * (front_intensity + offset)/intensity_sum
            back_intensity = back_mean_albedo * (back_intensity + offset)/intensity_sum

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




def process_batch(batch_start, batch_end, num_samples, voxel_grid, num_timestep, shared_index_map_name, shared_azimuth_map_name, shared_elevation_map_name):
    # 通过共享内存加载数据
    existing_index_map = np.memmap(shared_index_map_name, dtype=np.uint32, mode='r')
    existing_azimuth_map = np.memmap(shared_azimuth_map_name, dtype=np.float16, mode='r')
    existing_elevation_map = np.memmap(shared_elevation_map_name, dtype=np.float16, mode='r')

    azimuth_valid = existing_azimuth_map[batch_start*num_samples:batch_end*num_samples]
    elevation_valid = existing_elevation_map[batch_start*num_samples:batch_end*num_samples]
    voxel_indexes = existing_index_map[batch_start*num_samples:batch_end*num_samples]

    relevant_voxels = voxel_grid.reindex(voxel_indexes, fill_value=0.0)
    relevant_voxels['irradiance'] = relevant_voxels['irradiance'].apply(
        lambda x: np.zeros(num_timestep) if isinstance(x, float) else x
    )

    pixels_irradiance = np.vstack(relevant_voxels['irradiance'].values)

    x_mask = np.where((np.cos(elevation_valid) * np.cos(azimuth_valid)) > 0, relevant_voxels['right_intensity'].values, relevant_voxels['left_intensity'].values)
    y_mask = np.where((np.cos(elevation_valid) * np.sin(azimuth_valid)) > 0, relevant_voxels['back_intensity'].values, relevant_voxels['front_intensity'].values)
    z_mask = np.where((np.sin(elevation_valid)) > 0, relevant_voxels['up_intensity'].values, relevant_voxels['down_intensity'].values)

    contributions_raw = x_mask + y_mask + z_mask
    contributions = np.repeat(contributions_raw[:, np.newaxis], num_timestep, axis=1)
    
    return np.sum(contributions * pixels_irradiance / num_samples , axis=0), batch_start, batch_end

def batch_update_grid_point_irradiance(point_grid, voxel_grid, irradiance, index_map, azimuth_map, elevation_map, num_samples, batch_size):
    num_points = point_grid.shape[0]
    updated_irradiance = np.zeros_like(irradiance)
    num_time_steps = irradiance.shape[1]

    index_map_shape = index_map.shape

    with ProcessPoolExecutor() as executor:
        futures = []
        for i in range(0, num_points, batch_size):
            batch_start = i
            batch_end = min(i + batch_size, num_points)
            futures.append(
                executor.submit(process_batch, batch_start, batch_end, num_samples, voxel_grid, num_time_steps, index_map.filename, azimuth_map.filename, elevation_map.filename)
            )

        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing points"):
            result, batch_start, batch_end = future.result()
            updated_irradiance[batch_start:batch_end] += result

    return updated_irradiance




def calculate_isotropic(point_grid, shadow_map, svf_map, weather_data, solar_position):
    """
    point_grid, numpy array, shape (N, 6), N is the number of points, each row is (x, y, z, nx, ny, nz)
    shadow_map, numpy array, shape (N, M), shadow result for N point in M timesteps, value True represent in shadow, False represent not in shadow
    svf_map, numpy array, shape (N, 1), sky view factor for each point
    weather_data, numpy array, shape (M, 3), GHI, DHI, DNI data for each timestep
    solar position, numpy array, shape (M, 2), azimuth and elevation for each timestep

    calculate the radiation for each point in point_grid with isotropic model

    direct = DNI * cos(delta) * shadow_mask
    diffuse = DHI * SVF
    """
    point_normals = point_grid[:, 3:6]
    solar_position_radians = np.radians(solar_position)
    solar_position_radians[:, 0] = np.pi/2 - solar_position_radians[:, 0]

    solar_vectors = np.array([
    [
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation)
    ]
    for azimuth, elevation in zip(solar_position_radians[:,0], solar_position_radians[:,1])
    ])

    cos_delta_raw = (solar_vectors @ point_normals.T).T # shape (N, M)
    cos_delta= np.clip(cos_delta_raw, a_min=0, a_max=None)
    
    direct_component = cos_delta * shadow_map * weather_data[:, 2].reshape(1, -1) # shape (N, M)
    diffuse_component = svf_map * weather_data[:, 1].reshape(1, -1) # shape (N, M)

    return direct_component, diffuse_component


def read_sunpos(file_path):
    """
    file_path: solar position file in csv format
    read the solar position file and return a numpy array with shape (M, 2), 
    containing azimuth and elevation for each timestep
    """
    # Read the CSV file into a pandas DataFrame
    df = pd.read_csv(file_path)
    
    # Extract the azimuth and apparent elevation columns
    azimuth = df['azimuth'].values
    elevation = df['apparent_elevation'].values
    
    # Combine the azimuth and elevation into a numpy array with shape (M, 2)
    sun_positions = np.column_stack((azimuth, elevation))
    
    return sun_positions



def obtain_epw(epw_filename, sunpos_filename):
    epw = pvlib.iotools.read_epw(epw_filename)
    sunpos = pd.read_csv(sunpos_filename)
    sunpos.columns.values[0]='timestamp'

    irradiance_data = epw[0]
    irradiance_data.index = pd.to_datetime(irradiance_data.index)

    sunpos['timestamp'] = pd.to_datetime(sunpos['timestamp'], utc=True)
    all_ghi = []
    all_dni = []
    all_dhi = []
    # all_solar_zenith = []
    # all_solar_azimuth = []
    # all_dni_extra = []

    for index, row in sunpos.iterrows():
            row_time = row['timestamp'].tz_convert(irradiance_data.index.tz)
            month, day, hour = row_time.month, row_time.day, row_time.hour
            if month == 2 and day == 29:
                day = 28
            match = irradiance_data[(irradiance_data.index.month == month) & 
                        (irradiance_data.index.day == day) & 
                        (irradiance_data.index.hour == hour)]
            # solar_zenith = row['apparent_zenith']
            # solar_azimuth = row['azimuth']
            # dni_extra = pvlib.irradiance.get_extra_radiation(row['timestamp'])
            if (not match.empty):
                ghi, dni, dhi, dni_extra = match.iloc[0][['ghi', 'dni', 'dhi', 'etrn']]
                if dni_extra == 0:
                    dni_extra = pvlib.irradiance.get_extra_radiation(row['timestamp'])
                all_ghi.append(ghi)
                all_dni.append(dni)
                all_dhi.append(dhi)
                # all_solar_zenith.append(solar_zenith)
                # all_solar_azimuth.append(solar_azimuth)
                # all_dni_extra.append(dni_extra)
            else:
                print("error finding match")
    
    epw_data = np.column_stack((all_ghi, all_dhi, all_dni))
    return epw_data


if __name__=="__main__":

    with open('config.json', 'r') as file:
        CONFIG = json.load(file)

    folder_path = CONFIG['study_area']['data_root']
    data_root = os.path.join(folder_path, CONFIG['output_folder_name'])
    point_grid_path = os.path.join(data_root, 'intermediate', 'point_grid.dat')
    solar_position_path = os.path.join(data_root, 'intermediate', 'sun_pos.csv')

    index_map_path = os.path.join(data_root,  'index_map.dat')
    shadow_map_path = os.path.join(data_root, 'shadow_map.dat')

    azimuth_map_path = os.path.join(data_root,  'azimuth_map.dat')
    elevation_map_path = os.path.join(data_root, 'elevation_map.dat')

    horizon_factor_path = os.path.join(data_root, 'horizon_factor_map.dat')
    sky_view_factor_path = os.path.join(data_root,  'sky_view_factor_map.dat')

    voxel_dimension = (CONFIG['voxel_dim_x'], CONFIG['voxel_dim_y'], CONFIG['voxel_dim_z'])
    num_azimuth = 360//CONFIG['azimuth_resolution']
    num_elevation = 90//CONFIG['elevation_resolution']
    voxel_size = CONFIG['voxel_resolution']
    batch_size = CONFIG['irradiance_batch_size']

    num_bounces = CONFIG['num_bounces']


    num_samples = num_azimuth*num_elevation
    index_dim = (num_elevation,num_azimuth)

    point_grid = np.loadtxt(point_grid_path)
    svf_data = np.memmap(sky_view_factor_path, dtype=np.int32, mode='r')

    index_map = np.memmap(index_map_path, dtype=np.uint32, mode='r')
    azimuth_map = np.memmap(azimuth_map_path, dtype=np.float16, mode='r')
    elevation_map = np.memmap(elevation_map_path, dtype=np.float16, mode='r')

    solar_position = read_sunpos(solar_position_path)
    shadow_result = np.memmap(shadow_map_path, dtype=bool, mode='r')
    # weather_data = np.random.randint(0, 1000, size=(solar_position.shape[0], 3))
    epw_filename = os.path.join(folder_path, CONFIG['epw_file'])
    weather_data = obtain_epw(epw_filename, solar_position_path)

    svf_new_shape = svf_data[:,np.newaxis]
    svf_map = svf_new_shape.astype(np.float32)/num_samples
    shadow_map = np.reshape(shadow_result, (point_grid.shape[0], solar_position.shape[0]))

    direct_irradiance, diffuse_irradiance = calculate_isotropic(point_grid, shadow_map, svf_map, weather_data, solar_position)
    irradiance = direct_irradiance + diffuse_irradiance

    # first_voxel_grid = pd_integrate_voxel_info(point_grid, irradiance, voxel_dimension, voxel_size)
    # unique_shapes = first_voxel_grid['irradiance'].apply(lambda x: x.shape).unique()
    # updated_irradiance_1 = batch_update_grid_point_irradiance(point_grid, first_voxel_grid, irradiance, index_map, azimuth_map, elevation_map, num_samples)

    # print(np.unique(updated_irradiance_1))

    # lp = LineProfiler()
    # lp_wrapper = lp(batch_update_grid_point_irradiance)
    # lp_wrapper(point_grid, first_voxel_grid, irradiance, index_map, azimuth_map, elevation_map, num_samples, batch_size, albedo)
    # lp.print_stats()

    irradiance_list = [irradiance]

    for i in range(num_bounces):
        voxel_grid = pd_integrate_voxel_info(point_grid, irradiance_list[i], voxel_dimension, voxel_size)
        updated_irradiance = batch_update_grid_point_irradiance(point_grid, voxel_grid, irradiance, index_map, azimuth_map, elevation_map, num_samples, batch_size)
        irradiance_list.append(updated_irradiance)

    irradiance_arr = np.stack(irradiance_list)
    np.save(os.path.join(data_root, 'irradiance.npy'), irradiance_arr)


