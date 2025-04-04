import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
from line_profiler import LineProfiler
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
import pvlib
import time

from joblib import Parallel, delayed

def pd_integrate_voxel_info(point_grid, irradiance_vals, voxel_dim, voxel_size, bbox_min):
    """
    Integrate the point cloud data into a voxel grid and calculate the irradiance values for each voxel.
    point_grid: np.array, shape=(N, 6), dtype=float32, the point cloud data with normal vectors.
    irradiance_vals: np.array, shape=(N, M), dtype=float32, the irradiance values for each point in M timesteps.
    """
    voxel_grid = {}
    # bbox_min = np.min(point_grid[:, :3], axis=0)
    
    def compute_intensity_for_face(normals, face_normal, albedos, areas, irradiance_values):
        """
        irradiance_values: shape (N, M), representing the irradiance values for each point in M timesteps.
        normals: shape (N, 3), representing the normal vectors of each point.
        face_normal: shape (3,), representing the normal vector of the face.
        albedos: shape (N,), representing the albedo values of each point.
        
        """

        valid_indices = normals @ face_normal > 0
        valid_normals = normals[valid_indices]
        valid_areas = areas[valid_indices]
        
        normalized_normals = valid_normals / np.linalg.norm(valid_normals, axis=1, keepdims=True)
        normalized_face_normal = face_normal / np.linalg.norm(face_normal)
        valid_irradiance_values = irradiance_values[valid_indices]

        intensity_contributions = valid_irradiance_values * ((albedos[valid_indices] * (normalized_normals @ normalized_face_normal))[:, np.newaxis])
        
        normalized_ratio = valid_areas / np.sum(valid_areas)
        if len(intensity_contributions) > 0:
            intensity = np.sum(intensity_contributions * normalized_ratio[:, np.newaxis], axis=0)
        else:
            intensity = np.zeros(irradiance_values.shape[1])
        
        return intensity


    for i in range(point_grid.shape[0]):
        voxel_ids = ((point_grid[i, :3] - bbox_min) / voxel_size).astype(int)
        voxel_idx = voxel_ids[0] + voxel_ids[1] * voxel_dim[0] + voxel_ids[2] * voxel_dim[0] * voxel_dim[1]
        if voxel_idx not in voxel_grid:
            voxel_grid[voxel_idx] = {'normals': [], 'irradiance': [], 'albedos':[], 'areas': []}
        voxel_grid[voxel_idx]['normals'].append(point_grid[i, 3:6])
        voxel_grid[voxel_idx]['irradiance'].append(np.array(irradiance_vals[i]))
        voxel_grid[voxel_idx]['albedos'].append(point_grid[i, 6])
        voxel_grid[voxel_idx]['areas'].append(point_grid[i, 7])

    data = []
    data.append({
        'voxel_idx': np.iinfo(np.uint32).max,
        'intensities': np.array([np.zeros(irradiance_vals.shape[1]) for _ in range(6)])
    })
    
    for voxel_idx, voxel_data in voxel_grid.items():
        up_intensity= compute_intensity_for_face(np.array(voxel_data['normals']), np.array([0, 0, 1]), 
                                                 np.array(voxel_data['albedos']),np.array(voxel_data['areas']),  np.array(voxel_data['irradiance']))
        down_intensity = compute_intensity_for_face(np.array(voxel_data['normals']), np.array([0, 0, -1]), 
                                                    np.array(voxel_data['albedos']), np.array(voxel_data['areas']),  np.array(voxel_data['irradiance']))
        left_intensity= compute_intensity_for_face(np.array(voxel_data['normals']), np.array([-1, 0, 0]), 
                                                   np.array(voxel_data['albedos']), np.array(voxel_data['areas']),  np.array(voxel_data['irradiance']))
        right_intensity = compute_intensity_for_face(np.array(voxel_data['normals']), np.array([1, 0, 0]), 
                                                     np.array(voxel_data['albedos']), np.array(voxel_data['areas']),  np.array(voxel_data['irradiance']))
        front_intensity = compute_intensity_for_face(np.array(voxel_data['normals']), np.array([0, -1, 0]), 
                                                     np.array(voxel_data['albedos']), np.array(voxel_data['areas']),  np.array(voxel_data['irradiance']))
        back_intensity = compute_intensity_for_face(np.array(voxel_data['normals']), np.array([0, 1, 0]), 
                                                    np.array(voxel_data['albedos']), np.array(voxel_data['areas']),  np.array(voxel_data['irradiance']))
        data.append({
            'voxel_idx': voxel_idx,
            'intensities': np.array([up_intensity, down_intensity, left_intensity, right_intensity, front_intensity, back_intensity], dtype=np.float32)
        })

    df = pd.DataFrame(data)
    df.set_index('voxel_idx', inplace=True)
    return df


def process_batch(batch_start, batch_end, hemisphere_resolution, voxel_grid, num_timestep, shared_index_map_name, shared_cos_factor_map_name):
    # 通过共享内存加载数据
    n_azimuth = 360//hemisphere_resolution[0]
    n_elevation = 90//hemisphere_resolution[1]
    num_samples = n_azimuth * n_elevation

    existing_index_map = np.memmap(shared_index_map_name, dtype=np.uint32, mode='r')
    existing_cos_factor_map = np.memmap(shared_cos_factor_map_name, dtype=np.float16, mode='r')


    cos_factor_valid = existing_cos_factor_map[batch_start*num_samples*6:batch_end*num_samples*6]
    voxel_indexes = existing_index_map[batch_start*num_samples:batch_end*num_samples]

    # 重建 voxel_grid，获取相关体素数据
    relevant_voxels = voxel_grid.reindex(voxel_indexes)
    not_exist_ids = relevant_voxels.isna().any(axis=1)
    fill_values = {
        'intensities': np.array([np.zeros(num_timestep) for _ in range(6)], dtype=np.float32)
    }
    fill_df = pd.DataFrame([fill_values], index=relevant_voxels[not_exist_ids].index)
    relevant_voxels.loc[not_exist_ids] = fill_df

    results = np.zeros((batch_end - batch_start, num_timestep))

    # all_intensities = np.concatenate(relevant_voxels['intensities'].to_numpy()) * (2 * np.pi / num_samples)

    # # 计算逐元素相乘的结果，使用预先计算的 cos_factor_valid
    # intensity_contributions = cos_factor_valid[:, np.newaxis] * all_intensities  # 形状为 ((batch_end - batch_start) * num_samples * 6, num_timestep)

    # # 将所有强度贡献重新聚合到 (batch_end - batch_start, num_timestep) 形状
    # intensity_contributions_reshaped = intensity_contributions.reshape(batch_end - batch_start, num_samples, 6, num_timestep)


    intensity_contributions_reshaped = (cos_factor_valid[:, np.newaxis] * (np.concatenate(relevant_voxels['intensities'].to_numpy()) * (2 * np.pi / num_samples))).reshape(batch_end - batch_start, num_samples, 6, num_timestep)

    # breakpoint()
    # 沿着样本维度和面维度累加强度贡献，得到每个目标点的最终结果
    results = np.sum(intensity_contributions_reshaped, axis=(1, 2))

    # breakpoint()
    return results, batch_start, batch_end

# def process_single_batch(batch_start, batch_end, hemisphere_resolution, voxel_grid, num_time_steps, index_map_filename, cos_factor_map_filename):
#     result, batch_start, batch_end = process_batch(batch_start, batch_end, hemisphere_resolution, voxel_grid, num_time_steps, index_map_filename, cos_factor_map_filename)
#     return result, batch_start, batch_end

def batch_update_grid_point_irradiance(point_grid, voxel_grid, irradiance, index_map, cos_factor_map, hemisphere_resolution, batch_size):
    print("start bouncing")
    print("batch size", batch_size)
    num_points = point_grid.shape[0]
    updated_irradiance = irradiance.copy()
    num_time_steps = irradiance.shape[1]

    for i in tqdm(range(0, num_points, batch_size)):
        batch_start = i
        batch_end = min(i + batch_size, num_points)
        result, batch_start, batch_end = process_batch(batch_start, batch_end, hemisphere_resolution, voxel_grid, num_time_steps, index_map.filename, cos_factor_map.filename)
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

    nz = point_normals[:, 2].reshape(-1, 1)
    simplified_svf = (1 + nz) / 2
    simplified_gvf = 1 - simplified_svf

    # breakpoint()
    simplified_diffuse = simplified_svf * weather_data[:, 1].reshape(1, -1)
    simplified_reflective = 0.2 * simplified_gvf * weather_data[:,0].reshape(1, -1)

    return direct_component, diffuse_component, simplified_diffuse, simplified_reflective


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

    # breakpoint()
    epw_data = np.column_stack((all_ghi, all_dhi, all_dni))    
    return epw_data


def obtain_tud(ground_recording_filename, sunpos_filename):
    sunpos = pd.read_csv(sunpos_filename)
    sunpos.columns.values[0]='timestamp'
    ground_recording = pd.read_csv(ground_recording_filename)
    sunpos['timestamp'] = pd.to_datetime(sunpos['timestamp']).dt.tz_localize(None)
    ground_recording['local_time'] = pd.to_datetime(ground_recording['local_time'])
    filtered_ground_recording = ground_recording[ground_recording['local_time'].isin(sunpos['timestamp'])]
    filtered_ground_recording = filtered_ground_recording.drop_duplicates(subset='local_time', keep='first').reset_index(drop=True)
    
    all_dni = filtered_ground_recording['DNI'].values
    all_dhi = filtered_ground_recording['DHI'].values

    cos_theta = np.cos(np.radians(90 - filtered_ground_recording['sun_altitude'].values))
    all_ghi = all_dhi + all_dni * cos_theta

    tud_data = np.column_stack((all_ghi, all_dhi, all_dni))
    
    return tud_data

if __name__=="__main__":
    current_file_path = os.path.abspath(__file__)
    current_dir = os.path.dirname(current_file_path)
    parent_dir = os.path.dirname(current_dir)
    os.chdir(parent_dir)


    with open('config.json', 'r') as file:
        CONFIG = json.load(file)

    folder_path = CONFIG['study_area']['data_root']
    data_root = os.path.join(folder_path, CONFIG['output_folder_name'])
    point_grid_path = os.path.join(data_root, 'intermediate', 'point_grid.dat')
    solar_position_path = os.path.join(data_root, 'intermediate', 'sun_pos.csv')

    index_map_path = os.path.join(data_root,  'index_map.dat')
    shadow_map_path = os.path.join(data_root, 'shadow_map.dat')

    # azimuth_map_path = os.path.join(data_root,  'azimuth_map.dat')
    # elevation_map_path = os.path.join(data_root, 'elevation_map.dat')
    cos_factor_map_path = os.path.join(data_root, 'cosine_map.dat')

    horizon_factor_path = os.path.join(data_root, 'horizon_factor_map.dat')
    sky_view_factor_path = os.path.join(data_root,  'sky_view_factor_map.dat')

    voxel_dimension = (CONFIG['result']['voxel_dim_x'], CONFIG['result']['voxel_dim_y'], CONFIG['result']['voxel_dim_z'])

    voxel_size = CONFIG['voxel_resolution']
    batch_size = CONFIG['irradiance_batch_size']

    num_bounces = CONFIG['num_bounces']
    bbox_min = np.array(CONFIG['result']['bbox_min'])
    hemisphere_resolution = (CONFIG['azimuth_resolution'], CONFIG['elevation_resolution'])

    num_azimuth = 360//CONFIG['azimuth_resolution']
    num_elevation = 90//CONFIG['elevation_resolution']
    num_samples = num_azimuth*num_elevation
    # index_dim = (num_elevation,num_azimuth)

    point_grid = np.loadtxt(point_grid_path)
    svf_data = np.memmap(sky_view_factor_path, dtype=np.int32, mode='r')

    index_map = np.memmap(index_map_path, dtype=np.uint32, mode='r')
    # azimuth_map = np.memmap(azimuth_map_path, dtype=np.float16, mode='r')
    # elevation_map = np.memmap(elevation_map_path, dtype=np.float16, mode='r')
    cos_factor_map = np.memmap(cos_factor_map_path, dtype=np.float16, mode='r')

    solar_position = read_sunpos(solar_position_path)
    shadow_result = np.memmap(shadow_map_path, dtype=bool, mode='r')
    # weather_data = np.random.randint(0, 1000, size=(solar_position.shape[0], 3))
    scenario = CONFIG["scenario"]
    if scenario == "tud":
        print("Reading TUD scenario weather data")
        ground_records_path = os.path.join(folder_path, "measured_data.csv")
        weather_data = obtain_tud(ground_records_path, solar_position_path)
        # np.save(os.path.join(data_root, 'weather_data.npy'), weather_data)
    else:
        print("Reading regular scenario weather data")
        epw_filename = os.path.join(folder_path, CONFIG['epw_file'])
        weather_data = obtain_epw(epw_filename, solar_position_path)
        # breakpoint()
        np.save(os.path.join(data_root, 'weather_data.npy'), weather_data)

    svf_new_shape = svf_data[:,np.newaxis]
    svf_map = svf_new_shape.astype(np.float32)/num_samples
    shadow_map = np.reshape(shadow_result, (point_grid.shape[0], solar_position.shape[0]))

    start_time = time.time()
    print("Calculating direct beam and sky diffuse irradiance")
    direct_irradiance, diffuse_irradiance, simplified_diffuse, simplified_reflective = calculate_isotropic(point_grid, shadow_map, svf_map, weather_data, solar_position)
    print("Direct beam and sky diffuse irradiance calculated")
    end_time_direct_diffuse = time.time()
    exec_time_direct_diffuse = end_time_direct_diffuse- start_time
    CONFIG["result"]["direct_diffuse_time"] = exec_time_direct_diffuse
    print("Time for direct and diffuse computation: ", exec_time_direct_diffuse)
    irradiance = direct_irradiance + diffuse_irradiance

    simplified_values = np.stack([direct_irradiance, simplified_diffuse, simplified_reflective])
    np.save(os.path.join(data_root, 'simplified_irradiance.npy'), simplified_values)


    irradiance_list = [irradiance]

    
    print("Calculating global irradiance")
    print("Number of bounces", num_bounces)
    for i in range(num_bounces):
        print(f"Calculating bounce {i+1}")
        voxel_grid = pd_integrate_voxel_info(point_grid, irradiance_list[i], voxel_dimension, voxel_size, bbox_min)
        updated_irradiance = batch_update_grid_point_irradiance(point_grid, voxel_grid, irradiance, index_map, cos_factor_map, hemisphere_resolution, batch_size)
        irradiance_list.append(updated_irradiance)

    irradiance_arr = np.stack(irradiance_list)

    end_time = time.time()

    execution_time = end_time - start_time

    CONFIG["result"]["global_irradiance_time"] = execution_time

    with open('config.json', 'w') as file:
        json.dump(CONFIG, file, indent=4)
    
    backup_config_path = os.path.join(data_root, 'config.json')
    with open(backup_config_path, 'w') as file:
        json.dump(CONFIG, file, indent=4)

    np.save(os.path.join(data_root, 'irradiance.npy'), irradiance_arr)



