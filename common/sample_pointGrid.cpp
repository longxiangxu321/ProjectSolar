#include "sample_pointGrid.h"

namespace solarcity {

void calculate_mass_center(const vec3f &A, const vec3f &B, const vec3f &C, const int splits,std::vector<GridPoint> &grid_n, const int &current_depth,
                            const int &max_depth, const Triangle_info triangle_info)
{


    vec3f D,E,F;
    // vec3f direction = triangle_info.direction;
    // uint32_t surf_gmlid_ptr = triangle_info.surf_gmlid;
    // int surface_type = triangle_info.surface_type;

    D = (A+B)/2;
    E = (B+C)/2;
    F = (C+A)/2;

    if (current_depth > max_depth) {
        return;  // prevent excessive recursion
    }

    if (splits == 0) {
        vec3f vo = (A + B + C) /3;
        if (!vo.x || !vo.y || !vo.z) return;
        
        vec3f position = vec3f(vo.x + triangle_info.direction.x * 0.01, 
                                vo.y + triangle_info.direction.y * 0.01, 
                                vo.z + triangle_info.direction.z * 0.01);
        
        float gp_area = triangle_info.triangle_area/pow(4,splits);
        if (current_depth > splits) gp_area = triangle_info.triangle_area/pow(4, current_depth);
        
        // vec3f normal = vec3(direction.x(), direction.y(), direction.z());
        GridPoint gp(position, triangle_info, gp_area);

        // std::vector<vec3> hemisphere_samples = hemisphere_sampling(gp.normal, 5);
        // gp.hemisphere_samples = hemisphere_samples;
        grid_n.emplace_back(gp);

    }

    if (splits == -1) {
        
        if ( (length(A-B)/ length(B-C))>10 || (length(A-C)/ length(B-C))>10) {
            calculate_mass_center(A, D, F, 0, grid_n, current_depth + 1, max_depth, triangle_info);
        }
        if ( (length(A-B)/ length(A-C)) >10 || (length(B-C)/ length(A-C))>10) {
            calculate_mass_center(D, B, E, 0, grid_n, current_depth + 1, max_depth, triangle_info);
        }
        if ( (length(A-C)/ length(A-B)) >10 || (length(B-C)/ length(A-B))>10) {
            calculate_mass_center(F, E, C, 0, grid_n, current_depth + 1, max_depth, triangle_info);
        }
        return;
    }



    calculate_mass_center(A, D, F, splits - 1, grid_n, current_depth + 1, max_depth,
                          triangle_info);
    calculate_mass_center(D, B, E, splits - 1, grid_n, current_depth + 1, max_depth,
                          triangle_info);
    calculate_mass_center(F, E, C, splits - 1, grid_n, current_depth + 1, max_depth,
                          triangle_info);
    calculate_mass_center(D, E, F, splits - 1, grid_n, current_depth + 1, max_depth,
                          triangle_info);
}




std::vector<GridPoint> create_point_grid(const Model& citymodel, float sampling_density = 16) {
    // std::string output_file = "../grid.xyz";
    // std::string output_file = CFG["shadow_calc"]["pointgrid_path"];
    // std::ofstream out(output_file);

    // float sampling_density = 16;
//    std::string output_file = "../grid.xyz";
    // std::string output_file = CFG["shadow_calc"]["pointgrid_path"];
    // std::ofstream out(output_file);

    // float sampling_density = 16;



    std::vector<GridPoint> grid_current;

    // if (!out.is_open()) {
    //     std::cout << "Error opening file " << "'" << output_file <<"'.";
    // }

    // else
    {
        std::stringstream ss;


        for (auto it = citymodel.meshes.begin(); it!=citymodel.meshes.end(); ++it) {
            for (auto jt = (*it)->index.begin(); jt!=(*it)->index.end(); ++jt) {
                // specification_stream<<grid_current.size()<<" ";
                int num_s;
                vec3f v0 = (*it)->vertex[jt->x];
                vec3f v1 = (*it)->vertex[jt->y];
                vec3f v2 = (*it)->vertex[jt->z];
                vec3f side0 = v1 - v0;
                vec3f side1 = v2 - v0;

                if (v0==v1 || v1==v2 || v2==v0) continue;

                double triangle_area = length(cross(side0,side1)) / 2;
                // std::cout<<triangle_area<<std::endl;
                if (triangle_area <=2) num_s = 0;
                else if (triangle_area <4 && triangle_area >2) num_s = 1;
                else
                {
                    // double num_half_square = jt->area();
                    num_s = std::floor(std::log(triangle_area/sampling_density+1) / std::log(3));
                }
                size_t triangle_index = std::distance((*it)->index.begin(), jt);

                int surf_gmlid = (*it)->globalID[triangle_index];
                int surface_type = (*it)->surfaceType[triangle_index];
                float surface_albedo = (*it)->albedo[triangle_index];
                vec3f normal = (*it)->normal[triangle_index];

                Triangle_info triangle_info = {normal, surf_gmlid, surface_type, surface_albedo, triangle_area};


                calculate_mass_center(v0, v1, v2, num_s, grid_current, 0, num_s+2, triangle_info);


        }


        
        }
    // out << ss.str();
    return grid_current;
    }

    }


void save_point_grid(const std::vector<GridPoint> &grid_n, const vec3f &translation, const std::string &filename) {
    std::ofstream output_stream(filename);
    int index = 0;
    for (const auto &gp : grid_n) {
        if (gp.triangle_info.surf_gmlid == 55241912 || gp.triangle_info.surf_gmlid == 19125524) {
            std::cout<< "gmlid " << gp.triangle_info.surf_gmlid << " index "<< index<<std::endl;
        }
        output_stream << std::fixed<<std::setprecision(6)
        << gp.position.x - static_cast<float>(translation.x) << " " << gp.position.y -static_cast<float>(translation.y) << " " << gp.position.z - static_cast<float>(translation.z)<< " "
        << gp.triangle_info.direction.x << " "<< gp.triangle_info.direction.y << " "<< gp.triangle_info.direction.z << " "
        << gp.triangle_info.surface_albedo << " "<< gp.area<< " " << gp.triangle_info.surface_type<< " "<< gp.triangle_info.surf_gmlid<<"\n";
        index++;
    }
    output_stream.close();
}

std::vector<GridPoint> clean_point_grid(const std::vector<GridPoint> &grid_n) {
    int false_points = 0;
    std::vector<GridPoint> filtered_grid_point;
    for (int i = 0; i < grid_n.size(); i++) {
        if (std::isnan(grid_n[i].position.x) || std::isnan(grid_n[i].position.y) || std::isnan(grid_n[i].position.z)) {
            false_points++;
        }
        else {
            filtered_grid_point.push_back(grid_n[i]);
        }
    }
    std::cout<<"Number of false points deleted: "<<false_points<<std::endl;
    return filtered_grid_point;
}


}