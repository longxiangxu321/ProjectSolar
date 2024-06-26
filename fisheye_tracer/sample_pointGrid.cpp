#include "sample_pointGrid.h"

namespace osc {

void calculate_mass_center(const vec3f &A, const vec3f &B, const vec3f &C, const vec3f &direction, const int splits,
                           std::stringstream &output_stream, std::vector<GridPoint> &grid_n, const int &current_depth,
                           const int &max_depth, const uint32_t surf_gmlid_ptr)
{
//    vec3 sun_d(1,1,0.5);
    vec3f D,E,F;
    D = (A+B)/2;
    E = (B+C)/2;
    F = (C+A)/2;

    if (current_depth > max_depth) {
        return;  // prevent excessive recursion
    }

    if (splits == 0) {
        vec3f vo = (A + B + C) /3;
        if (!vo.x || !vo.y || !vo.z) return;
        
        vec3f position = vec3f(vo.x + direction.x * 0.01, vo.y + direction.y * 0.01, vo.z + direction.z * 0.01);
        // std::cout << vo.z << std::endl;
        output_stream << std::setprecision(10) << position.x << " " 
        << position.y << " " << position.z << " "
        << direction.x << " "<< direction.y << " "<< direction.z <<"\n";
        
        // vec3f normal = vec3(direction.x(), direction.y(), direction.z());
        GridPoint gp(position, direction);

        // std::vector<vec3> hemisphere_samples = hemisphere_sampling(gp.normal, 5);
        // gp.hemisphere_samples = hemisphere_samples;
        gp.surf_gmlid = surf_gmlid_ptr;
        grid_n.emplace_back(gp);

    }

    if (splits == -1) {
        
        if ( (length(A-B)/ length(B-C))>10 || (length(A-C)/ length(B-C))>10) {
            calculate_mass_center(A, D, F, direction, 0, output_stream, grid_n, current_depth + 1, max_depth,
                                  surf_gmlid_ptr);
        }
        if ( (length(A-B)/ length(A-C)) >10 || (length(B-C)/ length(A-C))>10) {
            calculate_mass_center(D, B, E, direction, 0, output_stream, grid_n, current_depth + 1, max_depth,
                                  surf_gmlid_ptr);
        }
        if ( (length(A-C)/ length(A-B)) >10 || (length(B-C)/ length(A-B))>10) {
            calculate_mass_center(F, E, C, direction, 0, output_stream, grid_n, current_depth + 1, max_depth,
                                  surf_gmlid_ptr);
        }
        return;
    }

    calculate_mass_center(A, D, F, direction, splits - 1, output_stream, grid_n, current_depth + 1, max_depth,
                          surf_gmlid_ptr);
    calculate_mass_center(D, B, E, direction, splits - 1, output_stream, grid_n, current_depth + 1, max_depth,
                          surf_gmlid_ptr);
    calculate_mass_center(F, E, C, direction, splits - 1, output_stream, grid_n, current_depth + 1, max_depth,
                          surf_gmlid_ptr);
    calculate_mass_center(D, E, F, direction, splits - 1, output_stream, grid_n, current_depth + 1, max_depth,
                          surf_gmlid_ptr);
}


std::vector<GridPoint> create_point_grid(const Model& citymodel) {
   std::string output_file = "../grid.xyz";
    // std::string output_file = CFG["shadow_calc"]["pointgrid_path"];
    std::ofstream out(output_file);



    std::vector<GridPoint> grid_current;

    if (!out.is_open()) {
        std::cout << "Error opening file " << "'" << output_file <<"'.";
    }

    else
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


                

                double triangle_area = length(cross(side0,side1)) / 2;
                // std::cout<<triangle_area<<std::endl;
                if (triangle_area <=2) num_s = 0;
                else if (triangle_area <4 && triangle_area >2) num_s = 1;
                else
                {
                    // double num_half_square = jt->area();
                    num_s = std::floor(std::log(triangle_area/16+1) / std::log(3));
                }
                size_t triangle_index = std::distance((*it)->index.begin(), jt);

                

                calculate_mass_center(v0, v1, v2, (*it)->normal[triangle_index], num_s, ss, grid_current, 0, num_s+2,
                                          (*it)->materialID[triangle_index]);


        }


        
        }
    out << ss.str();
    return grid_current;
    }

    }
}