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




inline bool xcompare(GridPoint a, GridPoint b) { return a.position.x < b.position.x; }
inline bool ycompare(GridPoint a, GridPoint b) { return a.position.y < b.position.y; }
inline bool zcompare(GridPoint a, GridPoint b) { return a.position.z < b.position.z; }

void buildKDTree(std::vector<GridPoint> &grid, GridPoint* kdtree, int* depth_list, 
    uint32_t start, uint32_t end, uint32_t current_node, int current_depth) {
    if (start >= end || current_node >= grid.size()) {
        return;
    }
    

    int axis = current_depth % 3;
    auto comparator = (axis == 0) ? xcompare : (axis == 1) ? ycompare : zcompare;
    int mid = (start + end) / 2;
    std::nth_element(grid.begin() + start, grid.begin() + mid, grid.begin() + end, comparator);

    kdtree[current_node] = grid[mid];
    depth_list[current_node] = current_depth;
    
    buildKDTree(grid, kdtree, depth_list, start, mid, 2*current_node+1, current_depth + 1);
    buildKDTree(grid, kdtree, depth_list, mid+1, end, 2*current_node+2, current_depth + 1);
}


KNNResult findKNN(const KDTree &kdTree, const int K, float radius, const GridPoint &target) {

    std::vector<uint32_t> stack;
    stack.push_back(0);

    uint32_t* indices = new uint32_t[K];
    std::fill(indices, indices + K, std::numeric_limits<uint32_t>::max());

    float* distances = new float[K];
    std::fill(distances, distances + K, std::numeric_limits<float>::max());

	

	while(!stack.empty()) {
		int node_idx = stack.back();
        stack.pop_back();

        if (node_idx >= kdTree.grid_size) continue;

		const GridPoint& node = kdTree.kdtree[node_idx];
        float dist = calculate_dist(node.position, target.position);
		
		if (dist < radius){
			for (int i = 0; i<K; ++i) {
				if(dist < distances[i]) {
					for (int j=K-1; j>i;--j) { //所有后续元素后移
						distances[j] = distances[j-1];
						indices[j] = indices[j-1];
					}
					distances[i]=dist;
					indices[i]=node_idx;
					break;
				}
			}
		}
		
		int axis = kdTree.node_depth[node_idx];
		float diff;
		
        if (axis == 0) diff = target.position.x - node.position.x;
        else if (axis == 1) diff = target.position.y - node.position.y;
        else diff = target.position.z - node.position.z;
		
        uint32_t near_child = (diff < 0) ? 2 * node_idx + 1 : 2 * node_idx + 2;
        uint32_t far_child = (diff < 0) ? 2 * node_idx + 2 : 2 * node_idx + 1;
		
		if (near_child < kdTree.grid_size) stack.push_back(near_child);
		
        if (fabsf(diff) < radius && far_child < kdTree.grid_size) {
            stack.push_back(far_child);
        }
				
	}

    KNNResult result;
    result.K = K;
    result.indices = indices;
    result.distances = distances;
    return result;

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