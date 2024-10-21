from cjio import cityjson
from copy import deepcopy
import os
import json
import subprocess
import shutil
from pathlib import Path
import argparse

def check_extension(filename, extension):
    return Path(filename).suffix.lower() == extension.lower()



def assign_cityobject_attribute(cm):
        """assign the semantic surface with new attribute global_idx.
        Returns a copy of the citymodel.
        """

        global_idx = 0
        cm_copy = deepcopy(cm)
        tree_idx = 0

        for i, cityobject in enumerate(cm_copy['CityObjects'].values()):
            if cityobject['type'] == 'BuildingPart' or cityobject['type'] == 'Building':
                for geom_idx in range(len(cityobject['geometry'])):
                    # breakpoint()
                    if cityobject['geometry'][geom_idx]["lod"]==2:
                        semantics = cityobject['geometry'][geom_idx]['semantics']
                        new_semantics = {}
                        n_surfaces = []
                        n_values = []

                        for i in range(len(semantics['values'])):
                            sem_value = semantics['values'][i]
                            new_surface_semantic = {}
                            previous_surf_semantic = semantics['surfaces'][sem_value]['type']
                            new_surface_semantic['type'] = previous_surf_semantic

                            new_surface_semantic['global_idx'] = global_idx
                            n_surfaces.append(new_surface_semantic)
                            n_values.append(i)
                            global_idx += 1

                        new_semantics['surfaces'] = n_surfaces
                        new_semantics['values'] = n_values
                        cityobject['geometry'][geom_idx]['semantics'] = new_semantics

            elif cityobject['type']=='TINRelief':
                for geom_idx in range(len(cityobject['geometry'])):
                    num_triangles = len(cityobject['geometry'][geom_idx]['boundaries'])

                    new_semantics = {}
                    n_surfaces = []
                    n_values = []

                    for i in range(num_triangles):
                        new_surf_semantic = {}
                        new_surf_semantic['global_idx'] = global_idx
                        n_surfaces.append(new_surf_semantic)
                        n_values.append(i)
                        global_idx += 1
                    
                    new_semantics['surfaces'] = n_surfaces
                    new_semantics['values'] = n_values
                    cityobject['geometry'][geom_idx]['semantics'] = new_semantics
            elif cityobject['type'] == 'SolitaryVegetationObject':
                attributes = cityobject['attributes']
                attributes['global_idx'] = global_idx
                geom_attri = {}
                geom_attri["type"] = "GeometryInstance"
                geom_attri["template"]=0
                geom_attri["boundaries"]=[tree_idx]
                geom_attri["transformationMatrix"]=[2.42, 0.0, 0.0, 0.0, 0.0, 2.42, 0.0, 0.0, 0.0, 0.0, 1.6, 0.0, 0.0, 0.0, 0.0, 1.0]
                cityobject["geometry"].append(geom_attri)
                cityobject['attributes'] = attributes
                tree_idx+=1
                global_idx += 1



        return cm_copy



if __name__ == '__main__':
    # current_file_path = os.path.abspath(__file__)
    # current_dir = os.path.dirname(current_file_path)
    # parent_dir = os.path.dirname(current_dir)
    # os.chdir(parent_dir)

    parser = argparse.ArgumentParser(description='处理命令行参数')

    parser.add_argument('filename', help='the file name of the cityjson file')
    parser.add_argument('output', help='the file name of the output cityjson file')
    args = parser.parse_args()

    filename = args.filename
    if os.path.exists(filename) == False:
        print(f"File {filename} does not exist.")
        exit()

    print(f"Processing cityjson file {filename}")

    cityjson_file = json.load(open(filename))

    enriched_cityjson = assign_cityobject_attribute(cityjson_file)

    with open(args.output, 'w') as f:
        json.dump(enriched_cityjson, f)

    print(f"Processed cityjson file saved to {args.output}")