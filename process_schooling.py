import numpy as np
import os
import pandas as pd
import csv
import math
from collections import defaultdict
import argparse
from pathlib import Path
import glob

headers = ["image_name","image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail","body_length","heading_x","heading_y","heading_z","x_mid","y_mid","z_mid"]

updated_data = []

    
parser = argparse.ArgumentParser(description='Create chunking strategy for dense reconstruction')
parser.add_argument('--depth_path', type=str, required=True, help='Path to folder with depth maps')
parser.add_argument('--annotations_path', type=str, required=True, help='Path to folder with annotations')
parser.add_argument('--output_path', type=str, required=True, help='Path to output folder')

args = parser.parse_args()
    
depth_path = Path(args.depth_path)
annotations_path = Path(args.annotations_path)
output_path = Path(args.output_path)

#depth_path = '/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/testing/depth_maps'
#annotations_path = '/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/testing/annotations'
#output_path = '/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/testing/output'
count = 0

for filename in os.listdir(depth_path):

    if not filename.lower().endswith(".npy"):
        continue

    filename_clean = os.path.splitext(filename)[0]

    output_csv = os.path.join(output_path, filename_clean + "_individual.csv")

    output_nnd_csv = os.path.join(output_path, filename_clean + "_individual_nnd.csv")

    input_csv = os.path.join(annotations_path, filename_clean + "_annotations.csv")
    
    depth = np.load(os.path.join(depth_path, filename_clean + ".npy"))

    summary_output_csv = os.path.join(output_path, filename_clean + "_summary.csv")

    count += 1

    headers = ["image_name","image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail","body_length","heading_x","heading_y","heading_z","x_mid","y_mid","z_mid", "floor_distance"]

    updated_data = []
    temp_data = []
    fish_temp_data = []
    floor_temp_data = []

    #Calculate depth on all
    with open(output_csv, mode="w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers)
        image_id = 1

        with open(input_csv) as f:
            reader = csv.reader(f)
            header = next(reader)

            groups = defaultdict(list)

            for row in reader:
                obj_id = row[2].strip()
                groups[obj_id].append(row)

            for obj_id, rows in groups.items():
                head = None
                tail = None

                #if len(obj_id) > 3:
                #    continue

                for row in rows:
                    obj_type = row[3].strip()

                    x = int(float(row[5].strip()))
                    y = int(float(row[6].strip()))

                    if obj_type == "Head":
                        head = (x,y)
                    elif obj_type == "Tail":
                        tail = (x,y)

                if head is not None:
                    x_head, y_head = head
                    z_head = depth[y_head, x_head]
                else:
                    z_head = None

                if tail is not None:
                    x_tail, y_tail = tail
                    z_tail = depth[y_tail, x_tail]
                else:
                    z_tail = None

                if head is None or tail is None:
                    continue
                
                if obj_id.startswith("floor"):
                    floor_temp_data.append([
                        filename_clean, count, obj_id, x_head, y_head, x_tail, y_tail, z_head, z_tail
                    ])

                if len(obj_id) < 3:
                    fish_temp_data.append([
                        filename_clean, count, obj_id, x_head, y_head, x_tail, y_tail, z_head, z_tail
                    ])



                #temp_row = [filename_clean, count, obj_id, x_head, y_head, x_tail, y_tail, z_head, z_tail]

                #temp_data.append(temp_row)

        print("Raw Z indexed for head and tail...")

        fish_temp_data = pd.DataFrame(fish_temp_data, columns=["image_name","image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail"])
        floor_temp_data = pd.DataFrame(floor_temp_data, columns=["image_name","image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail"])

        fish_temp_data["individual_ID"] = fish_temp_data["individual_ID"].astype(str)

        filtered_temp = fish_temp_data[fish_temp_data["individual_ID"].str.len() < 3]

        #Centre x y z head
        x_centred_head = filtered_temp["x_head"] - filtered_temp["x_head"].mean()
        y_centred_head = filtered_temp["y_head"] - filtered_temp["y_head"].mean()
        z_centred_head = filtered_temp["z_head"] - filtered_temp["z_head"].mean()

        spr_x_centred_head = x_centred_head.std()
        spr_y_centred_head = y_centred_head.std()
        spr_z_centred_head = z_centred_head.std()

        #average xy spread
        spr_xy = 0.5 * (spr_x_centred_head + spr_y_centred_head)
        #scale factor z
        sf_z = spr_xy/spr_z_centred_head

        #Calculate scaled z head
        z_centred_head = fish_temp_data["z_head"] - filtered_temp["z_head"].mean()
        z_head_scaled = sf_z * z_centred_head

        #Calculate scaled z tail       
        z_centred_tail = fish_temp_data["z_tail"] - filtered_temp["z_head"].mean()
        z_tail_scaled = sf_z * z_centred_tail

        #Add to dataframe
        fish_temp_data["z_head_scaled"] = z_head_scaled
        fish_temp_data["z_tail_scaled"] = z_tail_scaled

        #scale floor
        floor_temp_data["z_head_scaled"] = (
            sf_z * (
                floor_temp_data["z_head"] - 
                filtered_temp["z_head"].mean()
            )
        )

        floor_temp_data["z_tail_scaled"] = (
            sf_z * (
                floor_temp_data["z_tail"] - 
                filtered_temp["z_head"].mean()
            )
        )

        #compute floor plane
        floor_x = np.concatenate([
            floor_temp_data["x_head"].to_numpy(),
            floor_temp_data["x_tail"].to_numpy()
        ])

        floor_y = np.concatenate([
            floor_temp_data["y_head"].to_numpy(),
            floor_temp_data["y_tail"].to_numpy()
        ])

        floor_z = np.concatenate([
            floor_temp_data["z_head_scaled"].to_numpy(),
            floor_temp_data["z_tail_scaled"].to_numpy()
        ])



        A = np.c_[floor_x, floor_y, np.ones(len(floor_x))]
        a,b,c = np.linalg.lstsq(A, floor_z, rcond = None)[0]

        #Convert to list of dicts for easy row access
        for index, row in fish_temp_data.iterrows():
            x_head = row["x_head"]
            y_head = row["y_head"]
            x_tail = row["x_tail"]
            y_tail = row["y_tail"]
            z_head = row["z_head_scaled"]
            z_tail = row["z_tail_scaled"]
            obj_id = row["individual_ID"]

            #body length
            body_length = math.sqrt((x_head - x_tail)**2 + (y_head - y_tail)**2 + (z_head - z_tail)**2)

            #vector
            heading_x = (x_head - x_tail)/body_length
            heading_y = (y_head - y_tail)/body_length
            heading_z = (z_head - z_tail)/body_length

            #fish midpoint
            x_mid = (x_head+x_tail)/2
            y_mid = (y_head+y_tail)/2
            z_mid = (z_head+z_tail)/2

            #distance from floor
            floor_distance = abs(
                a * x_mid +
                b * y_mid -
                z_mid +
                c
            ) / math.sqrt(a**2 + b**2 + 1)

            updated_row = [filename_clean, count, obj_id, x_head, y_head, x_tail, y_tail, z_head, z_tail, 
                body_length, 
                heading_x, heading_y, heading_z,
                x_mid, y_mid, z_mid, floor_distance
            ]

            updated_data.append(updated_row)

            writer.writerow(updated_row)

        print("individual metrics calculated for", filename_clean)

    updated_data = pd.DataFrame(updated_data, columns=headers)

    headers_summary = ["image_ID","median_bl","centre_x","centre_y","centre_z","polarisation", "mid_back_x", "mid_back_y", "mid_back_z", "mid_high_x", "mid_high_y", "mid_high_z"]
    summary_data = []

    #filter out the floor! In an ideal world we want the depth to be created for the floor but then NA for the different metrics (so everything ends up in one dataframe)
    filtered_data = updated_data[updated_data["individual_ID"].astype(str).str.len() < 3]

    #Everything else only on filtered dataframe
    with open(summary_output_csv, mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers_summary)
        #median body length (scale every distance from this)
        median_bl = filtered_data["body_length"].median()

        #Centre of  
        centre_x = filtered_data["x_mid"].mean()   
        centre_y = filtered_data["y_mid"].mean()
        centre_z = filtered_data["z_mid"].mean()

        #Sum up all the unit vectors and divide by count - basically average heading of school
        summed_x = filtered_data["heading_x"].sum()/filtered_data["heading_x"].count()
        summed_y = filtered_data["heading_y"].sum()/filtered_data["heading_y"].count()
        summed_z = filtered_data["heading_z"].sum()/filtered_data["heading_z"].count()

        #Compute the magnitude of the averaged vector. yields a scalar value between 0 and 1
        polarisation = math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)

        #Back individual
        # Normalize the school's mean heading
        heading_magnitude = math.sqrt(
            summed_x**2 +
            summed_y**2 +
            summed_z**2
        )

        #unit vector / normalise it
        Px = summed_x / heading_magnitude
        Py = summed_y / heading_magnitude
        Pz = summed_z / heading_magnitude

        # Position of each fish relative to school centre
        dx = filtered_data["x_mid"] - centre_x
        dy = filtered_data["y_mid"] - centre_y
        dz = filtered_data["z_mid"] - centre_z

        # Project each fish onto the 3D school heading
        filtered_data = filtered_data.copy()

        #Dot product - position relative to school centre * schools heading - where the fish sits along the schools heading axos. each fish gets number along axis
        filtered_data["position_along_heading"] = (
            dx * Px +
            dy * Py +
            dz * Pz
        )

        position_along_heading = filtered_data["position_along_heading"]

        #Radial distance (perpendicular distance from centre line)
        radial_distance = np.sqrt(
            dx**2 + dy**2 + dz**2 - position_along_heading**2
        )

        filtered_data["radial_distance"] = radial_distance



        # Fish furthest BEHIND the school centre
        back_idx = filtered_data["position_along_heading"].idxmin()

        mid_back_x = filtered_data.loc[back_idx, "x_mid"]
        mid_back_y = filtered_data.loc[back_idx, "y_mid"]
        mid_back_z = filtered_data.loc[back_idx, "z_mid"]

        back_individual_ID = filtered_data.loc[back_idx, "individual_ID"]

        #Highest individual
        mid_high_y = filtered_data["y_mid"].max()
        mid_high_x = filtered_data["x_mid"][filtered_data["y_mid"] == mid_high_y].values[0]
        mid_high_z = filtered_data["z_mid"][filtered_data["y_mid"] == mid_high_y].values[0]
        

        updated_row = [count, median_bl, centre_x, centre_y, centre_z, polarisation, mid_back_x, mid_back_y, mid_back_z, mid_high_x, mid_high_y, mid_high_z]
        summary_data.append(updated_row)
        writer.writerow(updated_row)

    print("summary data calculated for", filename_clean)

    rows = []
    updated_data = []
    headers = ["image_name","image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail","body_length","heading_x","heading_y","heading_z","x_mid","y_mid","z_mid","floor_distance","norm_floor_distance","median_body_length","dist_from_centre","NND","heading_nn","heading_rel_to_group", "back_ind", "highest_ind", "mid_back_x", "mid_back_y", "mid_back_z", "mid_high_x", "mid_high_y", "mid_high_z", "position_along_heading", "radial_distance", "dist_to_back", "dist_to_highest", "norm_dist_to_back", "norm_dist_to_highest"]
    rows = filtered_data.to_dict("records")   

    for i, focal in enumerate(rows):

            #NND distances between centres of axes
            fx = float(focal["x_mid"])
            fy = float(focal["y_mid"])
            fz = float(focal["z_mid"])
            hi_x = float(focal["heading_x"])  
            hi_y = float(focal["heading_y"])
            hi_z = float(focal["heading_z"])

            #Distance from centre of school
            dist_from_centre = math.sqrt((fx - centre_x)**2 + (fy - centre_y)**2 + (fz - centre_z)**2)
            norm_dist_from_centre = dist_from_centre/median_bl

            norm_floor_distance = float(focal["floor_distance"])/median_bl
        
            min_nnd = float("inf")

            for j, other in enumerate(rows):
                if i == j:
                    continue  # skip self

                ox = float(other["x_mid"])
                oy = float(other["y_mid"])
                oz = float(other["z_mid"])

                dist = math.sqrt(
                    (fx - ox)**2 +
                    (fy - oy)**2 +
                    (fz - oz)**2
                    )

                if dist < min_nnd:
                    min_nnd = dist
                    nnd_id = j

            norm_nnd = min_nnd/median_bl

            # get nearest neighbour heading
            hj_x = float(rows[nnd_id]["heading_x"])
            hj_y = float(rows[nnd_id]["heading_y"])
            hj_z = float(rows[nnd_id]["heading_z"])

            # heading alignment (dot product, headings are unit vectors)
            heading_nn = hi_x * hj_x + hi_y * hj_y + hi_z * hj_z

            #heading relative to group average
            Px = summed_x / math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)
            Py = summed_y / math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)
            Pz = summed_z / math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)

            heading_group = hi_x*Px + hi_y*Py + hi_z*Pz

            #back individual
            if str(focal["individual_ID"]) == str(back_individual_ID):
                back_ind = 1
            else:
                back_ind = 0

            mid_back_x = mid_back_x
            mid_back_y = mid_back_y
            mid_back_z = mid_back_z


            #distance to back
            dist_from_back = math.sqrt((fx - mid_back_x)**2 + (fy - mid_back_y)**2 + (fz - mid_back_z)**2)
            norm_dist_from_back = dist_from_back/median_bl

            #high individual
            if fy == mid_high_y:
                highest_ind = 1
            else:
                highest_ind = 0

            mid_high_x = mid_high_x
            mid_high_y = mid_high_y
            mid_high_z = mid_high_z

            #distance to highest
            dist_from_highest = math.sqrt((fx - mid_high_x )**2 + (fy - mid_high_y)**2 + (fz - mid_high_z)**2)
            norm_dist_from_highest = dist_from_highest/median_bl

            # Append new metrics to the row
            focal["norm_floor_distance"] = norm_floor_distance
            focal["median_body_length"] = median_bl
            focal["dist_from_centre"] = norm_dist_from_centre
            focal["NND"] = norm_nnd
            focal["heading_nn"] = heading_nn
            focal["heading_rel_to_group"] = heading_group
            focal["back_ind"] = back_ind
            focal["highest_ind"] = highest_ind
            focal["mid_back_x"] = mid_back_x
            focal["mid_back_y"] = mid_back_y
            focal["mid_back_z"] = mid_back_z
            focal["mid_high_x"] = mid_high_x
            focal["mid_high_y"] = mid_high_y
            focal["mid_high_z"] = mid_high_z
            focal["position_along_heading"] = float(focal["position_along_heading"])
            focal["radial_distance"] = float(focal["radial_distance"])
            focal["dist_to_back"] = dist_from_back
            focal["dist_to_highest"] = dist_from_highest
            focal["norm_dist_to_back"] = norm_dist_from_back
            focal["norm_dist_to_highest"] = norm_dist_from_highest
            

            # Append the updated dictionary to your list
            updated_data.append(focal)  

    with open(output_nnd_csv, "w", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=headers, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(updated_data)

    # Calculate group cohesion
    df = pd.DataFrame(updated_data, columns=headers)
    group_cohesion = df["dist_from_centre"].median()

    # Update the first (and only) row in summary_data by appending group_cohesion
    summary_data[0].append(group_cohesion)

    # Now write the summary CSV with updated header and row
    headers2 = ["image_ID","median_bl","centre_x","centre_y","centre_z","polarisation","mid_back_x", "mid_back_y", "mid_back_z", "mid_high_x", "mid_high_y", "mid_high_z", "group_cohesion"]
    with open(summary_output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers2)
        writer.writerow(summary_data[0])

    print("inter-individual metrics calculated for", filename_clean)

#Write all summary files to one file
summary_files = glob.glob(
    os.path.join(output_path, "*summary.csv")
)

summary_dfs = []

for filename in summary_files:
    df = pd.read_csv(filename, index_col=False)
    summary_dfs.append(df)

df_out_summary = pd.concat(summary_dfs, axis=0, ignore_index=False)
df_out_summary.to_csv(f'{output_path}/summary_global.csv')

#Write all individual files to one file
individual_files = glob.glob(
    os.path.join(output_path, "*individual_nnd.csv")
)

individual_dfs = []

for filename in individual_files:
    df = pd.read_csv(filename, index_col=False)
    individual_dfs.append(df)

df_out_individual = pd.concat(individual_dfs, axis=0, ignore_index=False)
df_out_individual.to_csv(f'{output_path}/individual_global.csv')



