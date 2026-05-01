import numpy as np
import os
import pandas as pd
import csv
import math
from collections import defaultdict
import argparse
from pathlib import Path

headers = ["image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail","body_length","heading_x","heading_y","heading_z","x_mid","y_mid","z_mid"]

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
    filename_clean = os.path.splitext(filename)[0]

    output_csv = os.path.join(output_path, filename_clean + "_individual.csv")

    input_csv = os.path.join(annotations_path, filename_clean + "_annotations.csv")
    
    depth = np.load(os.path.join(depth_path, filename_clean + ".npy"))

    summary_output_csv = os.path.join(output_path, filename_clean + "_summary.csv")

    count += 1

    headers = ["image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail","body_length","heading_x","heading_y","heading_z","x_mid","y_mid","z_mid"]

    updated_data = []
    temp_data = []

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

                temp_row = [count, obj_id, x_head, y_head, x_tail, y_tail, z_head, z_tail]

                temp_data.append(temp_row)

        print("Raw Z indexed for head and tail...")

        temp_data = pd.DataFrame(temp_data, columns=["image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail"])

        #Centre x y z head
        x_centred = temp_data["x_head"] - temp_data["x_head"].mean()
        y_centred = temp_data["y_head"] - temp_data["y_head"].mean()
        z_centred = temp_data["z_head"] - temp_data["z_head"].mean()

        spr_x_centred = x_centred.std()
        spr_y_centred = y_centred.std()
        spr_z_centred = z_centred.std()

        #average xy spread
        spr_xy = 0.5 * (spr_x_centred + spr_y_centred)

        #scale factor z
        sf_z = spr_xy/spr_z_centred

        #Calculate scaled z head
        z_head_scaled = sf_z * z_centred

        #Calculate scaled z tail       
        z_centred_tail = temp_data["z_tail"] - temp_data["z_tail"].mean()
        z_tail_scaled = sf_z * z_centred_tail

        #Add to dataframe
        temp_data["z_head_scaled"] = z_head_scaled
        temp_data["z_tail_scaled"] = z_tail_scaled

        #Convert to list of dicts for easy row access
        for index, row in temp_data.iterrows():
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

            updated_row = [count, obj_id, x_head, y_head, x_tail, y_tail, z_head, z_tail, 
                body_length, 
                heading_x, heading_y, heading_z,
                x_mid, y_mid, z_mid
            ]

            updated_data.append(updated_row)

            writer.writerow(updated_row)

        print("individual metrics calculated for", filename_clean)

    updated_data = pd.DataFrame(updated_data, columns=headers)

    headers_summary = ["median_bl","centre_x","centre_y","centre_z","polarisation"]
    summary_data = []
    with open(summary_output_csv, mode='w') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headers_summary)
        #median body length (scale every distance from this)
        median_bl = updated_data["body_length"].median()

        #Centre of  
        centre_x = updated_data["x_mid"].mean()   
        centre_y = updated_data["y_mid"].mean()
        centre_z = updated_data["z_mid"].mean()

        #Sum up all the unit vectors and divide by count - basically average heading of school
        summed_x = updated_data["heading_x"].sum()/updated_data["heading_x"].count()
        summed_y = updated_data["heading_y"].sum()/updated_data["heading_y"].count()
        summed_z = updated_data["heading_z"].sum()/updated_data["heading_z"].count()

        #Compute the magnitude of the averaged vector. yields a scalar value between 0 and 1
        polarisation = math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)

        updated_row = [median_bl, centre_x, centre_y, centre_z, polarisation]
        summary_data.append(updated_row)
        writer.writerow(updated_row)

    print("summary data calculated for", filename_clean)

    rows = []
    updated_data = []
    headers = ["image_ID", "individual_ID","x_head", "y_head", "x_tail","y_tail","z_head","z_tail","body_length","heading_x","heading_y","heading_z","x_mid","y_mid","z_mid","dist_from_centre","NND","heading_nn","heading_rel_to_group"]
       
    with open(output_csv, "r") as csvfile1:
        reader = csv.reader(csvfile1)
        header = next(reader)
        for row in reader:
            rows.append(row)

    with open(output_csv, "w", newline="") as csvfile2:
        writer = csv.writer(csvfile2)
        writer.writerow(headers)  # Write headers

        for i, focal in enumerate(rows):

            #NND distances between centres of axes
            fx = float(focal[12])
            fy = float(focal[13])
            fz = float(focal[14])
            hi_x = float(focal[9])  
            hi_y = float(focal[10])
            hi_z = float(focal[11])

            #Distance from centre of school
            dist_from_centre = math.sqrt((fx - centre_x)**2 + (fy - centre_y)**2 + (fz - centre_z)**2)
            norm_dist_from_centre = dist_from_centre/median_bl
        
            min_nnd = float("inf")

            for j, other in enumerate(rows):
                if i == j:
                    continue  # skip self

                ox = float(other[12])
                oy = float(other[13])
                oz = float(other[14])

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
            hj_x = float(rows[nnd_id][9])
            hj_y = float(rows[nnd_id][10])
            hj_z = float(rows[nnd_id][11])

            # heading alignment (dot product, headings are unit vectors)
            heading_nn = hi_x * hj_x + hi_y * hj_y + hi_z * hj_z

            #heading relative to group average
            Px = summed_x / math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)
            Py = summed_y / math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)
            Pz = summed_z / math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)

            heading_group = hi_x*Px + hi_y*Py + hi_z*Pz

            # Append new metrics to the row
            enriched_row = focal + [norm_dist_from_centre, norm_nnd, heading_nn, heading_group]
            updated_data.append(enriched_row)

            # Write row to CSV
            writer.writerow(enriched_row)


    # Calculate group cohesion
    df = pd.DataFrame(updated_data, columns=headers)
    group_cohesion = df["dist_from_centre"].median()

    # Update the first (and only) row in summary_data by appending group_cohesion
    summary_data[0].append(group_cohesion)

    # Now write the summary CSV with updated header and row
    headers2 = ["median_bl","centre_x","centre_y","centre_z","polarisation","group_cohesion"]
    with open(summary_output_csv, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers2)
        writer.writerow(summary_data[0])

    print("inter-individual metrics calculated for",filename_clean)


