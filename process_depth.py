import numpy as np
import os
import pandas as pd
import csv
import math

input_csv = '/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/output/AO_order.csv'
output_csv = '/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/output/AO_order_w_depth.csv'
depth = np.load('/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/output/AO_order.npy')
summary_output_csv = '/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/output/AO_order_summary_stat.csv'
headers = ["ID", "x_head", "y_head", "x_tail","y_tail","z_head","z_tail","body_length","heading_x","heading_y","heading_z","x_mid","y_mid","z_mid",]

updated_data = []
with open(output_csv, mode="w", newline="") as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(headers)

    with open(input_csv) as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            x_head = int(float(row[1].strip()))
            y_head = int(float(row[2].strip()))
            z_head = depth[x_head,y_head]

            x_tail = int(float(row[3].strip()))
            y_tail = int(float(row[4].strip()))
            z_tail = depth[x_tail,y_tail]

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

            updated_row = row + [z_head, z_tail, 
                body_length, 
                heading_x, heading_y, heading_z,
                x_mid, y_mid, z_mid
            ]

            updated_data.append(updated_row)

            writer.writerow(updated_row)

    print("individual metrics calculated!")

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

    #Average heading of school
    summed_x = updated_data["heading_x"].sum()
    summed_y = updated_data["heading_y"].sum()
    summed_z = updated_data["heading_z"].sum()

    polarisation = math.sqrt(summed_x**2 + summed_y**2 + summed_z**2)/updated_data["heading_x"].count()

    updated_row = [median_bl, centre_x, centre_y, centre_z, polarisation]
    summary_data.append(updated_row)
    writer.writerow(updated_row)

print("summary data calculated!")

rows = []
updated_data = []
headers = ["ID", "x_head", "y_head", "x_tail","y_tail","z_head","z_tail","body_length","heading_x","heading_y","heading_z","x_mid","y_mid","z_mid","dist_from_centre","NND","heading_nn","heading_rel_to_group"]
       
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
        fx = float(focal[11])
        fy = float(focal[12])
        fz = float(focal[13])
        hi_x = float(focal[8])  
        hi_y = float(focal[9])
        hi_z = float(focal[10])

        #Distance from centre of school
        dist_from_centre = math.sqrt((fx - centre_x)**2 + (fy - centre_y)**2 + (fz - centre_z)**2)
        norm_dist_from_centre = dist_from_centre/median_bl
        
        min_nnd = float("inf")

        for j, other in enumerate(rows):
            if i == j:
                continue  # skip self

            ox = float(other[11])
            oy = float(other[12])
            oz = float(other[13])

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
        hj_x = float(rows[nnd_id][8])
        hj_y = float(rows[nnd_id][9])
        hj_z = float(rows[nnd_id][10])

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

print("inter-individual metrics calculated!")


