import numpy as np
import os
import pandas as pd
import csv
import math
from collections import defaultdict
import argparse
from pathlib import Path
import glob


headers = [
    "image_name", "image_ID", "individual_ID",
    "x_head", "y_head", "x_tail", "y_tail",
    "z_head", "z_tail",
    "body_length",
    "heading_x", "heading_y", "heading_z",
    "x_mid", "y_mid", "z_mid"
]

updated_data = []


parser = argparse.ArgumentParser(
    description="Create chunking strategy for dense reconstruction"
)

parser.add_argument(
    "--depth_path",
    type=str,
    required=True,
    help="Path to folder with depth maps"
)

parser.add_argument(
    "--annotations_path",
    type=str,
    required=True,
    help="Path to folder with annotations"
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help="Path to output folder"
)

args = parser.parse_args()

depth_path = Path(args.depth_path)
annotations_path = Path(args.annotations_path)
output_path = Path(args.output_path)

count = 0


for filename in os.listdir(depth_path):

    if not filename.lower().endswith(".npy"):
        continue

    filename_clean = os.path.splitext(filename)[0]

    output_csv = os.path.join(
        output_path,
        filename_clean + "_individual.csv"
    )

    output_nnd_csv = os.path.join(
        output_path,
        filename_clean + "_individual_nnd.csv"
    )

    input_csv = os.path.join(
        annotations_path,
        filename_clean + "_annotations.csv"
    )

    depth = np.load(
        os.path.join(
            depth_path,
            filename_clean + ".npy"
        )
    )

    summary_output_csv = os.path.join(
        output_path,
        filename_clean + "_summary.csv"
    )

    count += 1

    headers = [
        "image_name", "image_ID", "individual_ID",
        "x_head", "y_head", "x_tail", "y_tail",
        "z_head", "z_tail",
        "body_length",
        "heading_x", "heading_y", "heading_z",
        "x_mid", "y_mid", "z_mid"
    ]

    updated_data = []
    temp_data = []


    # ============================================================
    # CALCULATE Z SCALING FROM THE WHOLE DEPTH-MAP POINT CLOUD
    # ============================================================

    # Create x/y coordinates for every pixel
    y_coords, x_coords = np.indices(depth.shape)

    # Only use finite depth values
    valid = np.isfinite(depth)

    x_points = x_coords[valid].astype(float)
    y_points = y_coords[valid].astype(float)
    z_points = depth[valid].astype(float)

    # Check that there are valid depth values
    if len(z_points) == 0:
        print(
            "No valid depth values for",
            filename_clean,
            "- skipping image."
        )
        continue

    # Spatial variation of the entire reconstructed point cloud
    spr_x = np.std(x_points)
    spr_y = np.std(y_points)
    spr_z = np.std(z_points)

    # Average spatial variation in x and y
    spr_xy = 0.5 * (spr_x + spr_y)

    # Avoid division by zero
    if spr_z == 0:
        print(
            "Z variation is zero for",
            filename_clean,
            "- skipping image."
        )
        continue

    # Image-specific multiplicative scale factor for z
    sf_z = spr_xy / spr_z

    # Centre z using the mean depth of the entire point cloud
    z_reference = np.mean(z_points)

    # Create scaled depth map
    #
    # x and y remain in pixel coordinates.
    # z is rescaled so that its spatial variation is
    # comparable to x and y.
    depth_scaled = sf_z * (depth - z_reference)

    print("Whole-cloud depth scaling for", filename_clean)
    print("  X spread:", spr_x)
    print("  Y spread:", spr_y)
    print("  Z spread:", spr_z)
    print("  XY spread:", spr_xy)
    print("  Z scale factor:", sf_z)


    # ============================================================
    # READ ANNOTATIONS AND EXTRACT HEAD/TAIL Z VALUES
    # ============================================================

    with open(output_csv, mode="w", newline="") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(headers)

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
                        head = (x, y)

                    elif obj_type == "Tail":
                        tail = (x, y)


                if head is not None:

                    x_head, y_head = head

                    # Extract Z from the WHOLE-CLOUD-SCALED depth map
                    z_head = depth_scaled[y_head, x_head]

                else:

                    z_head = None


                if tail is not None:

                    x_tail, y_tail = tail

                    # Extract Z from the WHOLE-CLOUD-SCALED depth map
                    z_tail = depth_scaled[y_tail, x_tail]

                else:

                    z_tail = None


                if head is None or tail is None:
                    continue


                temp_row = [
                    filename_clean,
                    count,
                    obj_id,
                    x_head,
                    y_head,
                    x_tail,
                    y_tail,
                    z_head,
                    z_tail
                ]

                temp_data.append(temp_row)


        print("Whole-cloud scaled Z indexed for head and tail...")


        temp_data = pd.DataFrame(
            temp_data,
            columns=[
                "image_name",
                "image_ID",
                "individual_ID",
                "x_head",
                "y_head",
                "x_tail",
                "y_tail",
                "z_head",
                "z_tail"
            ]
        )

        temp_data["individual_ID"] = (
            temp_data["individual_ID"].astype(str)
        )


        # ============================================================
        # CALCULATE INDIVIDUAL 3D METRICS
        # ============================================================

        for index, row in temp_data.iterrows():

            x_head = row["x_head"]
            y_head = row["y_head"]

            x_tail = row["x_tail"]
            y_tail = row["y_tail"]

            z_head = row["z_head"]
            z_tail = row["z_tail"]

            obj_id = row["individual_ID"]


            # Body length in 3D
            body_length = math.sqrt(
                (x_head - x_tail) ** 2 +
                (y_head - y_tail) ** 2 +
                (z_head - z_tail) ** 2
            )


            # Heading vector
            heading_x = (
                (x_head - x_tail) / body_length
            )

            heading_y = (
                (y_head - y_tail) / body_length
            )

            heading_z = (
                (z_head - z_tail) / body_length
            )


            # Fish midpoint
            x_mid = (x_head + x_tail) / 2
            y_mid = (y_head + y_tail) / 2
            z_mid = (z_head + z_tail) / 2


            updated_row = [
                filename_clean,
                count,
                obj_id,

                x_head,
                y_head,

                x_tail,
                y_tail,

                z_head,
                z_tail,

                body_length,

                heading_x,
                heading_y,
                heading_z,

                x_mid,
                y_mid,
                z_mid
            ]

            updated_data.append(updated_row)

            writer.writerow(updated_row)


        print(
            "individual metrics calculated for",
            filename_clean
        )


    updated_data = pd.DataFrame(
        updated_data,
        columns=headers
    )


    # ============================================================
    # SUMMARY METRICS
    # ============================================================

    headers_summary = [
        "image_ID",
        "median_bl",
        "centre_x",
        "centre_y",
        "centre_z",
        "polarisation",
        "mid_back_x",
        "mid_back_y",
        "mid_back_z",
        "mid_high_x",
        "mid_high_y",
        "mid_high_z"
    ]

    summary_data = []


    # Filter out floor / non-fish IDs
    filtered_data = updated_data[
        updated_data["individual_ID"]
        .astype(str)
        .str.len() < 3
    ]


    with open(summary_output_csv, mode="w") as csvfile:

        writer = csv.writer(csvfile)
        writer.writerow(headers_summary)


        # Median body length
        median_bl = filtered_data["body_length"].median()


        # Centre of school
        centre_x = filtered_data["x_mid"].mean()
        centre_y = filtered_data["y_mid"].mean()
        centre_z = filtered_data["z_mid"].mean()


        # Mean school heading
        summed_x = (
            filtered_data["heading_x"].sum()
            / filtered_data["heading_x"].count()
        )

        summed_y = (
            filtered_data["heading_y"].sum()
            / filtered_data["heading_y"].count()
        )

        summed_z = (
            filtered_data["heading_z"].sum()
            / filtered_data["heading_z"].count()
        )


        # Polarisation
        polarisation = math.sqrt(
            summed_x ** 2 +
            summed_y ** 2 +
            summed_z ** 2
        )


        # Normalize group heading
        heading_magnitude = math.sqrt(
            summed_x ** 2 +
            summed_y ** 2 +
            summed_z ** 2
        )

        Px = summed_x / heading_magnitude
        Py = summed_y / heading_magnitude
        Pz = summed_z / heading_magnitude


        # Position relative to school centre
        dx = filtered_data["x_mid"] - centre_x
        dy = filtered_data["y_mid"] - centre_y
        dz = filtered_data["z_mid"] - centre_z


        filtered_data = filtered_data.copy()


        filtered_data["position_along_heading"] = (
            dx * Px +
            dy * Py +
            dz * Pz
        )


        # Fish furthest behind school centre
        back_idx = (
            filtered_data["position_along_heading"].idxmin()
        )

        mid_back_x = filtered_data.loc[
            back_idx, "x_mid"
        ]

        mid_back_y = filtered_data.loc[
            back_idx, "y_mid"
        ]

        mid_back_z = filtered_data.loc[
            back_idx, "z_mid"
        ]

        back_individual_ID = filtered_data.loc[
            back_idx, "individual_ID"
        ]


        # Highest individual
        mid_high_y = filtered_data["y_mid"].max()

        mid_high_x = filtered_data["x_mid"][
            filtered_data["y_mid"] == mid_high_y
        ].values[0]

        mid_high_z = filtered_data["z_mid"][
            filtered_data["y_mid"] == mid_high_y
        ].values[0]


        updated_row = [
            count,
            median_bl,
            centre_x,
            centre_y,
            centre_z,
            polarisation,

            mid_back_x,
            mid_back_y,
            mid_back_z,

            mid_high_x,
            mid_high_y,
            mid_high_z
        ]

        summary_data.append(updated_row)

        writer.writerow(updated_row)


    print(
        "summary data calculated for",
        filename_clean
    )


    # ============================================================
    # INTER-INDIVIDUAL METRICS
    # ============================================================

    rows = []
    updated_data = []


    headers = [
        "image_name",
        "image_ID",
        "individual_ID",
        "x_head",
        "y_head",
        "x_tail",
        "y_tail",
        "z_head",
        "z_tail",
        "body_length",
        "heading_x",
        "heading_y",
        "heading_z",
        "x_mid",
        "y_mid",
        "z_mid",
        "median_body_length",
        "dist_from_centre",
        "NND",
        "heading_nn",
        "heading_rel_to_group",
        "back_ind",
        "highest_ind",
        "mid_back_x",
        "mid_back_y",
        "mid_back_z",
        "mid_high_x",
        "mid_high_y",
        "mid_high_z",
        "dist_to_back",
        "dist_to_highest",
        "norm_dist_to_back",
        "norm_dist_to_highest"
    ]


    rows = filtered_data.to_dict("records")


    for i, focal in enumerate(rows):

        # Focal fish position
        fx = float(focal["x_mid"])
        fy = float(focal["y_mid"])
        fz = float(focal["z_mid"])

        hi_x = float(focal["heading_x"])
        hi_y = float(focal["heading_y"])
        hi_z = float(focal["heading_z"])


        # Distance from centre of school
        dist_from_centre = math.sqrt(
            (fx - centre_x) ** 2 +
            (fy - centre_y) ** 2 +
            (fz - centre_z) ** 2
        )

        norm_dist_from_centre = (
            dist_from_centre / median_bl
        )


        # ========================================================
        # NEAREST NEIGHBOUR
        # ========================================================

        min_nnd = float("inf")
        nnd_id = None


        for j, other in enumerate(rows):

            if i == j:
                continue


            ox = float(other["x_mid"])
            oy = float(other["y_mid"])
            oz = float(other["z_mid"])


            dist = math.sqrt(
                (fx - ox) ** 2 +
                (fy - oy) ** 2 +
                (fz - oz) ** 2
            )


            if dist < min_nnd:

                min_nnd = dist
                nnd_id = j


        norm_nnd = min_nnd / median_bl


        # Nearest neighbour heading
        hj_x = float(rows[nnd_id]["heading_x"])
        hj_y = float(rows[nnd_id]["heading_y"])
        hj_z = float(rows[nnd_id]["heading_z"])


        # Heading alignment
        heading_nn = (
            hi_x * hj_x +
            hi_y * hj_y +
            hi_z * hj_z
        )


        # Heading relative to group
        Px = (
            summed_x /
            math.sqrt(
                summed_x ** 2 +
                summed_y ** 2 +
                summed_z ** 2
            )
        )

        Py = (
            summed_y /
            math.sqrt(
                summed_x ** 2 +
                summed_y ** 2 +
                summed_z ** 2
            )
        )

        Pz = (
            summed_z /
            math.sqrt(
                summed_x ** 2 +
                summed_y ** 2 +
                summed_z ** 2
            )
        )


        heading_group = (
            hi_x * Px +
            hi_y * Py +
            hi_z * Pz
        )


        # ========================================================
        # BACK INDIVIDUAL
        # ========================================================

        if str(focal["individual_ID"]) == str(back_individual_ID):
            back_ind = 1
        else:
            back_ind = 0


        # Distance to back
        dist_from_back = math.sqrt(
            (fx - mid_back_x) ** 2 +
            (fy - mid_back_y) ** 2 +
            (fz - mid_back_z) ** 2
        )

        norm_dist_from_back = (
            dist_from_back / median_bl
        )


        # ========================================================
        # HIGHEST INDIVIDUAL
        # ========================================================

        if fy == mid_high_y:
            highest_ind = 1
        else:
            highest_ind = 0


        # Distance to highest
        dist_from_highest = math.sqrt(
            (fx - mid_high_x) ** 2 +
            (fy - mid_high_y) ** 2 +
            (fz - mid_high_z) ** 2
        )

        norm_dist_from_highest = (
            dist_from_highest / median_bl
        )


        # Add metrics
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

        focal["dist_to_back"] = dist_from_back

        focal["dist_to_highest"] = dist_from_highest

        focal["norm_dist_to_back"] = norm_dist_from_back

        focal["norm_dist_to_highest"] = norm_dist_from_highest


        updated_data.append(focal)


    # ============================================================
    # WRITE INDIVIDUAL NND FILE
    # ============================================================

    with open(output_nnd_csv, "w", newline="") as csvfile:

        writer = csv.DictWriter(
            csvfile,
            fieldnames=headers,
            extrasaction="ignore"
        )

        writer.writeheader()
        writer.writerows(updated_data)


    # ============================================================
    # GROUP COHESION
    # ============================================================

    df = pd.DataFrame(
        updated_data,
        columns=headers
    )

    group_cohesion = (
        df["dist_from_centre"].median()
    )


    # Add group cohesion
    summary_data[0].append(group_cohesion)


    headers2 = [
        "image_ID",
        "median_bl",
        "centre_x",
        "centre_y",
        "centre_z",
        "polarisation",
        "mid_back_x",
        "mid_back_y",
        "mid_back_z",
        "mid_high_x",
        "mid_high_y",
        "mid_high_z",
        "group_cohesion"
    ]


    with open(
        summary_output_csv,
        mode="w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow(headers2)

        writer.writerow(summary_data[0])


    print(
        "inter-individual metrics calculated for",
        filename_clean
    )


# ================================================================
# COMBINE ALL SUMMARY FILES
# ================================================================

summary_files = glob.glob(
    os.path.join(
        output_path,
        "*summary.csv"
    )
)

summary_dfs = []


for filename in summary_files:

    df = pd.read_csv(
        filename,
        index_col=False
    )

    summary_dfs.append(df)


df_out_summary = pd.concat(
    summary_dfs,
    axis=0,
    ignore_index=False
)

df_out_summary.to_csv(
    f"{output_path}/summary_global.csv"
)


# ================================================================
# COMBINE ALL INDIVIDUAL FILES
# ================================================================

individual_files = glob.glob(
    os.path.join(
        output_path,
        "*individual_nnd.csv"
    )
)

individual_dfs = []


for filename in individual_files:

    df = pd.read_csv(
        filename,
        index_col=False
    )

    individual_dfs.append(df)


df_out_individual = pd.concat(
    individual_dfs,
    axis=0,
    ignore_index=False
)

df_out_individual.to_csv(
    f"{output_path}/individual_global.csv"
)
