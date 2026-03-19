import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image

def plot_annotations(image_path, csv_path):
    img = Image.open(image_path)
    df = pd.read_csv(csv_path)

    # ensure numeric
    df["x"] = pd.to_numeric(df["x"])
    df["y"] = pd.to_numeric(df["y"])

    fig, ax = plt.subplots(figsize=(10, 8))
    ax.imshow(img)

    # match image coordinates (important!)
    ax.set_xlim(0, img.width)
    ax.set_ylim(img.height, 0)

    for _, row in df.iterrows():
        x, y = row["x"], row["y"]
        label = str(row["ObjType"])

        # point
        ax.scatter(x, y, c="red", s=30)

        # label
        ax.text(x + 3, y + 3, label, color="yellow", fontsize=8)

    ax.axis("off")
    plt.tight_layout()
    plt.show()


# Example usage
plot_annotations(
    "/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/testing/Tarpon_010324_3_sl4_sh.jpg",
    "/Users/ellag/Desktop/PhD/academic_projects/tang_bss_schooling/testing/Tarpon_010324_3_sl4_sh_annotations.csv"
)