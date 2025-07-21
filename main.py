from facial_tinting import *
import cv2
import os
import pandas as pd
from itertools import product
from tqdm.auto import tqdm
from pathlib import Path
def find_source_images(root):
    """Return a list of Path objects for every existing Mx.png."""
    root = Path(root)
    pattern = [f"M{i}.png" for i in range(1, 11)]
    paths   = []
    for gender in Genders:
        for ethnicity in Ethnicity:
            for age in Age_Ranges:
                folder = root / gender / ethnicity / age
                for p in pattern:
                    file = folder / p
                    if file.exists():
                        paths.append(file)
    return paths
# Gender -> Race -> Age Range -> M1 .. M12
Genders = ["Male", "Female"]
Ethnicity = ["White", "Black", "Asian"]
Age_Ranges = ["18-24", "25-34", "35-44", "45-54", "55-64"]
def parseMorphs(inputFolder, outputFolder, LAB_Color_Tint):
    step_L, step_A, step_B = LAB_Color_Tint
    # Generate 13 values from -6*step to +6*step, including 0
    def make_range(step):
        if step == 0:
            return [0]
        return [round(i * step, 3) for i in range(-10, 11)]

    L_values = make_range(step_L)
    A_values = make_range(step_A)
    B_values = make_range(step_B)

    # Progress Updating
    num_morphs = 10
    total_masks = len(find_source_images(inputFolder)) * len(L_values) * len(A_values) * len(B_values)
    # EXCEL OUTPUT Tracking
    mask_records = []
    with tqdm(total=total_masks, desc="Generating tinted masks") as pbar:
        for gender in Genders:
            for ethnicity in Ethnicity:
                for age_range in Age_Ranges:
                    subfolder = os.path.join(inputFolder, gender, ethnicity, age_range)
                    if not os.path.exists(subfolder):
                        continue
                    for i in range(1, 11):  # M1.png to M10.png
                        filename = f"M{i}.png"
                        filepath = os.path.join(subfolder, filename)
                        if not os.path.exists(filepath):
                            tqdm.write(f"Missing file: {filepath}")
                            continue
                        maskCount = 0
                        parsing_map, image_rgb = get_parsing_map(filepath)
                        for L, A, B in product(L_values, A_values, B_values):
                            #print(f"Creating Tint: ({maskCount} / 12): {(L, A, B)}")
                            tinted = colorTint(parsing_map, image_rgb, (L, A, B))
                            if tinted is None:
                                pbar.update(1)
                                continue

                            converted = cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR)
                            output_path = os.path.join(outputFolder, gender, ethnicity, age_range, f"M{i}")
                            os.makedirs(output_path, exist_ok=True)
                            filename_out = f"Mask{maskCount}.png"
                            cv2.imwrite(os.path.join(output_path, filename_out), converted)

                            # RECORD MASK IN EXCEL
                            mask_id = f"{gender}_{ethnicity}_{age_range}_M{i}_Mask{maskCount}"
                            mask_records.append({"mask_id": mask_id, "L*": L, "A*": A, "B*": B})
                            maskCount += 1
                            pbar.update(1)
    df = pd.DataFrame(mask_records)
    df.to_excel(os.path.join(outputFolder, "mask_lab_values.xlsx"), index=False)
    print("Excel file 'mask_lab_values.xlsx' created.")
    return




parseMorphs("example_dir","output/Exp1 (1.5 change)", (1.5, 0, 0))
parseMorphs("example_dir","output/Exp1 (2.0 change)", (2, 0, 0))
parseMorphs("example_dir", "output/Exp2", (0, 0, 1.23))
parseMorphs("example_dir", "output/Exp3", (1.5, 0, 1.23))
# Show results
# cv2.imshow('Tint Level: 5', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 5)), cv2.COLOR_RGB2BGR))
# cv2.imshow('Tint Level: 10', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 10)), cv2.COLOR_RGB2BGR))
# cv2.imshow('Tint Level: 15', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 15)), cv2.COLOR_RGB2BGR))
# cv2.imshow('Tint Level: 20', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 20)), cv2.COLOR_RGB2BGR))
# cv2.imshow('Original', cv2.imread("merge_male1.jpg", cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()