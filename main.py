from facial_tinting import *
import cv2
import os
import pandas as pd
from itertools import product

# Gender -> Race -> Age Range -> M1 .. M12
Genders = ["Male", "Female"]
Ethnicity = ["White", "Black", "Asian"]
Age_Ranges = ["18-24", "25-34", "35-44", "45-54", "55-64"]
def parseMorphs(rootFolder, LAB_Color_Tint):
    step_L, step_A, step_B = LAB_Color_Tint

    # Generate 13 values from -6*step to +6*step, including 0
    def make_range(step):
        if step == 0:
            return [0]
        return [round(i * step, 3) for i in range(-6, 7)]

    L_values = make_range(step_L)
    A_values = make_range(step_A)
    B_values = make_range(step_B)

    # EXCEL OUTPUT Tracking
    mask_records = []
    for gender in Genders:
        for ethnicity in Ethnicity:
            for age_range in Age_Ranges:
                subfolder = os.path.join(rootFolder, gender, ethnicity, age_range)
                if not os.path.exists(subfolder):
                    continue
                for i in range(1, 11):  # M1.png to M10.png
                    filename = f"M{i}.png"
                    filepath = os.path.join(subfolder, filename)
                    if not os.path.exists(filepath):
                        print(f"Missing file: {filepath}")
                        continue
                    print(f"Processing file: ({gender}, {ethnicity}, {age_range}): {filename}")
                    maskCount = 0
                    for L, A, B in product(L_values, A_values, B_values):
                        print(f"Creating Tint: ({maskCount} / 12): {(L, A, B)}")
                        tinted = colorTint(filepath, (L, A, B))
                        if tinted is None:
                            continue

                        converted = cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR)
                        output_path = os.path.join("tinted", gender, ethnicity, age_range, f"M{i}")
                        os.makedirs(output_path, exist_ok=True)
                        filename_out = f"Mask{maskCount}.png"
                        cv2.imwrite(os.path.join(output_path, filename_out), converted)

                        # RECORD MASK IN EXCEL
                        mask_id = f"{gender}_{ethnicity}_{age_range}_M{i}_Mask{maskCount}"
                        mask_records.append({"mask_id": mask_id, "L*": L, "A*": A, "B*": B})
                        maskCount += 1
    df = pd.DataFrame(mask_records)
    df.to_excel(os.path.join("tinted", "mask_lab_values.xlsx"), index=False)
    print("Excel file 'mask_lab_values.xlsx' created.")
    return








parseMorphs("example_dir", (1.23, 0, 1.23))
# Show results
# cv2.imshow('Tint Level: 5', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 5)), cv2.COLOR_RGB2BGR))
# cv2.imshow('Tint Level: 10', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 10)), cv2.COLOR_RGB2BGR))
# cv2.imshow('Tint Level: 15', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 15)), cv2.COLOR_RGB2BGR))
# cv2.imshow('Tint Level: 20', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 20)), cv2.COLOR_RGB2BGR))
# cv2.imshow('Original', cv2.imread("merge_male1.jpg", cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()