from facial_tinting import *
import cv2
import os


# Gender -> Race -> Age Range -> M1 .. M12
Genders = ["Male", "Female"]
Ethnicity = ["White", "Black", "Asian"]
Age_Ranges = ["18-24", "25-34", "35-44", "45-54", "55-64"]
def parseMorphs(rootFolder):
    for gender in Genders:
        for ethnicity in Ethnicity:
            for age_range in Age_Ranges:
                subfolder = os.path.join(rootFolder, gender, ethnicity, age_range)
                if not os.path.exists(subfolder):
                    continue
                for i in range(1, 11):  # M1.jpg to M10.jpg
                    filename = f"M{i}.jpg"
                    filepath = os.path.join(subfolder, filename)
                    if not os.path.exists(filepath):
                        print(f"Missing file: {filepath}")
                        continue

                    tinted = colorTint(filepath, (0, 0, 5))
                    if tinted is None:
                        continue

                    converted = cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR)

                    # Create output path under "tinted/"
                    output_path = os.path.join("tinted", gender, ethnicity, age_range)
                    os.makedirs(output_path, exist_ok=True)
                    cv2.imwrite(os.path.join(output_path, filename), converted)

    return









# Show results
cv2.imshow('Tint Level: 5', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 5)), cv2.COLOR_RGB2BGR))
cv2.imshow('Tint Level: 10', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 10)), cv2.COLOR_RGB2BGR))
cv2.imshow('Tint Level: 15', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 15)), cv2.COLOR_RGB2BGR))
cv2.imshow('Tint Level: 20', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 20)), cv2.COLOR_RGB2BGR))
cv2.imshow('Original', cv2.imread("merge_male1.jpg", cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()