# ---------------------------------------------------------------------------
# imports you already listed – nothing new at the top level
# ---------------------------------------------------------------------------
from facial_tinting import get_parsing_map, colorTint   # already present
from itertools import product
from pathlib import Path
from tqdm.auto import tqdm
import pandas as pd
import cv2, os, concurrent.futures as cf


# ---------------------------------------------------------------------------
# helper: discover every existing portrait  …/Mx.png
# ---------------------------------------------------------------------------
GENDERS     = ["Male", "Female"]
ETHNICITIES = ["White", "Black", "Asian"]
AGE_RANGES  = ["18-24", "25-34", "35-44", "45-54", "55-64"]

def find_source_images(root):
    root = Path(root)
    pattern = [f"M{i}.png" for i in range(1, 11)]
    files = []
    for g in GENDERS:
        for e in ETHNICITIES:
            for a in AGE_RANGES:
                folder = root / g / e / a
                for p in pattern:
                    f = folder / p
                    if f.exists():
                        files.append(f)
    return files


# ---------------------------------------------------------------------------
# worker: one portrait  ⇒  all tints  (runs inside a worker process)
# ---------------------------------------------------------------------------
def _process_portrait(args):
    """
    args = (image_path, L_vals, A_vals, B_vals, out_dir)
    returns list[dict]  (rows for Excel)
    """
    image_path, L_vals, A_vals, B_vals, out_dir = args
    image_path = Path(image_path)
    gender, ethnicity, age_range = image_path.parts[-4:-1]
    model_name = image_path.stem                         # "M1", "M2", …

    # ---- heavy part done once for this portrait --------------------------------
    parsing_map, image_rgb = get_parsing_map(str(image_path))

    rows = []
    for idx, (dL, dA, dB) in enumerate(product(L_vals, A_vals, B_vals)):
        tinted = colorTint(parsing_map, image_rgb, (dL, dA, dB))

        # save PNG
        dst_dir = Path(out_dir) / gender / ethnicity / age_range / model_name
        dst_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(dst_dir / f"Mask{idx}.png"),
                    cv2.cvtColor(tinted, cv2.COLOR_RGB2BGR))

        rows.append({"mask_id": f"{gender}_{ethnicity}_{age_range}_"
                                f"{model_name}_Mask{idx}",
                     "L*": dL, "A*": dA, "B*": dB})
    return rows


# ---------------------------------------------------------------------------
#  PUBLIC FUNCTION  –  name kept unchanged
# ---------------------------------------------------------------------------
def parseMorphs(inputFolder, outputFolder, LAB_Color_Tint, max_workers=None):
    """
    Create all tints in parallel.

    inputFolder     root of the morph tree
    outputFolder    where to write tinted/ and the Excel sheet
    LAB_Color_Tint  (step_L, step_A, step_B)  – as before
    max_workers     #processes; None ⇒ os.cpu_count()
    """
    step_L, step_A, step_B = LAB_Color_Tint

    # build the lists of actual tint values
    def make_range(step):
        return [0] if step == 0 else [round(i*step, 3) for i in range(-10, 11)]
    L_vals, A_vals, B_vals = map(make_range, (step_L, step_A, step_B))

    portraits = find_source_images(inputFolder)
    if not portraits:
        print("No source images found.")
        return

    # total masks (for a correct progress bar)
    total_masks = len(portraits) * len(L_vals) * len(A_vals) * len(B_vals)
    os.makedirs(outputFolder, exist_ok=True)

    excel_rows = []

    with tqdm(total=total_masks, desc="Generating tinted masks") as bar:
        with cf.ProcessPoolExecutor(max_workers=max_workers) as pool:
            args_iter = ((p, L_vals, A_vals, B_vals, outputFolder)
                         for p in portraits)

            for rows in pool.map(_process_portrait, args_iter):
                excel_rows.extend(rows)
                bar.update(len(rows))

    # write combined Excel sheet
    df = pd.DataFrame(excel_rows)
    df.to_excel(os.path.join(outputFolder, "mask_lab_values.xlsx"), index=False)
    print("Excel file 'mask_lab_values.xlsx' created.")



# Show results
if __name__ == "__main__":
    parseMorphs("example_dir", "example_out/Exp1 (1.5 Step)", LAB_Color_Tint=(1.5, 0, 0), max_workers=None) #2.36 Change instead
    parseMorphs("example_dir", "example_out/Exp1 (2.0 Step)", LAB_Color_Tint=(2.0, 0, 0), max_workers=None)
    parseMorphs("example_dir", "example_out/Exp2", LAB_Color_Tint=(0, 0, 1.23), max_workers=None)
    parseMorphs("example_dir", "example_out/Exp3", LAB_Color_Tint=(1.5, 0, 1.23), max_workers=None)
cv2.waitKey(0)
cv2.destroyAllWindows()