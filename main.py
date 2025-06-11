import cv2
import numpy as np
import torch
from FaceParsing.model import BiSeNet  # your BiSeNet implementation file


## Resources
# Dataset for facial parsing: https://github.com/switchablenorms/CelebAMask-HQ/
# Pretrained model: https://github.com/zllrunning/face-parsing.PyTorch
# Load model and weights (adjust path as needed)
net = BiSeNet(n_classes=19)  # 19 classes for CelebAMask-HQ
net.load_state_dict(torch.load('FaceParsing/79999_iter.pth', map_location='cpu')) # Load Pre-trained model
net.eval()


def get_parsing_map(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    h, w, _ = image.shape

    # Preprocess input for BiSeNet (resize + normalize)
    input_image = cv2.resize(image, (512, 512))
    input_image = input_image.astype(np.float32) / 255.0
    input_image = (input_image - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
    input_image = input_image.transpose(2, 0, 1)  # HWC to CHW
    input_tensor = torch.from_numpy(input_image).unsqueeze(0)  # Add batch dim
    input_tensor = input_tensor.to(next(net.parameters()).dtype)
    # Forward pass
    with torch.no_grad():
        out = net(input_tensor)[0]  # BiSeNet returns a tuple, use first
        parsing = out.squeeze(0).cpu().numpy().argmax(0)

    # Resize parsing back to original image size
    parsing = cv2.resize(parsing.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    return parsing, image

def colorTint(image_path, LAB_Color_Tint):
    Tint_L, Tint_A, Tint_B = LAB_Color_Tint
    parsing_map, image_rgb = get_parsing_map(image_path)
    # Mapping is not the same as detailed in https://github.com/switchablenorms/CelebAMask-HQ/tree/master/face_parsing
    #         atts = ['skin', 'l_brow', 'r_brow', 'l_eye', 'r_eye', 'eye_g', 'l_ear', 'r_ear', 'ear_r',
    #                 'nose', 'mouth', 'u_lip', 'l_lip', 'neck', 'neck_l', 'cloth', 'hair', 'hat']
    skin_labels = [1, 7, 8, 10, 14]  # Include Skin, Ears, Nose, and Neck
    mask = np.zeros_like(parsing_map, dtype=np.uint8)
    for label in skin_labels:
        mask[parsing_map == label] = 255
    # Convert original RGB image to BGR for Open CV processing
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    # Convert image to LAB color space
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    # Split LAB channels
    L, A, B = cv2.split(lab)
    # Create a boolean mask (True where skin/ears/nose/neck)
    mask_bool = mask.astype(bool)
    # Shift the individual channels by input Tint
    L[mask_bool] = np.clip(L[mask_bool] + Tint_L, 0, 255)
    A[mask_bool] = np.clip(A[mask_bool] + Tint_A, 0, 255)
    B[mask_bool] = np.clip(B[mask_bool] + Tint_B, 0, 255)
    # Merge back into color vector
    lab_tinted = cv2.merge([L, A, B])

    # Convert back to BGR
    bgr_tinted = cv2.cvtColor(lab_tinted, cv2.COLOR_LAB2BGR)

    # Convert back to RGB for display or further processing if you want
    rgb_tinted = cv2.cvtColor(bgr_tinted, cv2.COLOR_BGR2RGB)
    return rgb_tinted













# Show results
cv2.imshow('Tint Level: 5', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 5)), cv2.COLOR_RGB2BGR))
cv2.imshow('Tint Level: 10', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 10)), cv2.COLOR_RGB2BGR))
cv2.imshow('Tint Level: 15', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 15)), cv2.COLOR_RGB2BGR))
cv2.imshow('Tint Level: 20', cv2.cvtColor(colorTint("merge_male1.jpg", (0, 0, 20)), cv2.COLOR_RGB2BGR))
cv2.imshow('Original', cv2.imread("merge_male1.jpg", cv2.COLOR_BGR2RGB))
cv2.waitKey(0)
cv2.destroyAllWindows()