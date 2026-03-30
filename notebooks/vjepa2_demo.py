# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import subprocess
import time

import numpy as np
import torch
import torch.nn.functional as F
from decord import VideoReader
from transformers import AutoModel, AutoVideoProcessor

from datasets import load_dataset

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope



import cv2

# ---------------- CAMERA CONFIG AND OPTICAL FLOW BASED MOTION DETECTION PARAMETERS ----------------
CAM_ID = 0
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 60

# Motion detection thresholds
START_THRESHOLD = 0.7     # start motion
STOP_THRESHOLD = 0.5      # stop motion

MIN_MOTION_FRAMES = 10     # avoid noise
NO_MOTION_FRAMES = 10     # confirm motion end

SAVE_PATH = "motion.npy"

# ----------------------------------------------

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# HuggingFace model repo name
hf_model_name = (
    "facebook/vjepa2-vitl-fpc16-256-ssv2"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384
    )

classifier_model_path = "checkpoints/ssv2-vitl-16x2x3.pt" #ssv2-vitg-384-64x2x3.pt

ssv2_classes_path = "ssv2_classes.json"

# -------------------------------------------

def compute_flow_magnitude(prev_gray, gray):
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, gray, None,
        0.5, 3, 15, 3, 5, 1.2, 0
    )
    mag, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return np.mean(mag)


def load_pretrained_vjepa_classifier_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 classifier
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["classifiers"][0]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    #print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


def build_pt_video_transform(img_size):
    short_side_size = int(256.0 / 224 * img_size)
    # Eval transform has no random cropping nor flip
    eval_transform = video_transforms.Compose(
        [
            video_transforms.Resize(short_side_size, interpolation="bilinear"),
            video_transforms.CenterCrop(size=(img_size, img_size)),
            volume_transforms.ClipToTensor(),
            video_transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
        ]
    )
    return eval_transform

# sample 64 frames evenly from the input video, or repeat last frame if not enough frames
def sample_frames(frames, num_samples=64):
    T = len(frames)
    
    if T < num_samples:
        # repeat last frame if frames are less than num_samples
        indices = np.linspace(0, T - 1, num_samples).astype(int)
    else:
        indices = np.linspace(0, T - 1, num_samples).astype(int)
    
    return frames[indices]


def forward_vjepa_video(model_hf, hf_transform, clip):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the image
        video = sample_frames(clip, num_samples=16)
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W

        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    #return out_patch_features_hf, out_patch_features_pt
    return out_patch_features_hf


def get_vjepa_video_classification_results(classifier, out_patch_features):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features)

    #print(f"Classifier output shape: {out_classifier.shape}")

    print("Top 3 predicted class names:")
    top3_indices = out_classifier.topk(3).indices[0]
    top3_probs = F.softmax(out_classifier.topk(3).values[0]) * 100.0  # convert to percentage
    for idx, prob in zip(top3_indices, top3_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob}%)")
    print("------------------------------------------------------------------------------")
    return


def run_sample_inference(model_hf, hf_transform, classifier, clip):

    # Inference on video
    out_patch_features = forward_vjepa_video(
        model_hf, hf_transform, clip
    )

    print("Feature shape:", out_patch_features.shape)

    get_vjepa_video_classification_results(classifier, out_patch_features)


def main():

    # Initialize the HuggingFace model, load pretrained weights
    model_hf = AutoModel.from_pretrained(hf_model_name)
    model_hf.cuda().eval()

    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)
    img_size = hf_transform.crop_size["height"]  # E.g. 384, 256, etc.

    embed_dim_hf = model_hf.config.hidden_size


    # Initialize the classifier
    classifier = (
        AttentiveClassifier(embed_dim=embed_dim_hf, num_heads=16, depth=4, num_classes=174).cuda().eval()
    )
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    # ----------------------------------------------   

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    prev_gray = None

    # State variables
    in_motion = False
    motion_buffer = []

    motion_count = 0
    no_motion_count = 0

    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        motion_score = compute_flow_magnitude(prev_gray, gray)
        prev_gray = gray

        # -------- Motion Logic --------
        if not in_motion:
            if motion_score > START_THRESHOLD:
                motion_count += 1
                if motion_count >= MIN_MOTION_FRAMES:
                    print(">>> Motion START")
                    in_motion = True
                    motion_buffer = []  # clear old buffer
                    no_motion_count = 0
            else:
                motion_count = 0

        else:
            motion_buffer.append(frame)

            if motion_score < STOP_THRESHOLD:
                no_motion_count += 1
                if no_motion_count >= NO_MOTION_FRAMES:
                    print(">>> Motion END")

                    if len(motion_buffer) > 0:
                        video_np = np.array(motion_buffer, dtype=np.uint8)

                        # Save (overwrite previous)
                        np.save(SAVE_PATH, video_np)
                        print(f"Saved motion: {video_np.shape}")

                        start_time = time.perf_counter()
                        # run VJEPA inference on the captured motion
                        run_sample_inference(model_hf, hf_transform, classifier, video_np)

                        end_time = time.perf_counter()
                        elapsed_time = end_time - start_time
                        print(f"Inferencing took {elapsed_time:.4f} seconds")

                    # Reset state
                    in_motion = False
                    motion_buffer = []
                    motion_count = 0
                    no_motion_count = 0
            else:
                no_motion_count = 0

        # -------- Debug Display --------
        display = frame.copy()
        cv2.putText(display, f"Motion: {motion_score:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        state_text = "MOVING" if in_motion else "IDLE"
        cv2.putText(display, state_text,
                    (10, 70), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 0, 255) if in_motion else (255, 0, 0), 2)

        cv2.imshow("Live", display)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run with: `python -m notebooks.vjepa2_demo`
    main()
