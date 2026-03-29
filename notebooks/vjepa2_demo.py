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

import src.datasets.utils.video.transforms as video_transforms
import src.datasets.utils.video.volume_transforms as volume_transforms
from src.models.attentive_pooler import AttentiveClassifier
from src.models.vision_transformer import vit_giant_xformers_rope

import cv2

# ---------------- CAMERA and ACTION CAPTURE PARAMETERS ----------------
CAM_ID = 0
FPS = 60
FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Motion detection
MOTION_THRESHOLD = 25
MOTION_AREA_RATIO = 0.001

# Temporal logic
MIN_MOTION_FRAMES = 5
NO_MOTION_FRAMES_TO_END = 20
MIN_NO_MOTION_BEFORE_START = 15   # IMPORTANT (prevents merging)

OUTPUT_FILE = "latest_motion.npy"

# ----------------------------------------------

IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

# HuggingFace model repo name
hf_model_name = (
    "facebook/vjepa2-vitg-fpc64-384"  # Replace with your favored model, e.g. facebook/vjepa2-vitg-fpc64-384
    )

# Path to local PyTorch weights
pt_model_path = "checkpoints/vitg-384.pt"

classifier_model_path = "checkpoints/ssv2-vitg-384-64x2x3.pt"

ssv2_classes_path = "ssv2_classes.json"


# -------------------------------------------

def detect_motion(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, MOTION_THRESHOLD, 255, cv2.THRESH_BINARY)

    motion_ratio = np.sum(thresh > 0) / thresh.size
    return motion_ratio > MOTION_AREA_RATIO


def play_video(frames, fps=20):
    delay = int(1000 / fps)
    for f in frames:
        cv2.imshow("Captured Motion", f)
        if cv2.waitKey(delay) & 0xFF == 27:
            break
    cv2.destroyWindow("Captured Motion")

# --------------------------------------------

def load_pretrained_vjepa_pt_weights(model, pretrained_weights):
    # Load weights of the VJEPA2 encoder
    # The PyTorch state_dict is already preprocessed to have the right key names
    pretrained_dict = torch.load(pretrained_weights, weights_only=True, map_location="cpu")["encoder"]
    pretrained_dict = {k.replace("module.", ""): v for k, v in pretrained_dict.items()}
    pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
    msg = model.load_state_dict(pretrained_dict, strict=False)
    print("Pretrained weights found at {} and loaded with msg: {}".format(pretrained_weights, msg))


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


def forward_vjepa_video(model_hf, model_pt, hf_transform, pt_transform, clip):
    # Run a sample inference with VJEPA
    with torch.inference_mode():
        # Read and pre-process the image
        #video = get_video()  # T x H x W x C
        video = sample_frames(clip, num_samples=64) 
        video = torch.from_numpy(video).permute(0, 3, 1, 2)  # T x C x H x W
        x_pt = pt_transform(video).cuda().unsqueeze(0)
        x_hf = hf_transform(video, return_tensors="pt")["pixel_values_videos"].to("cuda")
        # Extract the patch-wise features from the last layer
        out_patch_features_pt = model_pt(x_pt)
        out_patch_features_hf = model_hf.get_vision_features(x_hf)

    return out_patch_features_hf, out_patch_features_pt


def get_vjepa_video_classification_results(classifier, out_patch_features_pt):
    SOMETHING_SOMETHING_V2_CLASSES = json.load(open("ssv2_classes.json", "r"))

    with torch.inference_mode():
        out_classifier = classifier(out_patch_features_pt)

    #print(f"Classifier output shape: {out_classifier.shape}")

    print("Top 5 predicted class names:")
    top5_indices = out_classifier.topk(5).indices[0]
    top5_probs = F.softmax(out_classifier.topk(5).values[0]) * 100.0  # convert to percentage
    for idx, prob in zip(top5_indices, top5_probs):
        str_idx = str(idx.item())
        print(f"{SOMETHING_SOMETHING_V2_CLASSES[str_idx]} ({prob}%)")
    print("------------------------------------------------------------------------------")
    return


def run_sample_inference(model_hf, model_pt, hf_transform, pt_video_transform, clip):

    # Inference on video
    out_patch_features_hf, out_patch_features_pt = forward_vjepa_video(
        model_hf, model_pt, hf_transform, pt_video_transform, clip
    )
    
    # print(
    #     f"""
    #     Inference results on video:
    #     HuggingFace output shape: {out_patch_features_hf.shape}
    #     PyTorch output shape:     {out_patch_features_pt.shape}
    #     Absolute difference sum:  {torch.abs(out_patch_features_pt - out_patch_features_hf).sum():.6f}
    #     Close: {torch.allclose(out_patch_features_pt, out_patch_features_hf, atol=1e-3, rtol=1e-3)}
    #     """
    # )

    # Initialize the classifier
    classifier = (
        AttentiveClassifier(embed_dim=model_pt.embed_dim, num_heads=16, depth=4, num_classes=174).cuda().eval()
    )
    load_pretrained_vjepa_classifier_weights(classifier, classifier_model_path)

    get_vjepa_video_classification_results(classifier, out_patch_features_pt)

def main():


    # --------------------------------------------------

    # Initialize the HuggingFace model, load pretrained weights
    model_hf = AutoModel.from_pretrained(hf_model_name)
    model_hf.cuda().eval()

    # Build HuggingFace preprocessing transform
    hf_transform = AutoVideoProcessor.from_pretrained(hf_model_name)
    img_size = hf_transform.crop_size["height"]  # E.g. 384, 256, etc.

    # Initialize the PyTorch model, load pretrained weights
    model_pt = vit_giant_xformers_rope(img_size=(img_size, img_size), num_frames=64)
    model_pt.cuda().eval()
    load_pretrained_vjepa_pt_weights(model_pt, pt_model_path)

    # Build PyTorch preprocessing transform
    pt_video_transform = build_pt_video_transform(img_size=img_size)

    # ----------------------------------------------   

    cap = cv2.VideoCapture(CAM_ID)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, FPS)

    prev_gray = None

    # State variables
    in_motion = False
    ready_for_motion = False

    motion_buffer = []

    motion_count = 0
    no_motion_count = 0
    no_motion_gap = 0

    print(" Webcam started. Press 'q' to quit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if prev_gray is None:
            prev_gray = gray
            continue

        motion = detect_motion(prev_gray, gray)

        # ----------------------------------------
        # TRACK NO-MOTION GAP (for clean separation)
        # ----------------------------------------
        if not motion:
            no_motion_gap += 1
            no_motion_count += 1
        else:
            no_motion_gap = 0
            no_motion_count = 0

        # ----------------------------------------
        # ARM SYSTEM (only after sufficient no-motion)
        # ----------------------------------------
        if not in_motion and no_motion_gap >= MIN_NO_MOTION_BEFORE_START:
            ready_for_motion = True

        # ----------------------------------------
        # START MOTION (clean start, no old frames)
        # ----------------------------------------
        if motion and not in_motion and ready_for_motion:
            motion_count += 1

            if motion_count >= MIN_MOTION_FRAMES:
                print("Motion START (clean)")
                in_motion = True
                motion_buffer = []   # IMPORTANT: no previous frames
                ready_for_motion = False
                motion_count = 0

        # ----------------------------------------
        # RECORD FRAMES
        # ----------------------------------------
        if in_motion:
            motion_buffer.append(frame)

        # ----------------------------------------
        # END MOTION
        # ----------------------------------------
        if in_motion and no_motion_count >= NO_MOTION_FRAMES_TO_END:
            print("Motion END")

            if len(motion_buffer) > 0:
                clip = np.array(motion_buffer, dtype=np.uint8)
                print(f"Saved clip shape: {clip.shape}")

                np.save(OUTPUT_FILE, clip)

                cv2.waitKey(1000)  # Pause before playback

                # Playback
                play_video(clip, FPS)
                run_sample_inference(model_hf, model_pt, hf_transform, pt_video_transform, clip)

            # RESET STATE
            in_motion = False
            motion_buffer = []
            motion_count = 0
            no_motion_count = 0
            no_motion_gap = 0

        prev_gray = gray

        # ----------------------------------------
        # LIVE VIEW
        # ----------------------------------------
        cv2.imshow("Live Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    # Run with: `python -m notebooks.vjepa2_demo`
    main()
