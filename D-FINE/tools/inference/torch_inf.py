#!/usr/bin/env python3
# run_inference.py

import os
import sys
import cv2    # for video, if you still need that part
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image, ImageDraw

sys.path.append(
    os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../")
    )
)
from src.core import YAMLConfig


def save_crops(
    im_pil: Image.Image,
    labels: torch.Tensor,
    boxes: torch.Tensor,
    scores: torch.Tensor,
    thrh: float,
    crop_folder: str,
    base_name: str,
):
    """
    Given a single PIL image (im_pil), plus its labels/boxes/scores (all 1×N tensors),
    this will crop out each box where score > thrh, and save it under crop_folder/
    with filenames like <base_name>_crop0.jpg, <base_name>_crop1.jpg, etc.
    """

    # 1) Make sure the output folder exists
    os.makedirs(crop_folder, exist_ok=True)

    # boxes: (1, num_boxes, 4), scores: (1, num_boxes), labels: (1, num_boxes)
    box_list = boxes[0]    # shape: (num_boxes, 4)
    score_list = scores[0] # shape: (num_boxes,)
    label_list = labels[0] # shape: (num_boxes,)

    crop_idx = 0
    for idx_box, (box, scr, lbl) in enumerate(
        zip(box_list, score_list, label_list)
    ):
        if scr.item() < thrh:
            continue

        # Convert box coords to ints
        x1, y1, x2, y2 = box.tolist()
        left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)

        # Crop from the original PIL image
        crop_im = im_pil.crop((left, top, right, bottom))

        # Save as "<base_name>_crop<idx>.jpg"
        crop_filename = f"{base_name}_crop{crop_idx}.jpg"
        out_path = os.path.join(crop_folder, crop_filename)

        crop_im.save(out_path)
        print(
            f"   → Saved crop {crop_idx} "
            f"(label={lbl.item()}, score={scr.item():.2f}) to {out_path}"
        )

        crop_idx += 1

    if crop_idx == 0:
        print("   (No boxes above threshold; no crops were saved.)")


def draw(
    images,
    labels,
    boxes,
    scores,
    thrh: float = 0.7,
    base_name: str = "image",
):
    """
    Draws all boxes with score>thrh onto the single‐image list,
    then saves the result as "<base_name>_boxes.jpg" in the current folder.
    """

    for i, im in enumerate(images):
        drawer = ImageDraw.Draw(im)

        scr = scores[i]
        lab = labels[i][scr > thrh]
        box = boxes[i][scr > thrh]
        scrs = scr[scr > thrh]

        for j, b in enumerate(box):
            drawer.rectangle(list(b), outline="red")
            drawer.text(
                (b[0], b[1]),
                text=f"{lab[j].item()} {round(scrs[j].item(), 2)}",
                fill="blue",
            )

        # Save the “full image with boxes” as "<base_name>_boxes.jpg"
        output_filename = f"{base_name}_boxes.jpg"
        im.save(output_filename)
        print(f"   → Saved full‐image with boxes to {output_filename}")


def process_image(
    model,
    device,
    file_path: str,
    crop_folder_root: str = "./output/crops",
    thrh: float = 0.7,
):
    """
    1) Loads the image,
    2) Runs the model → (labels, boxes, scores),
    3) Draws + saves one image with all boxes (optional),
    4) Crops each box > thrh and saves under crop_folder_root/<basename>/.
    """

    # 1) Load image as PIL, get original size
    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    # 2) Pre‐process: Resize to 640×640, ToTensor
    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    # 3) Forward pass: model(images, orig_size) → (labels, boxes, scores)
    output = model(im_data, orig_size)
    labels, boxes, scores = output

    # # 4) Draw all high‐confidence boxes onto the original and save
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    # draw([im_pil], labels, boxes, scores, thrh=thrh, base_name=base_name)

    # 5) Create a per‐image crop folder: "./output/crops/<base_name>/"
    this_crop_folder = os.path.join(crop_folder_root, base_name)

    # 6) Save each crop where score > thrh
    save_crops(
        im_pil=im_pil,
        labels=labels,
        boxes=boxes,
        scores=scores,
        thrh=thrh,
        crop_folder=this_crop_folder,
        base_name=base_name,
    )


def process_video(model, device, file_path):
    """
    (Optional) If you still need video processing, you can leave this roughly as‐is.
    You’ll want to similarly call save_crops() on each frame if you want per‐frame crops.
    """

    cap = cv2.VideoCapture(file_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    orig_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter("torch_results.mp4", fourcc, fps, (orig_w, orig_h))

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )

    frame_count = 0
    print("Processing video frames…")
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to PIL
        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        # Forward
        output = model(im_data, orig_size)
        labels, boxes, scores = output

        # Draw on full frame and save one “frame_with_boxes.jpg” each time:
        # (you can tweak this to avoid overwriting every frame)
        draw([frame_pil], labels, boxes, scores, thrh=0.7, base_name=f"frame_{frame_count}")

        # If you want per‐frame crops, you could call save_crops(...) here:
        # save_crops(frame_pil, labels, boxes, scores, 0.7, "./output/video_crops", f"frame_{frame_count}")

        # Convert back to OpenCV and write to out video
        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        out.write(frame)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames…")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'torch_results.mp4'.")


def main(args):
    """Main entry point—loads D-FINE model, then calls process_image or process_video."""

    # 1) Load your YAML config + checkpoint exactly as before
    cfg = YAMLConfig(args.config, resume=args.resume)

    # If you want to disable HGNetv2 pretrained weights at inference:
    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume‐mode for now")

    cfg.model.load_state_dict(state)

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = cfg.model.deploy()
            self.postprocessor = cfg.postprocessor.deploy()

        def forward(self, images, orig_target_sizes):
            outputs = self.model(images)
            outputs = self.postprocessor(outputs, orig_target_sizes)
            return outputs

    device = args.device
    model = Model().to(device)

    # 2) Decide if image or video
    file_path = args.input
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        # Run our updated process_image (it now also saves individual crops)
        crop_root = args.output_dir or "./outputs/crops"
        process_image(
            model=model,
            device=device,
            file_path=file_path,
            crop_folder_root=crop_root,
            thrh=0.6,
        )
        print("Image processing complete.")
    else:
        process_video(model, device, file_path)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-c", "--config", type=str, required=True, help="Path to YAML config"
    )
    parser.add_argument(
        "-r", "--resume", type=str, required=True, help="Path to checkpoint .pth"
    )
    parser.add_argument(
        "-i", "--input", type=str, required=True, help="Path to input image/video"
    )
    parser.add_argument(
        "-d", "--device", type=str, default="cpu", help="CUDA device, e.g. cuda:0"
    )

    parser.add_argument(
        "-o", "--output-dir",\
        dest="output_dir",
        help="(optional) override the YAML's output_dir setting",
    )

    args = parser.parse_args()
    print(">>> torch_inf.py received output_dir:", args.output_dir, file=sys.stderr)
    main(args)
