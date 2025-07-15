import os
import sys
import cv2   
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torchvision.ops import nms
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
    os.makedirs(crop_folder, exist_ok=True)

    box_list = boxes[0]    
    score_list = scores[0] 
    label_list = labels[0] 

    crop_idx = 0
    for idx_box, (box, scr, lbl) in enumerate(
        zip(box_list, score_list, label_list)
    ):
        if scr.item() < thrh:
            continue

        x1, y1, x2, y2 = box.tolist()
        left, top, right, bottom = int(x1), int(y1), int(x2), int(y2)

        crop_im = im_pil.crop((left, top, right, bottom))

        crop_filename = f"{base_name}_crop{crop_idx}.jpg"
        out_path = os.path.join(crop_folder, crop_filename)

        crop_im.save(out_path)
        print(
            f"Saved crop {crop_idx} "
            f"(label={lbl.item()}, score={scr.item():.2f}) to {out_path}"
        )

        crop_idx += 1

    if crop_idx == 0:
        print("(No boxes above threshold; no crops were saved.)")


def draw(
    images,
    labels,
    boxes,
    scores,
    thrh: float = 0.7,
    base_name: str = "image",
):

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

        output_filename = f"{base_name}_boxes.jpg"
        im.save(output_filename)
        print(f"Saved full-image with boxes to {output_filename}")


def process_image(
    model,
    device,
    file_path: str,
    crop_folder_root: str = "./output/crops",
    thrh: float = 0.7,
):

    im_pil = Image.open(file_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to(device)

    transforms = T.Compose(
        [
            T.Resize((640, 640)),
            T.ToTensor(),
        ]
    )
    im_data = transforms(im_pil).unsqueeze(0).to(device)

    output = model(im_data, orig_size)
    labels, boxes, scores = output

    iou_thresh = 0.5

    box_list = boxes[0]    
    score_list = scores[0] 
    label_list = labels[0] 

    keep_idx = nms(box_list, score_list, iou_thresh)

    boxes = box_list[keep_idx].unsqueeze(0)   
    scores = score_list[keep_idx].unsqueeze(0) 
    labels = label_list[keep_idx].unsqueeze(0) 

    base_name = os.path.splitext(os.path.basename(file_path))[0]
    this_crop_folder = os.path.join(crop_folder_root, base_name)

    save_crops(
        im_pil=im_pil,
        labels=labels,
        boxes=boxes,
        scores=scores,
        thrh=thrh,
        crop_folder=this_crop_folder,
        base_name=base_name,
    )
    print("Image processing complete.")


def process_video(model, device, file_path):
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

        frame_pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        w, h = frame_pil.size
        orig_size = torch.tensor([[w, h]]).to(device)
        im_data = transforms(frame_pil).unsqueeze(0).to(device)

        output = model(im_data, orig_size)
        labels, boxes, scores = output

        draw([frame_pil], labels, boxes, scores, thrh=0.7, base_name=f"frame_{frame_count}")

        frame = cv2.cvtColor(np.array(frame_pil), cv2.COLOR_RGB2BGR)
        out.write(frame)

        frame_count += 1
        if frame_count % 10 == 0:
            print(f"Processed {frame_count} frames…")

    cap.release()
    out.release()
    print("Video processing complete. Result saved as 'torch_results.mp4'.")


def main(args):

    cfg = YAMLConfig(args.config, resume=args.resume)

    if "HGNetv2" in cfg.yaml_cfg:
        cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        if "ema" in checkpoint:
            state = checkpoint["ema"]["module"]
        else:
            state = checkpoint["model"]
    else:
        raise AttributeError("Only support resume-mode for now")

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

    file_path = args.input
    ext = os.path.splitext(file_path)[-1].lower()

    if ext in [".jpg", ".jpeg", ".png", ".bmp"]:
        crop_root = args.output_dir or "./outputs/crops"
        process_image(
            model=model,
            device=device,
            file_path=file_path,
            crop_folder_root=crop_root,
            thrh=0.6,
        )
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
    print(
        ">>> torch_inf.py received output_dir:", args.output_dir,
        file=sys.stderr
    )
    main(args)
