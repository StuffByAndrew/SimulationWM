"""Promptable image segmentation with MobileSAM (Apache-2.0, ViT-Tiny image encoder) on CPU, with local read-only weights.

CLI (prints a JSON summary; writes OUT_DIR/masks.npz with boolean masks[N,H,W] and scores[N], and OUT_DIR/mask_NNN.png):
  python3 segment.py IMAGE OUT_DIR --point X,Y[,LABEL] [--point ...] [--box X0,Y0,X1,Y1] [--multimask]
  python3 segment.py IMAGE OUT_DIR --auto [--points-per-side 16]
Pixels are measured from the top left. LABEL 1 marks foreground (default) and 0 background. A prompted call returns the
best mask, or three candidates with --multimask. --auto segments everything from a grid of point prompts, largest first.
Library: sys.path.insert(0, "/opt/public/tools/segmentation"); from segment import Segmenter
  s = Segmenter(); s.set_image(rgb_uint8_HxWx3); masks, scores = s.predict(points=[(x, y, 1)], box=None); s.auto(rgb)
Loading the model takes a few seconds; one Segmenter can serve many prompts on the same image.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import time
import warnings

import numpy as np
from PIL import Image

VENDOR = Path(__file__).resolve().parent / "vendor"


class Segmenter:
    def __init__(self):
        import torch
        torch.set_num_threads(len(os.sched_getaffinity(0)))
        sys.path.insert(0, str(VENDOR))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            from mobile_sam import SamAutomaticMaskGenerator, SamPredictor, sam_model_registry
            self.model = sam_model_registry["vit_t"]()
            self.model.load_state_dict(torch.load(VENDOR / "mobile_sam.pt", map_location="cpu", weights_only=True))
        self.model.eval()
        self.torch, self.predictor, self.generator = torch, SamPredictor(self.model), SamAutomaticMaskGenerator

    def set_image(self, rgb):
        rgb = np.ascontiguousarray(rgb)
        assert rgb.dtype == np.uint8 and rgb.ndim == 3 and rgb.shape[2] == 3, "image must be uint8 RGB HxWx3"
        with self.torch.inference_mode():
            self.predictor.set_image(rgb)

    def predict(self, points=(), box=None, multimask=False):
        """Masks [N,H,W] bool and scores [N] for points [(x, y, label)] and/or box (x0, y0, x1, y1) on the current image."""
        assert len(points) or box is not None, "give at least one point or a box"
        points = np.asarray(points, float).reshape(-1, 3)
        with self.torch.inference_mode():
            masks, scores, _ = self.predictor.predict(point_coords=points[:, :2] if len(points) else None,
                                                      point_labels=points[:, 2] if len(points) else None,
                                                      box=None if box is None else np.asarray(box, float), multimask_output=multimask)
        order = np.argsort(-scores)
        return masks[order].astype(bool), scores[order]

    def auto(self, rgb, points_per_side=16):
        """All masks from a grid of prompts, largest first, as dicts with segmentation, area, bbox (XYWH), predicted_iou."""
        with self.torch.inference_mode():
            found = self.generator(self.model, points_per_side=points_per_side).generate(np.ascontiguousarray(rgb))
        return sorted(found, key=lambda item: -item["area"])


def _numbers(text, count):
    values = [float(value) for value in text.split(",")]
    assert len(values) in count, "expected %s comma-separated numbers, got %r" % (" or ".join(map(str, count)), text)
    return values


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0], epilog=__doc__.split("\n", 1)[1], formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("image")
    parser.add_argument("out_dir")
    parser.add_argument("--point", action="append", default=[], help="X,Y[,LABEL]")
    parser.add_argument("--box", help="X0,Y0,X1,Y1")
    parser.add_argument("--multimask", action="store_true")
    parser.add_argument("--auto", action="store_true")
    parser.add_argument("--points-per-side", type=int, default=16)
    args = parser.parse_args(argv)
    assert args.auto != bool(args.point or args.box), "use either --auto or point/box prompts"
    rgb = np.asarray(Image.open(args.image).convert("RGB"))
    clock, cpu = time.perf_counter(), time.process_time()
    segmenter = Segmenter()
    timing = {"load_seconds": time.perf_counter() - clock}
    if args.auto:
        found = segmenter.auto(rgb, args.points_per_side)
        masks = np.array([item["segmentation"] for item in found], bool).reshape(-1, *rgb.shape[:2])
        scores = np.array([item["predicted_iou"] for item in found], float)
    else:
        segmenter.set_image(rgb)
        timing["embed_seconds"] = time.perf_counter() - clock - timing["load_seconds"]
        points = [_numbers(text, (2, 3)) + [1.0] * (len(text.split(",")) == 2) for text in args.point]
        masks, scores = segmenter.predict(points, _numbers(args.box, (4,)) if args.box else None, args.multimask)
    timing.update(total_seconds=time.perf_counter() - clock, cpu_seconds=time.process_time() - cpu)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "masks.npz", masks=masks, scores=scores)
    summary = []
    for index, (mask, score) in enumerate(zip(masks, scores)):
        Image.fromarray(mask.astype(np.uint8) * 255).save(out / ("mask_%03d.png" % index))
        y, x = np.nonzero(mask)
        summary.append({"index": index, "file": str(out / ("mask_%03d.png" % index)), "score": round(float(score), 4), "area_px": int(mask.sum()),
                        "bbox_xyxy": [int(x.min()), int(y.min()), int(x.max()), int(y.max())] if len(x) else None})
    print(json.dumps({"image": args.image, "size": [rgb.shape[1], rgb.shape[0]], "masks": summary, "npz": str(out / "masks.npz"),
                      "timing": {key: round(value, 3) for key, value in timing.items()}}))


if __name__ == "__main__":
    main()
