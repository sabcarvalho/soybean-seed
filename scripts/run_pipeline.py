"""Entry point — reads a config JSON and runs the pipeline."""
import argparse
import json, sys, os
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from segmentation.config import PipelineConfig, FileTemplates
from segmentation.io_utils import discover_ids, read_seed_entry
from segmentation.pipeline import SegmentationPipeline


def main(config_path: str):
    with open(config_path) as f:
        cfg_json = json.load(f)
    seg = cfg_json["segmentation"]
    cfg = PipelineConfig(**seg.get("pipeline", {}))
    templates = FileTemplates(**seg.get("file_templates", {}))
    data_dir = cfg_json["paths"]["raw_data"]
    output_dir = cfg_json["paths"]["output"]

    disc = seg.get("discovery", {})
    if disc.get("enabled", True):
        labeled_ids, unlabeled_ids = discover_ids(
            data_dir, templates,
            require=tuple(disc.get("require", ("rgb", "rx", "bil", "hdr"))),
        )
    else:
        labeled_ids = seg.get("labeled_ids", [])
        unlabeled_ids = seg.get("unlabeled_ids", [])

    labeled_set = set(labeled_ids)
    all_ids = [(i, True) for i in labeled_ids] + \
              [(i, False) for i in unlabeled_ids if i not in labeled_set]

    for image_id, is_labeled in all_ids:
        try:
            data = read_seed_entry(
                image_id, data_dir, templates,
                n_bands=cfg.n_hyperspectral_bands, load_json=is_labeled)
            pipe = SegmentationPipeline(data, cfg)
            pipe.run()
            if is_labeled: 
                dir = cfg_json["paths"]["cropped_labeled"]
            else:
                dir = cfg_json["paths"]["cropped_unlabeled"]
            os.makedirs(dir, exist_ok=True)
            pipe.save_cropped_seeds(dir, f"bandeja_{image_id}")
        except Exception as e:
            print(f"[ERRO] {image_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)