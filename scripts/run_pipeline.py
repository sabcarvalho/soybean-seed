"""Entry point — reads a config JSON and runs the pipeline."""
import argparse
import json, sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))
from seed_segmentation.config import PipelineConfig, FileTemplates
from seed_segmentation.io_utils import discover_ids, read_seed_entry
from seed_segmentation.pipeline import SegmentationPipeline


def main(config_path: str):
    with open(config_path) as f:
        cfg_json = json.load(f)

    cfg = PipelineConfig(**cfg_json.get("pipeline", {}))
    templates = FileTemplates(**cfg_json.get("file_templates", {}))
    data_dir = cfg_json["data_dir"]
    output_dir = cfg_json["output_dir"]

    disc = cfg_json.get("discovery", {})
    if disc.get("enabled", True):
        labeled_ids, unlabeled_ids = discover_ids(
            data_dir, templates,
            require=tuple(disc.get("require", ("rgb", "rx", "bil", "hdr"))),
        )
    else:
        labeled_ids = cfg_json.get("labeled_ids", [])
        unlabeled_ids = cfg_json.get("unlabeled_ids", [])

    labeled_set = set(labeled_ids)
    all_ids = [(i, True) for i in labeled_ids] + \
              [(i, False) for i in unlabeled_ids if i not in labeled_set]

    for image_id, is_labeled in all_ids:
        #try:
            data = read_seed_entry(
                image_id, data_dir, templates,
                n_bands=cfg.n_hyperspectral_bands, load_json=is_labeled)
            pipe = SegmentationPipeline(data, cfg)
            pipe.run()
            pipe.save_cropped_seeds(output_dir, f"bandeja_{image_id}")
        # except Exception as e:
        #     print(f"[ERRO] {image_id}: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json")
    args = parser.parse_args()
    main(args.config)