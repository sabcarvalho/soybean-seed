"""Entry point — reads a config JSON and runs the pipeline."""
import argparse
import json
from pathlib import Path

from seed_segmentation.config import PipelineConfig, FileTemplates
from seed_segmentation.io_utils import read_seed_entry
from seed_segmentation.pipeline import SegmentationPipeline


def main(config_path: str):
    with open(config_path) as f:
        cfg_json = json.load(f)

    cfg = PipelineConfig(**cfg_json.get("pipeline", {}))
    templates = FileTemplates(**cfg_json.get("file_templates", {}))
    data_dir = cfg_json["data_dir"]
    output_dir = cfg_json["output_dir"]

    all_ids = list(cfg_json.get("labeled_ids", [])) + \
              list(cfg_json.get("unlabeled_ids", []))

    for image_id in all_ids:
        labeled = image_id in cfg_json.get("labeled_ids", [])
        try:
            data = read_seed_entry(
                image_id, data_dir, templates,
                n_bands=cfg.n_hyperspectral_bands, load_json=labeled)
            pipe = SegmentationPipeline(data, cfg)
            pipe.run()
            pipe.save_cropped_seeds(output_dir, f"bandeja_{image_id}")
        except Exception as e:
            print(f"[ERRO] {image_id}: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.json",
                        help="Caminho para o config JSON")
    args = parser.parse_args()
    main(args.config)