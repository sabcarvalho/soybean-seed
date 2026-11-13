"""I/O helpers for reading seeds and annotations."""
import json, os, cv2, re, glob
import numpy as np
from spectral import envi

def _template_to_regex(template: str) -> re.Pattern:
    """Converts 'RIP2_{id}-RGB.tif' into a regex with a named group 'id'."""
    escaped = re.escape(template)
    pattern = escaped.replace(r"\{id\}", r"(?P<id>.+?)")
    return re.compile(f"^{pattern}$")


def discover_ids(data_dir, templates, require=("rgb", "rx", "bil", "hdr"),
                 verbose=True):
    """Scans `data_dir` and returns (annotated_ids, unannotated_ids). 

    An ID is discovered based on the RGB file (using the `templates.rgb` template). 
    An ID is classified as:
    - **annotated** if the corresponding JSON file exists; 
    - **ignored** if any modality listed in `require` is missing. 

    Returns sorted lists (using numerical/lexicographical sorting).
    """
    rgb_re = _template_to_regex(templates.rgb)
    labeled, unlabeled, skipped = [], [], []

    for path in sorted(glob.glob(os.path.join(data_dir, "*"))):
        fname = os.path.basename(path)
        m = rgb_re.match(fname)
        if not m:
            continue
        image_id = m.group("id")

        missing = []
        for key in require:
            tpl = getattr(templates, key)
            if not os.path.exists(os.path.join(data_dir, tpl.format(id=image_id))):
                missing.append(key)

        if missing:
            skipped.append((image_id, missing))
            continue

        json_path = os.path.join(data_dir, templates.json.format(id=image_id))
        if os.path.exists(json_path):
            labeled.append(image_id)
        else:
            unlabeled.append(image_id)

    if verbose:
        print(f"[discover_ids] pasta: {data_dir}")
        print(f"  Anotadas    : {len(labeled)}")
        print(f"  Não anot.   : {len(unlabeled)}")
        if skipped:
            print(f"  Ignoradas   : {len(skipped)} (arquivos faltando)")
            for sid, miss in skipped:
                print(f"    - {sid}: faltando {miss}")

    return sorted(labeled), sorted(unlabeled)

def anote_points(data):
    """Extract [point, label] from a labelme-style JSON."""
    return [[shape["points"][0], shape["label"]] for shape in data["shapes"]]


def read_seed_entry(image_id, data_dir, templates, n_bands=150,
                    load_json=True):
    """Load all modalities for a given image id."""
    rgb = cv2.imread(os.path.join(data_dir, templates.rgb.format(id=image_id)))
    rx = cv2.imread(os.path.join(data_dir, templates.rx.format(id=image_id)))

    hdr = os.path.join(data_dir, templates.hdr.format(id=image_id))
    bil = os.path.join(data_dir, templates.bil.format(id=image_id))
    cube = envi.open(hdr, bil)
    hiper = np.array([cube.read_band(i) for i in range(n_bands)])

    json_data = None
    if load_json:
        with open(os.path.join(data_dir, templates.json.format(id=image_id))) as f:
            json_data = json.load(f)

    return {"rgb": rgb, "rx": rx, "hiper": hiper, "json": json_data}