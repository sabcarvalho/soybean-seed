import glob
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

def create_paths(root, is_labeled=True):
    """Scan root/{RGB,RX,HIPER} and return (list of dicts, labels)."""
    rgb_paths = glob.glob(os.path.join(root, "RGB", "*.png"))
    path_data, labels = [], []
    unlabeled = 0

    for p_rgb in rgb_paths:
        file_name = os.path.basename(p_rgb)
        base_name = os.path.splitext(file_name)[0]

        if is_labeled:
            parts = base_name.split("_")
            label = parts[-1]
            base_name = "_".join(parts[:-1])
            p_hiper = os.path.join(root, "HIPER", base_name + ".npy")
            if len(label) == 0:
                unlabeled += 1
                continue
        else:
            label = None
            p_hiper = os.path.join(root, "HIPER", base_name[:-1] + ".npy")

        p_rx = os.path.join(root, "RX", file_name)

        if os.path.exists(p_rx) and os.path.exists(p_hiper):
            path_data.append({"rgb": p_rgb, "rx": p_rx, "hiper": p_hiper})
            if is_labeled:
                labels.append(label)
    if is_labeled:
        labels = LabelEncoder().fit_transform(np.array(labels))
        if unlabeled > 0:
            print(f"Sementes sem rotulo (descartadas): {unlabeled}")
    return path_data, labels