import os
import torch
import pandas as pd
import numpy as np

os.makedirs("Camelyon16/pt_files", exist_ok=True)
os.makedirs("dataset_csv/camelyon16", exist_ok=True)

# create 6 toy slides
slides = [
    ("slide_001", 0),
    ("slide_002", 0),
    ("slide_003", 1),
    ("slide_004", 1),
    ("slide_005", 0),
    ("slide_006", 1),
]

# each slide = variable number of patches, each patch = 1024-dim feature
rng = np.random.default_rng(2026)
for slide_id, label in slides:
    n_patches = int(rng.integers(200, 500))
    features = torch.randn(n_patches, 1024)
    torch.save(features, f"Camelyon16/pt_files/{slide_id}.pt")

# repo expects columns: train, train_label, val, val_label, test, test_label
df = pd.DataFrame({
    "train":      ["slide_001", "slide_002", "slide_003", "slide_004"],
    "train_label":[0,           0,           1,           1          ],
    "val":        ["slide_005", None,        None,        None       ],
    "val_label":  [0,           None,        None,        None       ],
    "test":       ["slide_006", None,        None,        None       ],
    "test_label": [1,           None,        None,        None       ],
})

df.to_csv("dataset_csv/camelyon16/fold0.csv")
print("Toy data created.")
