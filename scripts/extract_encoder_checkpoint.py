"""Extract the shared PLM encoder from a trained PLM-ICD checkpoint."""

from pathlib import Path
import sys

import torch


if len(sys.argv) != 3:
    raise SystemExit("usage: extract_encoder_checkpoint.py MODEL_PT OUT_PT")

source_path, output_path = map(Path, sys.argv[1:])
payload = torch.load(source_path, map_location="cpu", weights_only=False)
state = payload.get("model", payload)
encoder_state = {
    key: value.cpu().clone()
    for key, value in state.items()
    if key.startswith("roberta_encoder.")
}
if not encoder_state:
    raise ValueError(f"No roberta_encoder.* tensors found in {source_path}")
output_path.parent.mkdir(parents=True, exist_ok=True)
torch.save(
    {
        "model": encoder_state,
        "encoder_keys": sorted(encoder_state),
        "source_checkpoint": str(source_path),
        "num_classes": payload.get("num_classes"),
    },
    output_path,
)
print(f"wrote {output_path} with {len(encoder_state)} encoder tensors")
