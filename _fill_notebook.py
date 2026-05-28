"""Fill Part 4 cells of student_draft.ipynb (by cell id), leaving outputs intact."""
import json
from pathlib import Path

NB_PATH = Path(__file__).parent / "student_draft.ipynb"
R = {}

R["2b4123a2"] = '''@torch.no_grad()
def predict_bbox_in_crop(model, crop_rgb: np.ndarray, device: torch.device = DEVICE) -> np.ndarray:
    """Run the model on a single RGB crop, return normalized (x, y, w, h).

    Uses the SAME preprocessing as training (resize to 224, ImageNet normalize),
    so tracking-time inputs match training-time inputs exactly.
    """
    model.eval()
    if crop_rgb.shape[0] != IMAGE_SIZE or crop_rgb.shape[1] != IMAGE_SIZE:
        crop_rgb = cv2.resize(crop_rgb, (IMAGE_SIZE, IMAGE_SIZE))

    t = torch.from_numpy(crop_rgb).permute(2, 0, 1).float() / 255.0
    mean = torch.tensor(_IMAGENET_MEAN).view(3, 1, 1)
    std  = torch.tensor(_IMAGENET_STD).view(3, 1, 1)
    t = (t - mean) / std

    pred = model(t.unsqueeze(0).to(device))[0].cpu().numpy()
    return pred.astype(np.float32)
'''

R["0f43d81c"] = '''@torch.no_grad()
def track_sequence(
    model,
    frame_paths: List[Path],
    initial_bbox: np.ndarray,
    device: torch.device = DEVICE,
    scale: float = SEARCH_SCALE,
    image_size: int = IMAGE_SIZE,
) -> np.ndarray:
    """
    Track the target object through a sequence.

    Returns predictions: (N, 4) array of full-frame (x, y, w, h) boxes.

    Important: only frame 1 uses the given initial_bbox. Every later frame is
    driven by the model's OWN previous prediction (no ground truth).
    """
    model.eval()
    N = len(frame_paths)
    preds = np.zeros((N, 4), dtype=np.float32)
    preds[0] = np.asarray(initial_bbox, dtype=np.float32)

    cur = preds[0].copy()
    for t in range(1, N):
        img = read_image_rgb(frame_paths[t])
        # 1. Search crop around the previous prediction.
        crop, info = crop_search_region(img, cur, scale=scale, output_size=image_size)
        # 2. Predict the box inside the crop (normalized).
        norm_pred = predict_bbox_in_crop(model, crop, device=device)
        # 3. Convert back to full-frame coordinates.
        frame_box = crop_to_frame_coordinates(norm_pred, info)
        # 4. Keep it inside the image.
        frame_box = clip_bbox_to_image(frame_box, img.shape)
        # 5. Save, and use it as the "previous" box for the next frame.
        preds[t] = frame_box
        cur = frame_box
    return preds
'''

R["e82dbfc7"] = '''def save_predictions_csv(predictions: np.ndarray, output_path: Path):
    """
    Save predictions as:
        frame,x,y,w,h
        1,...
        2,...

    Coordinates are full-frame pixel coordinates (not normalized crop coords).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "x", "y", "w", "h"])
        for i, b in enumerate(predictions, start=1):
            writer.writerow([i, f"{b[0]:.3f}", f"{b[1]:.3f}",
                             f"{b[2]:.3f}", f"{b[3]:.3f}"])
'''

R["47809561"] = '''def run_tracking_on_validation_sequences(
    model,
    data_root: Path,
    sequence_names: List[str],
    output_root: Path,
    device: torch.device = DEVICE,
    scale: float = SEARCH_SCALE,
) -> Dict[str, np.ndarray]:
    """Track each validation sequence (init from GT frame 1) and save CSVs."""
    out_dir = Path(output_root) / "validation"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in sequence_names:
        frames, gts = load_sequence_with_gt(Path(data_root) / name)
        preds = track_sequence(model, frames, gts[0], device=device, scale=scale)
        save_predictions_csv(preds, out_dir / f"{name}_predictions.csv")
        results[name] = preds
        print(f"  [val] {name}: {len(frames)} frames tracked")
    return results
'''

R["6c9f2935"] = '''def run_tracking_on_test_sequences(
    model,
    data_root: Path,
    sequence_names: List[str],
    output_root: Path,
    device: torch.device = DEVICE,
    scale: float = SEARCH_SCALE,
) -> Dict[str, np.ndarray]:
    """Track each test sequence (init from init_rect.txt) and save CSVs."""
    out_dir = Path(output_root) / "test"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for name in sequence_names:
        frames, init_box = load_test_sequence(Path(data_root) / name)
        preds = track_sequence(model, frames, init_box, device=device, scale=scale)
        save_predictions_csv(preds, out_dir / f"{name}_predictions.csv")
        results[name] = preds
        print(f"  [test] {name}: {len(frames)} frames tracked")
    return results
'''

R["16f0ccd6"] = '''print("Tracking validation sequences...")
val_predictions = run_tracking_on_validation_sequences(
    model, DATA_ROOT, val_sequences, OUTPUT_ROOT, device=DEVICE, scale=SEARCH_SCALE,
)

print("\\nTracking test sequences...")
test_predictions = run_tracking_on_test_sequences(
    model, DATA_ROOT, test_sequences, OUTPUT_ROOT, device=DEVICE, scale=SEARCH_SCALE,
)
print("\\nDone. CSVs saved under outputs/validation/ and outputs/test/.")
'''


def main():
    nb = json.loads(NB_PATH.read_text())
    n = 0
    for cell in nb["cells"]:
        cid = cell.get("id")
        if cid in R:
            cell["cell_type"] = "code"
            cell["outputs"] = []
            cell["execution_count"] = None
            cell["source"] = R[cid].splitlines(keepends=True)
            n += 1
    NB_PATH.write_text(json.dumps(nb, indent=1))
    print(f"Updated {n} cells.")


if __name__ == "__main__":
    main()
