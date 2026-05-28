# CLAUDE.md — BBM418/AIN433 Programming Assignment 2
## Single-Object Tracking with CNN-Based Bounding-Box Regression

This file provides context from course lectures (BBM418/AIN433, Spring 2026) so Claude Code can write code that is consistent with the course material and that can be referenced in the discussion sections.

---

## Assignment Overview

Implement a single-object tracking pipeline on video sequences using a CNN-based bounding-box regression model. The tracker is initialized with a ground-truth bounding box in frame 1, then predicts the target location in each subsequent frame using a local search-region crop.

**Parts:**
1. Dataset Loading and Visualization (10 pts)
2. Search-Region Sample Generation (15 pts)
3. CNN-Based Bounding-Box Regression Model (25 pts)
4. Single-Object Tracking on Video Sequences (25 pts)
5. Evaluation and Analysis (25 pts)
6. Leaderboard Bonus (up to +20 pts)

---

## Course Concepts to Use and Reference

### 1. Localization as a Regression Problem (Week 7 & 8)

From the lectures, object localization is framed as a **regression problem**, not a classification problem. The idea taught in class (Week 7, slide "Idea #1: Localization as Regression") is:
- **Input:** Image crop
- **Output:** 4 numbers — box coordinates `(x, y, w, h)`
- **Loss:** L2 (Euclidean) distance between predicted and ground-truth box

This is exactly what Part 3 implements. The course showed this architecture:
```
Image → CNN (conv + pooling) → Feature vector → FC regression head → (x, y, w, h)
```
(Week 7: "Simple Recipe for Classification + Localization", Week 8: detection slides)

Also covered: the **multitask loss** combining classification (Softmax) + localization (L2/Smooth L1). In this assignment we only need the localization branch — no classification head needed.

**Use this in your discussion:** "As taught in Week 7-8, we treat bounding-box prediction as a regression problem. Rather than classifying what the object is, the model regresses directly to the 4 coordinates (x, y, w, h) of the bounding box inside the search crop."

---

### 2. CNN Backbone and Feature Extraction (Week 6 & 8)

From Week 6 (Deep Learning 1) and Week 8 (Detection):
- CNNs extract hierarchical features through stacked convolution + pooling layers.
- A **pretrained ResNet18** backbone (ImageNet-pretrained) is used as the feature extractor.
- The final classification layer of ResNet is replaced with a regression head outputting 4 values.
- This is called **transfer learning / fine-tuning** — starting from ImageNet weights rather than training from scratch.

The course covered ResNet-style architectures in the context of detection (Week 8: Feature Pyramid Networks, Faster R-CNN). The backbone produces a feature map; global average pooling collapses it to a 512-d vector for ResNet18; then a linear layer outputs 4 values.

**Model architecture (as taught):**
```
Search Crop (224×224) → Preprocessing (resize, normalize) → ResNet18 backbone 
→ Global Average Pooling → FC(512 → 4) → Sigmoid → normalized (x, y, w, h) ∈ [0,1]
```

**ImageNet normalization** (required for pretrained ResNet):
```python
mean = [0.485, 0.456, 0.406]
std  = [0.229, 0.224, 0.225]
```

---

### 3. Intersection over Union — IoU (Week 7 & 8)

IoU is the primary evaluation metric taught in class (Week 7: "Comparing Boxes: Intersection over Union", Week 8):
```
IoU = Area of Intersection / Area of Union
```

Thresholds taught in class:
- IoU > 0.5 → "decent" detection
- IoU > 0.7 → "pretty good"
- IoU > 0.9 → "almost perfect"

In this assignment:
- **Mean IoU** over all frames = main metric
- **Success Rate @ IoU ≥ 0.5** = percentage of frames with sufficient overlap
- **Failure** = frame where IoU < 0.1

```python
def compute_iou(box1, box2):
    # box format: (x, y, w, h) — top-left corner + size
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Convert to (x1, y1, x2, y2)
    b1 = [x1, y1, x1+w1, y1+h1]
    b2 = [x2, y2, x2+w2, y2+h2]
    
    inter_x1 = max(b1[0], b2[0])
    inter_y1 = max(b1[1], b2[1])
    inter_x2 = min(b1[2], b2[2])
    inter_y2 = min(b1[3], b2[3])
    
    inter_w = max(0, inter_x2 - inter_x1)
    inter_h = max(0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h
    
    area1 = w1 * h1
    area2 = w2 * h2
    union_area = area1 + area2 - inter_area
    
    return inter_area / union_area if union_area > 0 else 0.0
```

---

### 4. Loss Function — Smooth L1 (Week 6)

From Week 6, multiple loss functions were taught:
- **L1 Loss, L2 Loss, Hinge Loss, Zero-One Loss**

For bounding-box regression, **Smooth L1 Loss** is used (also called Huber loss). It was introduced in Fast R-CNN (covered in Week 8). It is more robust to outliers than MSE:

```
L(x) = 0.5 * x² / β       if |x| < β
       |x| - 0.5 * β       otherwise
```

In PyTorch: `torch.nn.SmoothL1Loss()`

**Why Smooth L1 over L2?** As covered in the R-CNN slides (Week 8), bounding-box regression can have large errors early in training. Smooth L1 behaves like L2 for small errors (stable gradient) and like L1 for large errors (less sensitive to outliers / tracking drift).

---

### 5. Gradient Descent and Optimization (Week 6)

From Week 6 (Deep Learning 1):
- **SGD** (Stochastic Gradient Descent): updates weights on each mini-batch
- **Mini-batch gradient descent**: common sizes are 32, 64, 128
- **Learning rate** is critical — too large → unstable, too small → slow convergence
- **Adam optimizer** adapts the learning rate per parameter, generally more stable than plain SGD

```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

Training loop pattern (as taught):
```python
for epoch in range(num_epochs):
    model.train()
    for batch in train_loader:
        optimizer.zero_grad()
        pred = model(crops)
        loss = criterion(pred, targets)
        loss.backward()   # backpropagation (Week 6)
        optimizer.step()
    
    model.eval()
    with torch.no_grad():
        for batch in val_loader:
            val_loss = criterion(model(crops), targets)
```

---

### 6. Object Detection Pipeline Context (Week 7 & 8)

The sliding window approach (Week 8) motivated why we use a **local search region** instead of scanning the whole image:
- An 800×600 image has ~58 million possible boxes — completely impractical to scan.
- Tracking exploits **temporal continuity**: objects move smoothly between consecutive frames, so we only need to search a small region around the previous prediction.

The **R-CNN family** (Week 8) introduced the key ideas:
- Crop a region → resize to 224×224 → CNN → predict box
- R-CNN Box Regression: predict a "transform" `(tx, ty, tw, th)` to refine a proposal

In this assignment, we use a simpler direct regression: predict the absolute normalized `(x, y, w, h)` inside the crop rather than a delta/transform.

---

### 7. Video and Motion Context (Week 10)

From Week 10 (Video), key relevant concepts:
- **Optical flow**: estimates per-pixel motion between frames. While we don't use optical flow directly, the same assumption underlies our tracker — that objects move smoothly (brightness constancy + small motion).
- **Background subtraction** (Mixture of Gaussians): another approach for detecting moving objects, contrasted with our CNN-based approach.
- **Lucas-Kanade optical flow** errors: large motion, non-smooth motion, appearance changes — these same failure modes affect our tracker.

The **temporal search region** approach in this assignment is directly motivated by the optical flow assumption: because objects don't teleport between frames, we only need to search locally. This is the key bridge between detection (Week 7-8) and video understanding (Week 10).

---

## Implementation Guide

### Dataset Structure
```
<sequence_name>/
    img/
        0001.jpg
        0002.jpg
        ...
    groundtruth_rect.txt   # train/val: x,y,w,h per line
    init_rect.txt          # test only: first frame box
```

### Part 1: Dataset Loading

```python
import os
import glob
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def load_sequence(seq_path):
    """
    Load sorted image paths and ground-truth bounding boxes from a sequence folder.
    Returns:
        img_paths: sorted list of image file paths
        bboxes: np.array of shape (N, 4) in (x, y, w, h) format
    """
    img_dir = os.path.join(seq_path, 'img')
    img_paths = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    
    gt_path = os.path.join(seq_path, 'groundtruth_rect.txt')
    bboxes = []
    with open(gt_path, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                vals = [float(v) for v in line.replace('\t', ',').split(',')]
                bboxes.append(vals[:4])
    bboxes = np.array(bboxes)
    
    # Sanity check: number of frames == number of annotations
    assert len(img_paths) == len(bboxes), \
        f"Frame/annotation mismatch: {len(img_paths)} frames, {len(bboxes)} boxes"
    
    return img_paths, bboxes

def draw_bbox(ax, bbox, color='green', label=None):
    """Draw (x, y, w, h) bounding box on a matplotlib axis."""
    x, y, w, h = bbox
    rect = patches.Rectangle((x, y), w, h,
                               linewidth=2, edgecolor=color, facecolor='none')
    ax.add_patch(rect)
    if label:
        ax.text(x, y - 5, label, color=color, fontsize=9, fontweight='bold')
```

### Part 2: Search-Region Sample Generation

```python
def extract_search_crop(image, prev_bbox, scale=2.5):
    """
    Crop a square search region from `image` centered on prev_bbox.
    Pads image if crop goes outside boundaries.
    
    Args:
        image: PIL Image or np.array (H, W, 3)
        prev_bbox: (x, y, w, h) in full-frame pixel coords
        scale: search region scale factor (default 2.5, from assignment spec)
    
    Returns:
        crop: np.array (crop_size, crop_size, 3)
        crop_info: dict with keys 'x0', 'y0', 'crop_size' for coordinate conversion
    """
    if isinstance(image, np.ndarray):
        H, W = image.shape[:2]
        img_arr = image
    else:
        W, H = image.size
        img_arr = np.array(image)
    
    x, y, w, h = prev_bbox
    cx = x + w / 2
    cy = y + h / 2
    
    crop_size = scale * max(w, h)
    half = crop_size / 2
    
    # Top-left of crop (may be negative = outside image)
    x0 = cx - half
    y0 = cy - half
    
    # Pad image to handle boundary cases.
    # NOTE: the actual notebook uses cv2.copyMakeBorder with BORDER_CONSTANT and
    # gray fill (128,128,128). This is a tunable choice — see "Experimental Knobs".
    pad = int(np.ceil(half)) + 1
    padded = np.pad(img_arr, ((pad, pad), (pad, pad), (0, 0)),
                    mode='constant', constant_values=128)
    
    # Adjust coordinates for padded image
    px0 = int(round(x0 + pad))
    py0 = int(round(y0 + pad))
    px1 = px0 + int(round(crop_size))
    py1 = py0 + int(round(crop_size))
    
    crop = padded[py0:py1, px0:px1]
    # The notebook resizes the crop to 224x224 INSIDE this function (cv2.resize,
    # INTER_LINEAR), and stores `output_size` in CropInfo for inverse mapping.
    
    return crop, {'x0': x0, 'y0': y0, 'crop_size': crop_size}

def bbox_to_crop_coords(bbox_fullframe, crop_info):
    """
    Convert a full-frame (x, y, w, h) bbox to normalized crop coordinates [0, 1].
    
    Steps (as described in assignment):
    1. Shift by crop top-left corner
    2. Normalize by crop size
    """
    x, y, w, h = bbox_fullframe
    x0 = crop_info['x0']
    y0 = crop_info['y0']
    crop_size = crop_info['crop_size']
    
    nx = (x - x0) / crop_size
    ny = (y - y0) / crop_size
    nw = w / crop_size
    nh = h / crop_size
    
    return np.array([nx, ny, nw, nh], dtype=np.float32)

def bbox_from_crop_coords(crop_pred, crop_info):
    """
    Convert normalized crop prediction back to full-frame (x, y, w, h).
    Inverse of bbox_to_crop_coords.
    """
    nx, ny, nw, nh = crop_pred
    x0 = crop_info['x0']
    y0 = crop_info['y0']
    crop_size = crop_info['crop_size']
    
    x = nx * crop_size + x0
    y = ny * crop_size + y0
    w = nw * crop_size
    h = nh * crop_size
    
    return np.array([x, y, w, h], dtype=np.float32)

def generate_training_samples(seq_paths, scale=2.5):
    """
    Generate all (crop, normalized_bbox) training samples from a list of sequences.
    For each consecutive frame pair (t-1, t), crops the search region from frame t
    using the bbox from frame t-1, then records the frame-t bbox in crop coordinates.
    """
    samples = []  # list of (crop_np, target_np)
    for seq_path in seq_paths:
        img_paths, bboxes = load_sequence(seq_path)
        for t in range(1, len(img_paths)):
            img = np.array(Image.open(img_paths[t]).convert('RGB'))
            prev_bbox = bboxes[t - 1]
            curr_bbox = bboxes[t]
            
            crop, crop_info = extract_search_crop(img, prev_bbox, scale=scale)
            target = bbox_to_crop_coords(curr_bbox, crop_info)
            
            samples.append((crop, target))
    return samples
```

### Part 3: CNN Model

```python
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as T

# ImageNet normalization (required for pretrained ResNet — from course Week 6/8)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def get_transform():
    """Preprocessing pipeline: resize → tensor → normalize."""
    return T.Compose([
        T.ToPILImage(),
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

class TrackingRegressor(nn.Module):
    """
    CNN-based bounding-box regressor.

    Architecture (consistent with Week 7 'Simple Recipe for Classification +
    Localization' but with a small MLP head instead of a single linear layer):
        ResNet18 backbone (ImageNet-pretrained, classifier replaced by Identity)
        → 512-d feature vector
        → Linear(512 → 256) → ReLU → Dropout(0.1) → Linear(256 → 4)
        → Sigmoid (applied in forward) → normalized (x, y, w, h) ∈ [0, 1]

    The MLP head + dropout is a small regularizer over the assignment's minimal
    "single FC layer" recipe; both head depth and dropout rate are experimental
    knobs (see "Experimental Knobs" below).
    """
    def __init__(self, pretrained=True):
        super().__init__()
        weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        backbone = models.resnet18(weights=weights)
        feat_dim = backbone.fc.in_features   # 512 for ResNet18
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.head = nn.Sequential(
            nn.Linear(feat_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.1),
            nn.Linear(256, 4),
        )
    
    def forward(self, x):
        feat = self.backbone(x)
        return torch.sigmoid(self.head(feat))   # outputs ∈ [0, 1]


# Training function
def train_model(model, train_loader, val_loader, num_epochs=10, lr=1e-4, device='cuda'):
    """
    Train the tracker CNN.

    - Optimizer: Adam (Week 6).
    - Loss: SmoothL1Loss(beta=0.1). The smaller beta makes the loss act more like
      pure L1 over a wider error range — the typical normalized-coord residual is
      already small, so the default beta=1.0 would put almost everything in the
      quadratic regime. (Week 6: loss functions; Week 8: R-CNN box regression.)
    - LR scheduler: StepLR(step_size=5, gamma=0.5) — halves the learning rate
      every 5 epochs. Both knobs are experimental.
    - Model selection: best checkpoint by VALIDATION CROP-LEVEL MEAN IoU, not by
      val loss. This is closer to the actual evaluation metric and tends to
      track the leaderboard score better.
    """
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss(beta=0.1)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    
    train_losses, val_losses, val_ious = [], [], []
    best_iou = -1.0
    best_weights = None
    
    for epoch in range(num_epochs):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        for crops, targets in train_loader:
            crops, targets = crops.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = model(crops)
            loss = criterion(preds, targets)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(crops)
        train_losses.append(epoch_loss / len(train_loader.dataset))
        
        # --- Validation: loss + crop-level mean IoU ---
        model.eval()
        val_loss, val_iou = 0.0, 0.0
        with torch.no_grad():
            for crops, targets in val_loader:
                crops, targets = crops.to(device), targets.to(device)
                preds = model(crops)
                val_loss += criterion(preds, targets).item() * len(crops)
                val_iou += mean_iou_batch(preds, targets) * len(crops)
        val_losses.append(val_loss / len(val_loader.dataset))
        val_ious.append(val_iou / len(val_loader.dataset))
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{num_epochs} | train {train_losses[-1]:.4f} | "
              f"val {val_losses[-1]:.4f} | val IoU {val_ious[-1]:.4f}")
        
        # Select best model by val IoU (not val loss).
        if val_ious[-1] > best_iou:
            best_iou = val_ious[-1]
            best_weights = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_weights, "checkpoint/best_model.pt")
    
    model.load_state_dict(best_weights)
    return model, train_losses, val_losses, val_ious
```

### Part 4: Tracker

```python
def track_sequence(model, img_paths, init_bbox, transform, scale=2.5, device='cuda'):
    """
    Run the tracker on a sequence.
    
    Tracking loop (as specified in assignment):
    1. Initialize with ground-truth bbox from frame 1
    2. For each next frame:
       a. Crop search region around previous predicted bbox
       b. Preprocess and run model
       c. Convert predicted crop coords back to full-frame coords
       d. Save prediction, use as new prev bbox
    
    Returns:
        predictions: list of (x, y, w, h) in full-frame coords, one per frame
    """
    model.eval()
    predictions = [np.array(init_bbox, dtype=np.float32)]
    prev_bbox = np.array(init_bbox, dtype=np.float32)
    
    for t in range(1, len(img_paths)):
        img = np.array(Image.open(img_paths[t]).convert('RGB'))
        
        # Crop search region using previous prediction
        crop, crop_info = extract_search_crop(img, prev_bbox, scale=scale)
        
        # Preprocess (same pipeline as training — critical!)
        crop_tensor = transform(crop).unsqueeze(0).to(device)
        
        # Predict normalized bbox in crop coords
        with torch.no_grad():
            pred_norm = model(crop_tensor).cpu().numpy()[0]
        
        # Convert back to full-frame coordinates
        pred_fullframe = bbox_from_crop_coords(pred_norm, crop_info)
        predictions.append(pred_fullframe)
        prev_bbox = pred_fullframe
    
    return predictions
```

### Part 5: Evaluation Metrics

```python
def evaluate_sequence(predictions, ground_truth):
    """
    Compute tracking metrics for a single sequence.
    
    Metrics (as specified in assignment, based on IoU taught in Week 7-8):
    - Mean IoU over all frames
    - Success Rate @ IoU >= 0.5
    - Average Center Error (Euclidean distance)
    - Failure Count (frames with IoU < 0.1)
    """
    ious, center_errors = [], []
    
    for pred, gt in zip(predictions, ground_truth):
        iou = compute_iou(pred, gt)
        ious.append(iou)
        
        # Center error
        pred_cx = pred[0] + pred[2] / 2
        pred_cy = pred[1] + pred[3] / 2
        gt_cx   = gt[0] + gt[2] / 2
        gt_cy   = gt[1] + gt[3] / 2
        center_errors.append(np.sqrt((pred_cx - gt_cx)**2 + (pred_cy - gt_cy)**2))
    
    ious = np.array(ious)
    center_errors = np.array(center_errors)
    
    return {
        'mean_iou':       float(np.mean(ious)),
        'success_rate':   float(np.mean(ious >= 0.5)),
        'avg_center_err': float(np.mean(center_errors)),
        'failure_count':  int(np.sum(ious < 0.1)),
    }
```

---

## Hyperparameters (Current Notebook Defaults)

These are the values currently set in cell 13 of `student_draft.ipynb`.

| Parameter | Default | Notes |
|---|---|---|
| `SEARCH_SCALE` | 2.5 | Search region = `2.5 × max(w, h)` around previous bbox |
| `IMAGE_SIZE` | 224 | Crop resized to this before the backbone (ResNet18 standard) |
| `BATCH_SIZE` | 32 | Reduce if OOM |
| `LEARNING_RATE` | 1e-4 | Adam initial LR |
| `NUM_EPOCHS` | 10 | |
| `NUM_WORKERS` | 2 (CUDA) / 0 (CPU) | DataLoader workers |
| optimizer | Adam | |
| loss | `SmoothL1Loss(beta=0.1)` | β smaller than default 1.0 because residuals are in normalized [0,1] |
| LR scheduler | `StepLR(step_size=5, gamma=0.5)` | Halve LR every 5 epochs |
| backbone | ResNet18, ImageNet-pretrained | `fc` replaced with regression MLP head |
| head | `Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→4) → Sigmoid` | |
| pad value (crop) | `(128, 128, 128)` constant gray | Used when crop goes outside image |
| crop interpolation | `cv2.INTER_LINEAR` | |
| target clip | `[-0.5, 1.5]` | Guards against outlier targets when curr box leaves crop |
| jitter (train aug) | shift ±0.1·max(w,h), scale ∈ [0.9, 1.1] | Applied to `prev_bbox`; off for val/test |
| model selection | best by **val crop-level mean IoU** | Not val loss |
| seed | 42 | `torch.backends.cudnn.deterministic = True` |

---

## Experimental Knobs (things to vary later)

The PDF lists hyperparameters as a **starting point, not a requirement**. Everything below is fair game to ablate — record `(knob, value, val crop-level IoU, val sequence-level IoU)` in a CSV so we can build a comparison table for the report.

### 1. Hyperparameters
- **`SEARCH_SCALE`** (2.5). Try 2.0 and 3.0. Larger scale → more context, but the object occupies fewer pixels in the 224×224 crop (worse fine localization). Smaller scale → fast objects leave the crop entirely.
- **`BATCH_SIZE`** (32). Try 16 / 64 — affects gradient noise and BN stats in ResNet.
- **`LEARNING_RATE`** (1e-4). Try 3e-4, 5e-5 — interacts with scheduler.
- **`NUM_EPOCHS`** (10). Train longer (20, 30) and check whether val IoU is still climbing or already plateauing.
- **LR scheduler**: `StepLR(5, 0.5)` vs `CosineAnnealingLR(T_max=num_epochs)` vs none vs `ReduceLROnPlateau` on val IoU.

### 2. Crop boundary handling
Current: constant gray `(128, 128, 128)`. Alternatives:
- **Edge-replicate** (`cv2.BORDER_REPLICATE`) — keeps texture stats natural at the border but can streak.
- **Reflect** (`cv2.BORDER_REFLECT_101`) — mirrors content; no streaks.
- **Clip the crop to image** (no padding) — search region becomes non-square; breaks the coordinate math unless re-centered / shrunk.
- **Black (0,0,0)** or **mean ImageNet color** as the constant fill.

### 3. Augmentation
Current: jitter `prev_bbox` by ±0.1·max(w,h) shift and ×[0.9, 1.1] scale during training only.
- **Turn jitter OFF** → model trained on a perfect prev box; test-time tracker noise becomes a distribution shift and drift worsens.
- **Increase jitter** (±0.2 shift, ×[0.8, 1.2] scale) — simulates a noisier tracker.
- **Color jitter** (brightness/contrast/saturation) — counters Week-10 brightness-constancy failure modes.
- **Horizontal flip** — easy win if not orientation-sensitive (flip box x as well).
- **Gaussian noise** on prev_bbox coords instead of uniform jitter.
- **Random small zoom** of the search region itself.

### 4. Output activation
Current: `sigmoid` → outputs in [0, 1] (the crop); targets clipped to [-0.5, 1.5].
- **No activation** — rely on the loss; model can predict outside the crop (useful when target partially leaves search region).
- **`tanh`** rescaled to [-0.5, 1.5] — symmetric range that matches the target-clip range.
- **`(dx, dy, log dw, log dh)` offsets from crop center** — R-CNN parameterization (Week 8). Different output space; needs matching loss and inverse mapping.

### 5. Loss function
Current: `SmoothL1Loss(beta=0.1)`.
- **`beta=1.0`** (PyTorch default) — closer to pure L2 for typical residuals.
- **`nn.L1Loss`** or **`nn.MSELoss`** — Week-6 baselines.
- **GIoU / DIoU / CIoU loss** — directly optimizes box overlap; careful with normalized coords and w/h ≤ 0 early in training.
- **Weighted sum** of Smooth L1 on (x, y) and Smooth L1 on (w, h) — center vs size errors hurt tracking differently.

### 6. Backbone
Current: ResNet18, all layers fine-tuned.
- **Bigger**: ResNet34 / ResNet50 — better features, slower.
- **Lighter**: MobileNetV2/V3, EfficientNet-B0 — faster, lets us train more epochs in the same budget.
- **Frozen backbone**: train only the head — fast baseline, likely underfits.
- **Partially frozen**: freeze `conv1`–`layer2`, fine-tune `layer3`–`layer4` + head (standard transfer-learning compromise from Week 6/8).

### 7. Regression head
Current: `Linear(512→256) → ReLU → Dropout(0.1) → Linear(256→4)`.
- **Minimal**: single `Linear(512→4)` — the literal Week-7 slide version.
- **Deeper**: 3 FC layers with more hidden units.
- **Dropout** ∈ {0.0, 0.2, 0.5}.
- **BatchNorm / LayerNorm** between FC layers.

### 8. Sampling strategy (which (prev, curr) pairs to train on)
Current: every consecutive pair `(t-1, t)`.
- **Skip-frame pairs** `(t-k, t)`, k ∈ {2, 3, 5} — teaches larger motion (closer to what happens at test time after a drift).
- **Mixed gap sampling**: random k per sample.
- **Filter near-static pairs** (box barely moves → trivial).
- **Filter occluded / out-of-frame pairs** when annotations indicate them.

### 9. Reproducibility caveats
Even with `SEED=42` + `cudnn.deterministic=True`, results differ across torch/CUDA/cuDNN versions by ±0.01–0.05 IoU. Log the environment alongside the result.

### 10. Tracker logic (Part 4) — no retraining needed
- **Search scale at tracking time ≠ training scale**: train at 2.5, track at 3.0 (more room for fast frames) or 2.0 (more pixels on target).
- **Smoothing**: `new_box = α · pred + (1-α) · prev_box`, α ∈ [0.5, 1.0]. Reduces jitter, slows reaction.
- **Re-detection on failure**: if predicted w or h jumps beyond a ratio threshold, or center moves more than the search radius, revert to previous prediction.
- **Multi-scale search**: run at {2.0, 2.5, 3.0}, pick the most plausible box (closest to crop center, or smallest size change).
- **Aspect-ratio constraint**: keep `w/h` near the initial frame's ratio (single-object assumption).
- **Box clipping**: tune whether to clip before or after using as next prev box.

### 11. Bonus / leaderboard ideas
- **Online fine-tuning** on frame 1 of the test sequence (a few gradient steps using the GT init box) — adapts to target appearance.
- **Ensembles**: two seeds / two scales, average predictions.
- **Larger backbone + longer training** if compute allows.
- **Targeted augmentation** chosen after analyzing val-set failure modes.

---

## Output File Format

**Validation/Test predictions** (`outputs/<seq_name>_predictions.csv`):
```
frame,x,y,w,h
1,198,214,34,81
2,197.2,214.1,34.5,80.7
```

**Test predictions** (`outputs/test/<seq_name>_predictions.csv`) — same format.

**Model checkpoint:** `checkpoint/best_model.pt`

---

## Experimental Knobs (Things to Vary for Ablation / Leaderboard)

These are the *differable* design choices in the current pipeline. Each row is
something we can flip in isolation and re-run training + tracking. When running
an ablation, change exactly one knob per run, record the resulting validation
table (Mean IoU, Success@0.5, Avg Center Err, Failures), and tie the result back
to a course concept in the discussion section.

### A. Search-region and crop knobs (Part 2)

| Knob | Current | Alternatives to try | What it tests |
|---|---|---|---|
| `SEARCH_SCALE` | 2.5 | 1.5, 2.0, 3.0, 4.0 | Trade-off between "object can leave crop on fast motion" (small scale) vs "target shrinks → less detail" (large scale). Ties to optical-flow / small-motion assumption (Week 10). |
| `IMAGE_SIZE` | 224 | 128, 160, 192, 256 | Effective spatial resolution at the regression head; speed vs accuracy. |
| Crop pad value | `(128, 128, 128)` constant gray | `BORDER_REPLICATE` (edge), `(0,0,0)` black, `BORDER_REFLECT` | Out-of-image regions create artificial edges; gray is neutral but replicate keeps texture continuity. |
| Resize interpolation | `INTER_LINEAR` | `INTER_AREA` (downscale), `INTER_CUBIC` | Mostly minor — interesting if `IMAGE_SIZE` is small. |
| Coordinate target | absolute `(x,y,w,h)` in crop | log-space `(w,h)`; deltas from crop center; (cx,cy,w,h) | R-CNN uses log-space + deltas (Week 8); could improve scale robustness. |
| Target clip range | `[-0.5, 1.5]` | `[0, 1]` (drop out-of-crop samples), `[-1, 2]` | Controls how aggressively we train on "target left the crop" cases. |

### B. Model knobs (Part 3)

| Knob | Current | Alternatives to try | What it tests |
|---|---|---|---|
| Backbone | ResNet18 (pretrained) | ResNet34, ResNet50, MobileNetV3, EfficientNet-B0 | Capacity vs speed; transfer-learning effect (Week 6/8). |
| Pretrained | True (ImageNet) | False (train from scratch) | Direct measurement of transfer-learning benefit on a small dataset (Week 6/8). |
| Backbone freezing | All trainable | Freeze backbone (head only), freeze first 2 stages, freeze BN only | Classic transfer-learning ablation; useful when train set is small. |
| Head depth | 2-layer MLP (512→256→4) | 1 linear (512→4), 3-layer MLP, wider hidden (512→512→4) | The assignment spec only requires single FC; the deeper head is a knob. |
| Dropout (head) | 0.1 | 0.0, 0.2, 0.5 | Regularization (Week 6). |
| Output activation | Sigmoid | None (raw), Tanh + rescale | Sigmoid clips outputs to `[0,1]` — discuss whether that's actually wanted when target may lie in `[-0.5, 1.5]`. |
| Global pooling | implicit ResNet GAP | GAP + GMP concat, attention pooling | Feature aggregation. |

### C. Loss knobs

| Knob | Current | Alternatives to try | What it tests |
|---|---|---|---|
| Loss type | `SmoothL1Loss(beta=0.1)` | `MSELoss` (L2), `L1Loss`, IoU loss, GIoU loss, DIoU loss | Course covers L1/L2/Smooth L1 (Week 6); IoU-family losses optimize the eval metric directly. |
| Smooth-L1 `beta` | 0.1 | 0.01, 0.5, 1.0 (default) | Where the quadratic→linear transition happens, given that residuals live in normalized crop coords. |
| Per-coord weighting | uniform | `(w, h)` weighted higher than `(x, y)`, or vice versa | Position vs size accuracy. |

### D. Optimization knobs

| Knob | Current | Alternatives to try | What it tests |
|---|---|---|---|
| Optimizer | Adam | SGD+momentum (0.9), AdamW (with `weight_decay=1e-4`), RMSprop | Week 6 optimizers. |
| `LEARNING_RATE` | 1e-4 | 3e-5, 5e-5, 3e-4, 1e-3 | Sensitivity / convergence (Week 6). |
| LR scheduler | `StepLR(5, 0.5)` | Constant, `CosineAnnealingLR`, `ReduceLROnPlateau(val_iou)`, warmup | Convergence behavior; revisits Week 6 LR discussion. |
| `BATCH_SIZE` | 32 | 8, 16, 64 | Mini-batch GD (Week 6). |
| `NUM_EPOCHS` | 10 | 5, 15, 25 | Under/overfitting; show curves. |
| Weight decay | 0 | 1e-5, 1e-4, 5e-4 | Regularization. |
| Gradient clipping | off | `clip_grad_norm_(1.0)` | Stability for large residuals early in training. |

### E. Data / augmentation knobs

| Knob | Current | Alternatives to try | What it tests |
|---|---|---|---|
| Prev-bbox jitter | shift ±0.1·max(w,h), scale ∈ [0.9, 1.1] | off; larger (±0.2, [0.8, 1.2]); only-shift / only-scale | This is the core "simulate tracker drift at training time" knob. The intuition: at test time the previous box is the *model's own prediction*, not GT, so the search-crop center distribution differs from training. Jitter narrows that gap. |
| Color jitter | off | brightness/contrast/saturation ±0.2 | Robustness to appearance change (a Lucas-Kanade failure mode, Week 10). |
| Horizontal flip | off | p=0.5 (flip crop *and* target) | Generic augmentation; need to be careful with target mirroring. |
| Sample weighting | uniform | up-weight frames where prev→curr displacement is large | Focuses learning on harder transitions. |
| Frame skipping | consecutive pairs `(t-1, t)` | `(t-k, t)` with `k ∈ {1,2,3}` randomly | Simulates fast motion / dropped frames. |

### F. Inference / tracker knobs (Part 4)

| Knob | Current | Alternatives to try | What it tests |
|---|---|---|---|
| Search-region scale at test | same as train (2.5) | larger at test (e.g. 3.0) | Decouple train and test crop scales. |
| Center policy | use last prediction directly | use moving-average of last `k` predictions | Smooths drift. |
| Failure recovery | none | if predicted `w·h` collapses or jumps >2× from previous, reuse previous bbox | Heuristic for catastrophic drift. |
| Multi-scale search | single crop | average predictions over `scale ∈ {2.0, 2.5, 3.0}` | Test-time scale ensembling. |
| Test-time augmentation | none | predict on crop + horizontal flip, average | Standard TTA. |
| Box smoothing | none | exponential moving average over predicted boxes (EMA factor 0.7) | Reduces high-frequency jitter. |

### G. Bookkeeping for experiments

- Always keep `SEED=42` fixed when running ablations so differences are due to the knob, not the seed.
- Save each run's `outputs/validation/*_predictions.csv` to a subfolder like `outputs/runs/<run_name>/` before kicking off the next experiment, so we don't overwrite results.
- Log: knob changed, val Mean IoU per sequence, mean over val, plus train/val loss curves.

---

## Discussion Section Reference Points

When writing the analysis/discussion cells in the notebook, connect to these course concepts:

1. **Why search regions work** → optical flow assumption (Week 10): objects move smoothly, small motion between frames
2. **Why pretrained ResNet** → transfer learning (Week 6/8): the dataset is small; ImageNet features generalize well
3. **Localization as regression** → Week 7: "Treat localization as a regression problem" (Single Object Detection slide)
4. **Smooth L1 loss** → R-CNN box regression (Week 8): designed specifically for bbox regression robustness
5. **IoU evaluation** → Week 7/8: IoU > 0.5 = "decent", standard metric from ImageNet localization challenge
6. **Failure modes to discuss** → from Week 10 (Lucas-Kanade errors, same apply here):
   - Large/fast motion → object leaves search region
   - Brightness/appearance change → model features fail
   - Non-smooth motion → temporal assumption violated
   - Background clutter → similar-looking distractors confuse regressor
   - Occlusion → object disappears, tracking drifts to background
7. **Tracking drift** → a fundamental challenge not present in static detection (Week 7/8): errors accumulate because each prediction is used as the next frame's initialization. This is the key difference between crop-level validation and sequence-level tracking validation.
8. **Why we jitter the previous box during training** → without jitter, training only sees perfect prev boxes (the GT shifted by one frame). At test time the tracker feeds itself noisy predictions — a **distribution shift between train and test** (Week 6 generalization). Jitter narrows that gap. This is also why the same model can have high crop-level IoU but low sequence-level IoU.
9. **Why sigmoid on the output** → bounds predictions to the crop and makes the optimization landscape better behaved early in training (no runaway predictions). Cost: when the target genuinely leaves the search region the model cannot represent it (a structural failure mode worth pointing out).
10. **Why a small head with dropout** → dropout is a Week-6 regularizer; the head is the layer most likely to overfit on a small dataset, so dropout there (not in the backbone) is the standard transfer-learning recipe.
11. **Why an LR scheduler (StepLR)** → from Week 6: high LR explores, low LR refines. Halving every 5 epochs is the discrete version of "decrease LR when loss plateaus." Document the train/val curve shape with vs without the scheduler to make this concrete.
12. **Why selecting the best checkpoint by val crop-level IoU, not val loss** → val loss is in Smooth L1 units (a proxy); IoU is the actual evaluation metric (Week 7-8). Different loss values can map to the same IoU and vice versa. Picking by IoU correlates better with the leaderboard.
13. **Search scale trade-off** → small scale: fast objects exit the crop (Lucas-Kanade "large motion" failure, Week 10). Large scale: the object occupies fewer of the 224 pixels (loss of spatial precision — the same trade-off as image pyramid level choice in classical CV).
14. **Why pad with gray (not edge replication)** → constant gray is visually neutral and matches the mean of the ImageNet-normalized inputs after normalization; edge replication can synthesize fake texture that the regressor latches onto near image borders. A direct ablation here can be discussed.
15. **Train/test mismatch in box parameterization** → we predict absolute `(x, y, w, h)` in crop coords rather than the R-CNN `(dx, dy, log dw, log dh)` deltas (Week 8). Absolute coords are simpler but lose the size-invariance the log-parameterization gives — worth mentioning when discussing how the model handles size changes.
8. **Train/test distribution mismatch + prev-bbox jitter augmentation** → at training time the "previous box" is ground truth, but at test time it is the model's own (noisy) prediction. The jitter augmentation in `SearchRegionDataset._jitter_prev_bbox` simulates the test-time noise distribution so the model sees similar crop offsets/scales during training. Tie this to the general ML idea of train/test domain shift, and to the Week 10 observation that small-motion assumptions break down under appearance changes.
9. **Dropout in the regression head** → regularization technique from Week 6 (Deep Learning 1); small dataset + ResNet18 (11M params) easily overfits, so even a small `Dropout(0.1)` helps generalization.
10. **Learning-rate schedule (StepLR)** → Week 6 lecture on learning rate sensitivity: starting at 1e-4 lets the head adapt fast, halving every 5 epochs lets the backbone fine-tune without disrupting features learned on ImageNet.
11. **Sigmoid output activation** → the prediction is forced into `[0, 1]` so the model can never emit a degenerate negative box. Discuss the trade-off: when the target box partially leaves the crop, the true normalized coordinates can fall outside `[0, 1]`, which a sigmoid cannot represent — hence the `target = clip(target, -0.5, 1.5)` choice when *not* using sigmoid would be preferable.
12. **Smooth L1 β = 0.1** → because targets are normalized to `[0, 1]`, typical residuals are well below 1. The default `beta=1.0` would put almost every error in the quadratic regime, effectively reducing Smooth L1 to MSE. Lowering β to 0.1 restores the L1-like (robust) regime for medium errors, which is exactly the Week 8 motivation for using Smooth L1 over MSE for bounding-box regression.
13. **Best-checkpoint selection by val IoU instead of val loss** → the loss is a *surrogate* for the real metric. Selecting on the actual metric (mean IoU over validation crops) avoids the surrogate-vs-metric gap discussed informally throughout Weeks 6–8.
14. **MLP head vs single FC** → Week 7's "Simple Recipe" uses a single FC. Adding `Linear(512→256)+ReLU+Dropout+Linear(256→4)` adds a small amount of nonlinearity between the frozen ImageNet feature space and the 4-D coordinate output — useful because ImageNet features are tuned for *classification*, not localization.
15. **Why ResNet (over plain CNN)** → Week 6 introduced residual connections to address the vanishing-gradient/degradation problem in deep networks. We don't actually need deep nets here, but ResNet18 is the smallest pretrained option in the residual family and is the standard transfer-learning starting point.
16. **Why GAP (Global Average Pooling) at the end of the backbone** → covered as part of modern CNN design (Week 6/8): GAP gives a fixed-size 512-d vector regardless of input resolution, and acts as a structural regularizer compared to a flatten-then-FC approach.
17. **Localization vs detection vs tracking distinctions** → Week 7 (single-object localization, one box per image) is essentially what we do *per crop*. Week 8 (detection, many boxes per image) is the harder problem we avoid by using a search region. Week 10 (video / motion) is what makes the *sequence-level* problem harder than per-frame localization.
18. **IoU-aware vs L2-aware optimization** → bonus point: the loss optimizes coordinate error, but the metric is IoU. The two disagree most when the box is small relative to the crop. This motivates IoU/GIoU losses as a future direction (a knob in the ablation table above).

---

## Reproducibility

Add this cell at the top of the notebook (the draft notebook provides a seed cell):

```python
import random
import numpy as np
import torch

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

---

## Checklist

- [ ] `load_sequence()` returns sorted paths + (N,4) bboxes; sanity check frame count == bbox count
- [ ] `draw_bbox()` works for visualization throughout
- [ ] `extract_search_crop()` handles boundary cases (padding approach explained in notebook)
- [ ] `bbox_to_crop_coords()` and `bbox_from_crop_coords()` are exact inverses — verify with a sanity check
- [ ] `TrackingRegressor` uses pretrained ResNet18 with `fc` replaced by an MLP regression head; sigmoid applied in `forward`
- [ ] Same preprocessing pipeline used in training, validation, and tracking (no random augmentation during val/test)
- [ ] Training reports train loss, val loss, and val crop-level mean IoU per epoch
- [ ] Loss curves plotted and saved to `outputs/`
- [ ] Best model weights saved to `checkpoint/best_model.pt`
- [ ] Tracking loop uses each prediction as the next frame's search-region center
- [ ] Predictions saved as CSV in `outputs/<seq_name>_predictions.csv` (full-frame coords)
- [ ] Test predictions in `outputs/test/<seq_name>_predictions.csv`
- [ ] Evaluation table (Mean IoU, Success Rate @0.5, Avg Center Error, Failure Count) for all val sequences
- [ ] At least 3 success + 3 failure case analyses with course-grounded explanations
- [ ] Notebook runs top-to-bottom without errors