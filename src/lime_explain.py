# src/lime_explain.py
import numpy as np
import cv2
import torch
import torch.nn.functional as F
from lime import lime_image
from skimage.segmentation import mark_boundaries, slic
import matplotlib.pyplot as plt
import albumentations as A
from albumentations.pytorch import ToTensorV2


DR_CLASSES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']


def build_predict_fn(model, device, img_size=224):
    """
    Build a LIME-compatible prediction function.
    LIME passes batches of (N, H, W, 3) uint8 perturbed images.
    """
    transform = A.Compose([
        A.Resize(img_size, img_size),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

    def predict_fn(images):
        model.eval()
        tensors = []
        for img in images:
            t = transform(image=img.astype(np.uint8))['image']
            tensors.append(t)
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            logits = model(batch)
            probs  = F.softmax(logits, dim=1).cpu().numpy()
        return probs

    return predict_fn


def generate_lime_explanation(model, image_np, pred_class,
                               device='cuda', img_size=224,
                               num_samples=1000, num_features=8,
                               positive_only=True):
    """
    Run LIME on a single retinal image.

    Args:
        model:         trained DRClassifier
        image_np:      (H,W,3) uint8 numpy — original, NOT normalized
        pred_class:    int — class to explain
        num_samples:   LIME perturbation count (higher = slower but better)
        num_features:  number of superpixels to highlight
        positive_only: if True, only show regions supporting prediction

    Returns:
        explanation:   LIME ImageExplanation object
        overlay:       (H,W,3) uint8 — superpixels drawn on image
        top_segments:  list of (segment_id, weight) sorted by importance
    """
    predict_fn = build_predict_fn(model, device, img_size)

    explainer = lime_image.LimeImageExplainer(random_state=42)
    explanation = explainer.explain_instance(
        image_np,
        predict_fn,
        top_labels=5,
        hide_color=0,
        num_samples=num_samples,
        segmentation_fn=lambda x: slic(
            x, n_segments=80, compactness=10,
            sigma=1, start_label=0
        ),
        random_seed=42
    )

    # Get image + mask for predicted class
    temp_img, mask = explanation.get_image_and_mask(
        pred_class,
        positive_only=positive_only,
        num_features=num_features,
        hide_rest=False
    )

    overlay = mark_boundaries(temp_img / 255.0, mask, color=(1, 0.8, 0))
    overlay = (np.clip(overlay, 0, 1) * 255).astype(np.uint8)

    # Extract top segment weights
    seg_weights = explanation.local_exp.get(pred_class, [])
    top_segments = sorted(seg_weights, key=lambda x: abs(x[1]), reverse=True)

    return explanation, overlay, top_segments


def lime_heatmap(explanation, pred_class, image_shape):
    """
    Convert LIME weights into a smooth heatmap the same size as the image.
    Useful for overlaying on Grad-CAM for combined XAI visualization.
    """
    segments = explanation.segments
    weights  = dict(explanation.local_exp.get(pred_class, []))

    heatmap = np.zeros(image_shape[:2], dtype=np.float32)
    for seg_id, weight in weights.items():
        heatmap[segments == seg_id] = weight

    # Normalize to [0,1]
    pos = np.clip(heatmap, 0, None)
    if pos.max() > 0:
        pos /= pos.max()

    return pos


def plot_lime_analysis(image_np, overlay, top_segments,
                        pred_class, probs, save_path=None):
    """
    Three-panel LIME analysis figure:
    original | LIME overlay | segment weight bar chart
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].imshow(image_np)
    axes[0].set_title('Original fundus image', fontweight='bold')
    axes[0].axis('off')

    axes[1].imshow(overlay)
    axes[1].set_title(
        f'LIME — top regions supporting\n"{DR_CLASSES[pred_class]}"',
        fontweight='bold')
    axes[1].axis('off')

    # Segment weight bar chart
    n = min(10, len(top_segments))
    seg_ids  = [f'Seg {s[0]}' for s in top_segments[:n]]
    weights  = [s[1] for s in top_segments[:n]]
    colors   = ['#2ecc71' if w > 0 else '#e74c3c' for w in weights]
    axes[2].barh(seg_ids[::-1], weights[::-1], color=colors[::-1])
    axes[2].axvline(0, color='black', linewidth=0.8)
    axes[2].set_title('Segment contributions', fontweight='bold')
    axes[2].set_xlabel('LIME weight')

    plt.suptitle(
        f'LIME Analysis  |  Confidence: {probs[pred_class]*100:.1f}%',
        fontsize=13, fontweight='bold')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white')
    return fig