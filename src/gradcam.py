# src/gradcam.py
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from pytorch_grad_cam import GradCAM, GradCAMPlusPlus, ScoreCAM, EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DR_CLASSES = ['No DR', 'Mild DR', 'Moderate DR', 'Severe DR', 'Proliferative DR']


def denormalize(tensor):
    """Convert normalized tensor back to displayable RGB float32 [0,1]."""
    mean = np.array([0.485, 0.456, 0.406])
    std  = np.array([0.229, 0.224, 0.225])
    img  = tensor.permute(1, 2, 0).cpu().numpy()
    img  = std * img + mean
    return np.clip(img, 0, 1).astype(np.float32)


def get_target_layer(model):
    """Return last conv layer of EfficientNet backbone for Grad-CAM."""
    # Works for both B0 and B4
    return model.backbone.blocks[-1][-1].conv_pwl


def generate_gradcam(model, image_tensor, target_class=None,
                     method='gradcam++', device='cuda'):
    """
    Generate a Grad-CAM heatmap overlay.

    Args:
        model:         trained DRClassifier
        image_tensor:  (C,H,W) normalized tensor
        target_class:  int class index; if None uses predicted class
        method:        'gradcam', 'gradcam++', 'scorecam', 'eigencam'
        device:        'cuda' or 'cpu'

    Returns:
        overlay       (H,W,3) uint8  — heatmap blended on original
        grayscale_cam (H,W)   float32 — raw activation map [0,1]
        pred_class    int
        probs         (5,) float32
    """
    model.eval()
    inp = image_tensor.unsqueeze(0).to(device)

    # Get prediction first
    with torch.no_grad():
        logits = model(inp)
        probs  = torch.softmax(logits, dim=1)[0].cpu().numpy()
    pred_class = int(probs.argmax()) if target_class is None else target_class

    method_map = {
        'gradcam':   GradCAM,
        'gradcam++': GradCAMPlusPlus,
        'scorecam':  ScoreCAM,
        'eigencam':  EigenCAM,
    }
    CAMClass = method_map.get(method, GradCAMPlusPlus)
    target_layers = [get_target_layer(model)]

    with CAMClass(model=model, target_layers=target_layers) as cam:
        targets = [ClassifierOutputTarget(pred_class)]
        grayscale_cam = cam(input_tensor=inp, targets=targets)[0]  # (H,W)

    rgb_img = denormalize(image_tensor)
    overlay = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

    return overlay, grayscale_cam, pred_class, probs


def analyze_heatmap_regions(grayscale_cam, threshold=0.5):
    """
    Map high-activation regions to anatomical retinal terms.
    Returns list of region name strings.
    """
    h, w = grayscale_cam.shape
    regions = {
        'central macula':   grayscale_cam[h//3:2*h//3, w//3:2*w//3].mean(),
        'superior retina':  grayscale_cam[:h//3,        :].mean(),
        'inferior retina':  grayscale_cam[2*h//3:,      :].mean(),
        'nasal quadrant':   grayscale_cam[:,             :w//3].mean(),
        'temporal quadrant':grayscale_cam[:,             2*w//3:].mean(),
    }
    active = [r for r, v in sorted(regions.items(),
              key=lambda x: x[1], reverse=True) if v > threshold]
    return active[:3] if active else ['diffuse retinal areas']


def multi_method_comparison(model, image_tensor, pred_class,
                             device='cuda', save_path=None):
    """
    Generate a 4-panel figure comparing Grad-CAM, Grad-CAM++,
    EigenCAM, and LIME side by side.
    Returns fig object.
    """
    methods = ['gradcam', 'gradcam++', 'eigencam']
    titles  = ['Grad-CAM', 'Grad-CAM++', 'EigenCAM']

    rgb = denormalize(image_tensor)
    overlays = []
    for m in methods:
        overlay, _, _, _ = generate_gradcam(
            model, image_tensor, pred_class, method=m, device=device)
        overlays.append(overlay)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].imshow(rgb)
    axes[0].set_title('Original', fontsize=11, fontweight='bold')
    axes[0].axis('off')

    for ax, overlay, title in zip(axes[1:], overlays, titles):
        ax.imshow(overlay)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.axis('off')

    plt.suptitle(f'XAI Comparison — Predicted: {DR_CLASSES[pred_class]}',
                 fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight',
                    facecolor='white')
    return fig


def overlay_lesion_mask(gradcam_map, lesion_mask, threshold=0.4):
    """
    IDRiD-specific: overlay model attention vs ground-truth lesion mask.
    gradcam_map: (H,W) float [0,1]
    lesion_mask: (H,W) binary uint8
    Returns concordance score (IoU) between high-attention and lesion regions.
    """
    attention_binary = (gradcam_map > threshold).astype(np.uint8)
    intersection = np.logical_and(attention_binary, lesion_mask).sum()
    union        = np.logical_or(attention_binary, lesion_mask).sum()
    iou = intersection / (union + 1e-8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(gradcam_map, cmap='jet'); axes[0].set_title('Grad-CAM')
    axes[1].imshow(lesion_mask, cmap='Reds'); axes[1].set_title('Ground truth lesions')
    axes[2].imshow(attention_binary, cmap='Greens')
    axes[2].set_title(f'Attention mask (IoU={iou:.3f})')
    for ax in axes: ax.axis('off')
    plt.tight_layout()
    return iou, fig