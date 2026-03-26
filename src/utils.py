# src/utils.py
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np


def create_mock_model(device='cpu'):
    from src.model import build_model
    os.makedirs('models', exist_ok=True)
    model = build_model(num_classes=5, device=device)
    model.eval()
    torch.save(
        {'model_state_dict': model.state_dict(), 'val_acc': 0.0, 'epoch': 0},
        'models/best_model.pth'
    )
    print("Mock model saved to models/best_model.pth")
    return model


def verify_setup():
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    from src.gradcam import generate_gradcam, analyze_heatmap_regions

    device = 'cpu'
    model  = create_mock_model(device)

    # Fake a 224x224 RGB image
    dummy_np  = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    transform = A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])
    tensor = transform(image=dummy_np)['image']

    overlay, cam, pred, probs = generate_gradcam(
        model, tensor, device=device)
    regions = analyze_heatmap_regions(cam)

    print(f"\nSmoke test passed!")
    print(f"  Predicted class : {pred}")
    print(f"  Probabilities   : {np.round(probs, 3)}")
    print(f"  Active regions  : {regions}")
    print(f"  Overlay shape   : {overlay.shape}")
    print(f"\nRun: streamlit run app/app.py")


if __name__ == '__main__':
    verify_setup()