import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# this file was AI assisted with Claude

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, class_idx=None):
        """
        Args:
            input_tensor: (1, 1, H, W) spectrogram tensor on the correct device
            class_idx: target class index; if None, uses the predicted class
        Returns:
            cam: numpy array (H, W), values in [0, 1]
            pred_class: predicted class index
        """
        self.model.eval()
        input_tensor = input_tensor.requires_grad_(True)

        # forward
        logits = self.model(input_tensor)           # (1, num_classes)
        pred_class = logits.argmax(dim=1).item()

        if class_idx is None:
            class_idx = pred_class

        # backward on the target class score
        self.model.zero_grad()
        score = logits[0, class_idx]
        score.backward()

        # alpha = global average pooled gradients
        alpha = self.gradients.mean(dim=(2, 3), keepdim=True)  # (1, 64, 1, 1)

        # weighted combination of activations
        cam = (alpha * self.activations).sum(dim=1, keepdim=True)  # (1, 1, H', W')
        cam = F.relu(cam)

        # normalise and resize to input size
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)
        cam = F.interpolate(
            cam,
            size=input_tensor.shape[2:],  # (H, W)
            mode="bilinear",
            align_corners=False,
        )
        return cam.squeeze().cpu().numpy(), pred_class
        


def visualize_gradcam(gradcam, dataset, le, idx, device, target_size=(128, 128)):
    spec, label = dataset[idx]
    input_tensor = to_gradcam_input(spec, target_size=target_size).to(device)
    cam, pred_idx = gradcam.generate(input_tensor)

    # RGB spec for display: (3,H,W) → (H,W,3), normalized to [0,1]
    if spec.shape[0] == 3:
        spec_display = spec.permute(1, 2, 0).cpu().numpy()
        spec_display = (spec_display - spec_display.min()) / (spec_display.max() - spec_display.min() + 1e-8)
    else:
        spec_display = spec.squeeze().cpu().numpy()

    # resize CAM to match spec_display spatial dims
    cam_resized = F.interpolate(
        torch.tensor(cam).unsqueeze(0).unsqueeze(0),
        size=(spec_display.shape[0], spec_display.shape[1]),
        mode="bilinear",
        align_corners=False
    ).squeeze().numpy()

    true_chord = le.classes_[label]
    pred_chord = le.classes_[pred_idx]
    correct    = "✓" if pred_idx == label else "✗"

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    axes[0].imshow(spec_display, origin="lower", aspect="auto")
    axes[0].set_title(f"Mel Spectrogram\nTrue: {true_chord}")
    axes[0].set_xlabel("Time")
    axes[0].set_ylabel("Mel Frequency")

    axes[1].imshow(cam_resized, origin="lower", aspect="auto", cmap="jet",
                   vmin=0, vmax=1, interpolation="nearest")
    axes[1].set_title(f"Grad-CAM\nPred: {pred_chord} {correct}")
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Mel Frequency")

    # Overlay using resized CAM
    cam_colored = plt.cm.jet(cam_resized)[..., :3]        # (H,W,3)
    overlay = 0.55 * spec_display + 0.45 * cam_colored
    overlay = np.clip(overlay, 0, 1)

    axes[2].imshow(overlay, origin="lower", aspect="auto")
    axes[2].set_title(f"Overlay — True: {true_chord} | Pred: {pred_chord} {correct}")
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Mel Frequency")

    plt.tight_layout()
    plt.show()

def visualize_gradcam_grid(gradcam, dataset, le, device, indices, target_size=(128, 128)):
    n = len(indices)
    fig, axes = plt.subplots(n, 3, figsize=(14, 4 * n), squeeze=False)

    for row, idx in enumerate(indices):
        spec, label = dataset[idx]
        input_tensor = to_gradcam_input(spec, target_size=target_size).to(device)
        cam, pred_idx = gradcam.generate(input_tensor)

        # Prepare spec for display
        if spec.shape[0] == 3:
            spec_display = spec.permute(1, 2, 0).cpu().numpy()
            spec_display = (spec_display - spec_display.min()) / (spec_display.max() - spec_display.min() + 1e-8)
        else:
            spec_display = spec.squeeze().cpu().numpy()

        true_chord = le.classes_[label]
        pred_chord = le.classes_[pred_idx]
        correct    = "✓" if pred_idx == label else "✗"

        # Resize CAM to match spec using torch
        cam_tensor = torch.tensor(cam).unsqueeze(0).unsqueeze(0)  # (1,1,H,W)
        cam_resized = F.interpolate(
            cam_tensor,
            size=(spec_display.shape[0], spec_display.shape[1]),
            mode="bilinear",
            align_corners=False
        ).squeeze().numpy()

        cam_colored = plt.cm.jet(cam_resized)[..., :3]
        overlay     = np.clip(0.55 * spec_display + 0.45 * cam_colored, 0, 1)

        axes[row][0].imshow(spec_display, origin="lower", aspect="auto")
        axes[row][0].set_title(f"Mel Spectrogram\nTrue: {true_chord}")
        axes[row][0].set_xlabel("Time")
        axes[row][0].set_ylabel("Mel Frequency")

        axes[row][1].imshow(cam_resized, origin="lower", aspect="auto", cmap="jet", vmin=0, vmax=1)
        axes[row][1].set_title(f"Grad-CAM\nPred: {pred_chord} {correct}")
        axes[row][1].set_xlabel("Time")
        axes[row][1].set_ylabel("Mel Frequency")

        axes[row][2].imshow(overlay, origin="lower", aspect="auto")
        axes[row][2].set_title(f"Overlay — True: {true_chord} | Pred: {pred_chord} {correct}")
        axes[row][2].set_xlabel("Time")
        axes[row][2].set_ylabel("Mel Frequency")

    plt.tight_layout()
    plt.show()

def to_gradcam_input(spec):
    """Convert (1, 3, H, W) RGB tensor to (1, 1, H, W) grayscale for Grad-CAM."""
    if spec.ndim == 3 and spec.shape[0] == 3:
        spec = spec.mean(dim=0, keepdim=True)  # (3,H,W) → (1,H,W)
    return spec.unsqueeze(0)                   # (1,H,W) → (1,1,H,W)

def to_gradcam_input(spec, target_size=(128, 128)):
    """Convert RGB (3,H,W) tensor to grayscale (1,1,H,W) resized to match model input."""
    if spec.ndim == 3 and spec.shape[0] == 3:
        spec = spec.mean(dim=0, keepdim=True)  # (3,H,W) → (1,H,W)
    elif spec.ndim == 2:
        spec = spec.unsqueeze(0)               # (H,W) → (1,H,W)
    
    spec = spec.unsqueeze(0)                   # (1,H,W) → (1,1,H,W)
    spec = F.interpolate(spec, size=target_size, mode="bilinear", align_corners=False)
    return spec