# gradcam.py
import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt


class GradCAM:
    def __init__(self, model, target_layer):
        """
        target_layer: either a direct nn.Module reference (CustomCNN)
                      or a string layer name (ResNet18, e.g. 'layer4')
        """
        self.model = model

        # Resolve string → module
        if isinstance(target_layer, str):
            self.target_layer = dict(model.named_modules())[target_layer]
        else:
            self.target_layer = target_layer

        self.gradients = None
        self.activations = None
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.target_layer.register_forward_hook(forward_hook)
        self.target_layer.register_full_backward_hook(backward_hook)

    def generate(self, input_tensor, target_class=None):
        self.model.eval()
        output = self.model(input_tensor)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

        self.model.zero_grad()
        output[0, target_class].backward()

        weights = self.gradients.mean(dim=[2, 3], keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam, target_class


# ── Layer resolution helpers ───────────────────────────────────────────────

def get_target_layer(model):
    """
    Automatically pick the right target layer based on model type.
    Returns a (layer, label) tuple.
    """
    named = dict(model.named_modules())

    # ResNet18: always use layer4 (last residual block before avgpool)
    if "layer4" in named:
        return named["layer4"], "layer4"

    # CustomCNN: walk modules and grab the last Conv2d
    last_conv = None
    for module in model.modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
    if last_conv:
        return last_conv, "last_conv"

    raise ValueError("Could not automatically determine target layer.")


# ── Shared visualization ───────────────────────────────────────────────────

def _gradcam_for_sample(gradcam, dataset, idx, le, device, target_class=None):
    """Run Grad-CAM on a single dataset sample. Returns (img_uint8, cam_resized, overlay, true_chord, pred_chord)."""
    spectrogram, label_idx = dataset[idx]

    input_tensor = spectrogram.unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    cam, predicted_idx = gradcam.generate(input_tensor, target_class=target_class)

    true_chord = le.classes_[label_idx]
    pred_chord = le.classes_[predicted_idx]

    img = spectrogram.cpu().numpy()
    if img.shape[0] == 1:
        img = np.repeat(img, 3, axis=0)
    img = np.transpose(img, (1, 2, 0))
    img = (img - img.min()) / (img.max() - img.min() + 1e-8)
    img_uint8 = (img * 255).astype(np.uint8)

    cam_resized = cv2.resize(cam, (img_uint8.shape[1], img_uint8.shape[0]))
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    overlay = (0.5 * heatmap + 0.5 * img_uint8).astype(np.uint8)

    return img_uint8, cam_resized, overlay, true_chord, pred_chord


def visualize_gradcam(model, dataset, le, device, num_samples=6,
                      target_class=None, model_name="Model"):
    """Visualize Grad-CAM for a single model — works for both CustomCNN and ResNet18."""
    layer, layer_label = get_target_layer(model)
    gradcam = GradCAM(model, layer)

    fig, axes = plt.subplots(num_samples, 3, figsize=(12, num_samples * 3.5))
    fig.suptitle(f"Grad-CAM — {model_name} (target: {layer_label})",
                 fontsize=13, fontweight='bold')

    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    for row, idx in enumerate(indices):
        img, cam, overlay, true_chord, pred_chord = _gradcam_for_sample(
            gradcam, dataset, idx, le, device, target_class
        )
        correct = "✓" if true_chord == pred_chord else "✗"
        color = "green" if correct == "✓" else "red"

        axes[row, 0].imshow(img)
        axes[row, 0].set_title(f"True: {true_chord}", fontsize=9)
        axes[row, 1].imshow(cam, cmap='jet')
        axes[row, 1].set_title("Grad-CAM Heatmap", fontsize=9)
        axes[row, 2].imshow(overlay)
        axes[row, 2].set_title(f"Pred: {pred_chord} {correct}", fontsize=9, color=color)

        for ax in axes[row]:
            ax.axis('off')

    plt.tight_layout()
    plt.show()


# ── Side-by-side comparison ────────────────────────────────────────────────

def compare_gradcam(custom_cnn, resnet18, dataset, le, device,
                    num_samples=4, target_class=None):
    """
    Plot CustomCNN vs ResNet18 Grad-CAM side by side on the same samples.
    Columns: Spectrogram | CustomCNN overlay | ResNet18 overlay
    """
    cnn_layer, cnn_label   = get_target_layer(custom_cnn)
    res_layer, res_label   = get_target_layer(resnet18)
    gradcam_cnn = GradCAM(custom_cnn, cnn_layer)
    gradcam_res = GradCAM(resnet18,   res_layer)

    fig, axes = plt.subplots(num_samples, 3, figsize=(13, num_samples * 3.5))
    col_titles = ["Input Spectrogram",
                  f"CustomCNN ({cnn_label})",
                  f"ResNet18 ({res_label})"]
    for col, title in enumerate(col_titles):
        axes[0, col].set_title(title, fontsize=10, fontweight='bold', pad=8)

    indices = torch.randperm(len(dataset))[:num_samples].tolist()

    for row, idx in enumerate(indices):
        img, _, cnn_overlay, true_chord, cnn_pred = _gradcam_for_sample(
            gradcam_cnn, dataset, idx, le, device, target_class
        )
        _,   _, res_overlay, _,          res_pred = _gradcam_for_sample(
            gradcam_res, dataset, idx, le, device, target_class
        )

        cnn_correct = "✓" if cnn_pred == true_chord else "✗"
        res_correct = "✓" if res_pred == true_chord else "✗"

        axes[row, 0].imshow(img)
        axes[row, 0].set_ylabel(f"True: {true_chord}", fontsize=9, labelpad=4)

        axes[row, 1].imshow(cnn_overlay)
        axes[row, 1].set_title(f"{cnn_pred} {cnn_correct}", fontsize=8,
                               color="green" if cnn_correct == "✓" else "red")

        axes[row, 2].imshow(res_overlay)
        axes[row, 2].set_title(f"{res_pred} {res_correct}", fontsize=8,
                               color="green" if res_correct == "✓" else "red")

        for ax in axes[row]:
            ax.axis('off')

    plt.suptitle("Grad-CAM Comparison: CustomCNN vs ResNet18",
                 fontsize=13, fontweight='bold', y=1.01)
    plt.tight_layout()
    plt.show()