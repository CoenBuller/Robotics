# diagnose.py
# Run this after training to understand WHY the model behaves the way it does.
# Produces four analyses:
#   1. Confusion matrix          — what gets confused with what
#   2. Grad-CAM heatmaps         — what part of the spectrogram the CNN looks at
#   3. t-SNE of embeddings       — whether classes cluster cleanly in feature space
#   4. Data auditor              — checks your actual audio files for common problems

import os
import numpy as np
import torch
import torch.nn.functional as F
import librosa as lb
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

# ── adjust these to match your setup ─────────────────────────────────────────
from SoundClassifier import AudioCNN
from AudioAugmentationPipelin import AudioAugmentationPipeline
from config import AugmentConfig
from TrainClassifier import AudioDataset

MODEL_PATH   = "final-project/models/best_model.pt"
DATA_FOLDER  = "data"
CLASS_NAMES  = ["clap", "harmonica", "silence", "whistle"]   # must match training order
SR           = 15_872
OUT_DIR      = "diagnostics"
# ─────────────────────────────────────────────────────────────────────────────

os.makedirs(OUT_DIR, exist_ok=True)


def load_model(path, n_classes):
    model = AudioCNN(n_classes=n_classes)
    model.load_state_dict(torch.load(path, map_location="cpu"))
    model.eval()
    return model


def build_dataset():
    cfg = AugmentConfig(
        noise_prob=0, pitch_shift_prob=0, time_shift_prob=0,
        volume_scale_prob=0, spec_augment_prob=0,
        n_freq_masks=0, freq_mask_param=0,
        n_time_masks=0, time_mask_param=0,
    )
    ap = AudioAugmentationPipeline(config=cfg, n_mels=62, hop=512, n_fft=512, sr=SR)

    files, labels = [], []
    for label, folder in enumerate(CLASS_NAMES):
        p = os.path.join(DATA_FOLDER, folder)
        for f in os.listdir(p):
            files.append(os.path.join(p, f))
            labels.append(label)

    # No augmentation — we want to see real performance
    return AudioDataset(files, labels, ap, augment=False), files, labels


# ── 1. CONFUSION MATRIX ───────────────────────────────────────────────────────
def plot_confusion_matrix(model, dataset):
    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    all_preds, all_labels = [], []

    with torch.no_grad():
        for X, y in loader:
            preds = model(X).argmax(dim=1)
            all_preds.extend(preds.numpy())
            all_labels.extend(y.numpy())

    cm = confusion_matrix(all_labels, all_preds)

    # Normalise rows so each cell shows recall per class
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, data, title, fmt in zip(
        axes,
        [cm, cm_norm],
        ["Confusion matrix (counts)", "Confusion matrix (row-normalised recall)"],
        ["d", ".2f"],
    ):
        im = ax.imshow(data, cmap="Blues")
        ax.set_xticks(range(len(CLASS_NAMES))); ax.set_xticklabels(CLASS_NAMES, rotation=45, ha="right")
        ax.set_yticks(range(len(CLASS_NAMES))); ax.set_yticklabels(CLASS_NAMES)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True")
        ax.set_title(title)
        for i in range(len(CLASS_NAMES)):
            for j in range(len(CLASS_NAMES)):
                ax.text(j, i, format(data[i, j], fmt),
                        ha="center", va="center",
                        color="white" if data[i, j] > data.max() * 0.6 else "black")
        plt.colorbar(im, ax=ax)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "confusion_matrix.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[1] Saved: {path}")

    # Also print the text report — easy to read in terminal
    print("\nClassification report:")
    print(classification_report(all_labels, all_preds, target_names=CLASS_NAMES))
    return all_preds, all_labels


# ── 2. GRAD-CAM ───────────────────────────────────────────────────────────────
# Shows WHICH part of the spectrogram the CNN looks at when making a decision.
# If harmonic always lights up the same narrow band regardless of the sample,
# the model has learnt a spurious shortcut (e.g. recording noise, DC offset).

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = None
        self.activations = None
        target_layer.register_forward_hook(self._save_activation)
        target_layer.register_full_backward_hook(self._save_gradient)

    def _save_activation(self, _, __, output):
        self.activations = output.detach()

    def _save_gradient(self, _, __, grad_output):
        self.gradients = grad_output[0].detach()

    def __call__(self, x, class_idx):
        self.model.zero_grad()
        logits = self.model(x)
        logits[0, class_idx].backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)   # global avg pool over spatial dims
        cam = (weights * self.activations).sum(dim=1).squeeze()
        cam = F.relu(cam)
        cam -= cam.min(); cam /= (cam.max() + 1e-8)
        return cam.numpy()


def plot_gradcam(model, dataset, files, n_per_class=4):
    """
    For each class, show n_per_class examples with their spectrogram
    and Grad-CAM overlay. Misclassified samples are marked with a red border.
    """
    # Find the last Conv layer automatically
    last_conv = None
    for m in model.modules():
        if isinstance(m, torch.nn.Conv2d):
            last_conv = m
    if last_conv is None:
        print("[2] No Conv2d layer found — skipping Grad-CAM")
        return

    gcam = GradCAM(model, last_conv)

    fig_rows = len(CLASS_NAMES)
    fig_cols = n_per_class
    fig, axes = plt.subplots(fig_rows, fig_cols, figsize=(fig_cols * 3, fig_rows * 3))
    fig.suptitle("Grad-CAM: what the CNN focuses on per class", fontsize=13)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        # Collect indices for this class
        cls_indices = [i for i, (_, label) in enumerate(dataset) if label == cls_idx][:n_per_class]

        for col, idx in enumerate(cls_indices):
            x, label = dataset[idx]
            x_in = x.unsqueeze(0).requires_grad_(True)

            model.zero_grad()
            logits = model(x_in)
            pred = logits.argmax(dim=1).item()
            cam = gcam(x_in, pred)

            # Take first channel of the mel spectrogram for display
            spec = x[0].numpy()

            ax = axes[cls_idx][col]
            ax.imshow(spec, origin="lower", aspect="auto", cmap="magma")
            ax.imshow(
                cam,
                origin="lower", aspect="auto",
                cmap="jet", alpha=0.45,
                extent=[0, spec.shape[1], 0, spec.shape[0]],
            )
            title = f"true={cls_name}\npred={CLASS_NAMES[pred]}"
            ax.set_title(title, fontsize=8,
                         color="red" if pred != label else "black")
            ax.axis("off")

            # Red border on misclassified
            if pred != label:
                for spine in ax.spines.values():
                    spine.set_edgecolor("red"); spine.set_linewidth(3)

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "gradcam.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[2] Saved: {path}")


# ── 3. t-SNE OF EMBEDDINGS ────────────────────────────────────────────────────
# Extracts the penultimate layer (just before the classifier head) and plots
# all samples in 2D. If harmonic forms a very tight cluster, the model has
# memorised those specific recordings rather than learning the class in general.
# If harmonic overlaps with another class, those two are hard to separate.

def plot_tsne(model, dataset):
    # Hook the layer just before the final linear classifier
    embeddings, labels_list = [], []
    hooks = []

    def hook_fn(_, __, output):
        embeddings.append(output.detach().cpu().numpy())

    # Attach to the last linear layer's INPUT by hooking the second-to-last module
    modules = list(model.named_modules())
    linear_layers = [(name, m) for name, m in modules if isinstance(m, torch.nn.Linear)]
    if len(linear_layers) < 1:
        print("[3] No Linear layer found — skipping t-SNE")
        return

    # Hook the last linear layer's input via a pre-hook
    last_linear_name, last_linear = linear_layers[-1]
    hooks.append(last_linear.register_forward_pre_hook(
        lambda _, inp: embeddings.append(inp[0].detach().cpu().numpy())
    ))

    loader = DataLoader(dataset, batch_size=32, shuffle=False)
    with torch.no_grad():
        for X, y in loader:
            model(X)
            labels_list.extend(y.numpy())

    for h in hooks:
        h.remove()

    emb = np.concatenate(embeddings, axis=0)
    labels_arr = np.array(labels_list)

    perplexity = min(30, len(emb) - 1)
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42, max_iter=1000)
    coords = tsne.fit_transform(emb)

    colors = plt.cm.tab10(np.linspace(0, 0.4, len(CLASS_NAMES)))
    fig, ax = plt.subplots(figsize=(8, 6))
    for cls_idx, (cls_name, color) in enumerate(zip(CLASS_NAMES, colors)):
        mask = labels_arr == cls_idx
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   label=cls_name, color=color, alpha=0.7, s=40)

    ax.set_title("t-SNE of penultimate-layer embeddings")
    ax.legend()
    ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")

    plt.tight_layout()
    path = os.path.join(OUT_DIR, "tsne.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"[3] Saved: {path}")


# ── 4. DATA AUDITOR ───────────────────────────────────────────────────────────
# Checks for common problems in the raw audio files that cause overfitting:
#   - All samples too similar in duration (not enough natural variation)
#   - RMS too uniform (all recorded at same distance/volume)
#   - DC offset (microphone bias baked into every recording)
#   - Clipping (samples are saturated)
#   - Spectral centroid variance (low = all samples sound identical)

def audit_data(files, labels):
    print("\n[4] Data audit:")
    print(f"{'Class':<15} {'N':>5} {'dur_std':>9} {'rms_mean':>10} {'rms_std':>9} "
          f"{'dc_offset':>10} {'clipped%':>10} {'centroid_std':>14}")
    print("-" * 85)

    for cls_idx, cls_name in enumerate(CLASS_NAMES):
        cls_files = [f for f, l in zip(files, labels) if l == cls_idx]
        durations, rms_vals, dc_offsets, clipped, centroids = [], [], [], [], []

        for fp in cls_files:
            y, sr = lb.load(fp, sr=SR, duration=2.0)
            durations.append(len(y) / sr)
            rms_vals.append(np.sqrt(np.mean(y**2)))
            dc_offsets.append(np.abs(np.mean(y)))
            clipped.append(np.mean(np.abs(y) > 0.98))
            cent = lb.feature.spectral_centroid(y=y, sr=sr)
            centroids.append(np.mean(cent))

        print(
            f"{cls_name:<15}"
            f"{len(cls_files):>5}"
            f"{np.std(durations):>9.3f}s"
            f"{np.mean(rms_vals):>10.4f}"
            f"{np.std(rms_vals):>9.4f}"
            f"{np.mean(dc_offsets):>10.5f}"
            f"{np.mean(clipped)*100:>9.1f}%"
            f"{np.std(centroids):>14.1f}Hz"
        )

    print("\nWhat to look for:")
    print("  dur_std      ≈ 0     → all samples same length, no natural variation")
    print("  rms_std      ≈ 0     → all recorded at identical volume (too uniform)")
    print("  dc_offset    > 0.01  → microphone bias present; normalise your audio")
    print("  clipped%     > 0     → audio is saturating; record at lower gain")
    print("  centroid_std ≈ 0     → samples are spectrally identical (memorisation risk)")


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("Loading model and data...")
    model = load_model(MODEL_PATH, n_classes=len(CLASS_NAMES))
    dataset, files, labels = build_dataset()

    print("\n── 1. Confusion matrix ──────────────────────────────────")
    preds, true = plot_confusion_matrix(model, dataset)

    print("\n── 2. Grad-CAM ──────────────────────────────────────────")
    plot_gradcam(model, dataset, files, n_per_class=4)

    print("\n── 3. t-SNE embeddings ──────────────────────────────────")
    plot_tsne(model, dataset)

    print("\n── 4. Data audit ────────────────────────────────────────")
    audit_data(files, labels)

    print(f"\nAll outputs saved to: {OUT_DIR}/")