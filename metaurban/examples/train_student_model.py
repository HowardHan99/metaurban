
import argparse
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import List
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms


LABEL_NAMES = ["NEGATIVE_SOCIAL", "NEUTRAL", "POSITIVE_SOCIAL"]
EGO_DIM = 9


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class MultiModalSample:
    image_path: str
    ego_state: np.ndarray
    action: np.ndarray
    label: int
    stem: str


class SocialImageEgoActionDataset(Dataset):
    def __init__(self, samples: List[MultiModalSample], image_height: int, image_width: int):
        self.samples = samples
        self.transform = transforms.Compose([
            transforms.Resize((image_height, image_width)),  # 1024x576 -> 128x72 keeps 16:9
            transforms.ToTensor(),
        ])

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = Image.open(s.image_path).convert("RGB")
        img = self.transform(img)

        ego = torch.tensor(s.ego_state, dtype=torch.float32)
        action = torch.tensor(s.action, dtype=torch.float32)
        label = torch.tensor(s.label, dtype=torch.long)
        return img, ego, action, label


class FusionStudentNet(nn.Module):
    def __init__(
        self,
        ego_dim: int = 9,
        action_dim: int = 2,
        num_classes: int = 3,
        dropout: float = 0.2,
        pretrained_backbone: bool = False,
        freeze_backbone: bool = False,
    ):
        super().__init__()

        weights = models.ResNet18_Weights.DEFAULT if pretrained_backbone else None
        backbone = models.resnet18(weights=weights)
        image_feat_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.image_encoder = backbone

        if freeze_backbone:
            for p in self.image_encoder.parameters():
                p.requires_grad = False

        self.state_action_encoder = nn.Sequential(
            nn.Linear(ego_dim + action_dim, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
        )

        self.classifier = nn.Sequential(
            nn.Linear(image_feat_dim + 64, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes),
        )

    def forward(self, image, ego_state, action):
        img_feat = self.image_encoder(image)
        state_action = torch.cat([ego_state, action], dim=1)
        sa_feat = self.state_action_encoder(state_action)
        fused = torch.cat([img_feat, sa_feat], dim=1)
        logits = self.classifier(fused)
        return logits


def parse_args():
    p = argparse.ArgumentParser(description="Train student model: image + ego_state + action -> label")
    p.add_argument("--image-dir", type=str, default="./recorded_dataset/final_rgb_merged",
                   help="Directory containing step_XXXXXX.png")
    p.add_argument("--label-dir", type=str, default="./recorded_dataset/new_labels/final_merged_npy",
                   help="Directory containing labeled step_XXXXXX.npy with keys: state, action, label")
    p.add_argument("--out-dir", type=str, default=None)
    p.add_argument("--image-width", type=int, default=128)
    p.add_argument("--image-height", type=int, default=72)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--dropout", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--val-size", type=float, default=0.1)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--early-stop-patience", type=int, default=10)
    p.add_argument("--pretrained-backbone", action="store_true", default=True)
    p.add_argument("--freeze-backbone", action="store_true", default=False)
    return p.parse_args()


def find_samples(image_dir: Path, label_dir: Path) -> List[MultiModalSample]:
    label_files = sorted(label_dir.glob("step_*.npy"))
    if not label_files:
        raise RuntimeError(f"No labeled npy files found in {label_dir}")

    samples: List[MultiModalSample] = []
    for npy_path in label_files:
        stem = npy_path.stem
        img_path = image_dir / f"{stem}.png"
        if not img_path.exists():
            continue

        obj = np.load(npy_path, allow_pickle=True).item()

        if "state" not in obj or "action" not in obj or "label" not in obj:
            raise KeyError(f"{npy_path} must contain keys: state, action, label")

        state = np.asarray(obj["state"], dtype=np.float32).reshape(-1)
        if state.shape[0] < EGO_DIM:
            raise ValueError(f"{npy_path} state dim {state.shape[0]} is smaller than ego dim {EGO_DIM}")
        ego_state = state[:EGO_DIM].copy()

        action = np.asarray(obj["action"], dtype=np.float32).reshape(-1)
        if action.shape[0] < 2:
            raise ValueError(f"{npy_path} action dim {action.shape[0]} is smaller than expected 2")
        action = action[:2].copy()

        label = int(obj["label"])
        if label not in [0, 1, 2]:
            raise ValueError(f"{npy_path} has invalid label {label}, expected 0/1/2")

        samples.append(
            MultiModalSample(
                image_path=str(img_path),
                ego_state=ego_state,
                action=action,
                label=label,
                stem=stem,
            )
        )

    if not samples:
        raise RuntimeError("No matched (image, labeled npy) pairs found.")
    return samples


def split_samples(samples: List[MultiModalSample], test_size: float, val_size: float, seed: int):
    labels = np.array([s.label for s in samples])
    indices = np.arange(len(samples))

    train_val_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=labels,
    )

    train_val_labels = labels[train_val_idx]
    val_ratio_within_trainval = val_size / (1.0 - test_size)

    train_idx, val_idx = train_test_split(
        train_val_idx,
        test_size=val_ratio_within_trainval,
        random_state=seed,
        stratify=train_val_labels,
    )

    train_samples = [samples[i] for i in train_idx]
    val_samples = [samples[i] for i in val_idx]
    test_samples = [samples[i] for i in test_idx]
    return train_samples, val_samples, test_samples


def compute_class_weights(samples: List[MultiModalSample], num_classes: int = 3) -> torch.Tensor:
    labels = np.array([s.label for s in samples], dtype=np.int64)
    counts = np.bincount(labels, minlength=num_classes).astype(np.float32)
    weights = counts.sum() / np.maximum(counts, 1.0)
    weights = weights / weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def make_loader(samples: List[MultiModalSample], image_height: int, image_width: int, batch_size: int, shuffle: bool, num_workers: int):
    ds = SocialImageEgoActionDataset(samples, image_height=image_height, image_width=image_width)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


def run_epoch(model, loader, criterion, optimizer, device):
    is_train = optimizer is not None
    model.train(is_train)

    total_loss = 0.0
    all_preds = []
    all_targets = []

    for image, ego, action, label in loader:
        image = image.to(device)
        ego = ego.to(device)
        action = action.to(device)
        label = label.to(device)

        logits = model(image, ego, action)
        loss = criterion(logits, label)

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        total_loss += loss.item() * image.size(0)
        preds = logits.argmax(dim=1)
        all_preds.append(preds.detach().cpu().numpy())
        all_targets.append(label.detach().cpu().numpy())

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_targets, all_preds)
    macro_f1 = f1_score(all_targets, all_preds, average="macro")
    return avg_loss, acc, macro_f1


@torch.no_grad()
def evaluate_with_details(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []

    for image, ego, action, label in loader:
        image = image.to(device)
        ego = ego.to(device)
        action = action.to(device)

        logits = model(image, ego, action)
        preds = logits.argmax(dim=1).cpu().numpy()

        all_preds.append(preds)
        all_targets.append(label.numpy())

    y_pred = np.concatenate(all_preds)
    y_true = np.concatenate(all_targets)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_true,
        y_pred,
        target_names=LABEL_NAMES,
        digits=4,
        zero_division=0,
    )
    return acc, macro_f1, cm, report_dict, report_text


def plot_curve(train_values, val_values, ylabel, title, save_path: Path):
    plt.figure(figsize=(7, 5))
    epochs = np.arange(1, len(train_values) + 1)
    plt.plot(epochs, train_values, label="Train")
    plt.plot(epochs, val_values, label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def plot_confusion_matrix(cm: np.ndarray, save_path: Path):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest")
    plt.title("Test Confusion Matrix")
    plt.colorbar()
    tick_marks = np.arange(len(LABEL_NAMES))
    plt.xticks(tick_marks, LABEL_NAMES, rotation=20)
    plt.yticks(tick_marks, LABEL_NAMES)

    thresh = cm.max() / 2.0 if cm.max() > 0 else 0.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(save_path, dpi=200)
    plt.close()


def save_json(data, path: Path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main():
    args = parse_args()
    set_seed(args.seed)

    image_dir = Path(args.image_dir)
    label_dir = Path(args.label_dir)
    if args.out_dir is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_dir = Path(f"./recorded_dataset/student_runs/{timestamp}")
    else:
        out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = find_samples(image_dir=image_dir, label_dir=label_dir)
    train_samples, val_samples, test_samples = split_samples(
        samples=samples,
        test_size=args.test_size,
        val_size=args.val_size,
        seed=args.seed,
    )

    train_loader = make_loader(train_samples, args.image_height, args.image_width, args.batch_size, True, args.num_workers)
    val_loader = make_loader(val_samples, args.image_height, args.image_width, args.batch_size, False, args.num_workers)
    test_loader = make_loader(test_samples, args.image_height, args.image_width, args.batch_size, False, args.num_workers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FusionStudentNet(
        ego_dim=EGO_DIM,
        action_dim=2,
        num_classes=3,
        dropout=args.dropout,
        pretrained_backbone=args.pretrained_backbone,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    class_weights = compute_class_weights(train_samples).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "train_macro_f1": [],
        "val_macro_f1": [],
    }

    best_val_f1 = -1.0
    best_epoch = -1
    patience_counter = 0
    best_model_path = out_dir / "best_student_image_ego_action.pt"

    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc, train_f1 = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_f1 = run_epoch(model, val_loader, criterion, None, device)

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["train_macro_f1"].append(train_f1)
        history["val_macro_f1"].append(val_f1)

        print(
            f"Epoch {epoch:03d} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} train_f1={train_f1:.4f} | "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} val_f1={val_f1:.4f}"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            patience_counter = 0
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "ego_dim": EGO_DIM,
                    "action_dim": 2,
                    "num_classes": 3,
                    "dropout": args.dropout,
                    "image_width": args.image_width,
                    "image_height": args.image_height,
                    "pretrained_backbone": args.pretrained_backbone,
                    "freeze_backbone": args.freeze_backbone,
                    "label_names": LABEL_NAMES,
                },
                best_model_path,
            )
        else:
            patience_counter += 1

        if patience_counter >= args.early_stop_patience:
            print(f"Early stopping at epoch {epoch}. Best epoch = {best_epoch}, best val macro-F1 = {best_val_f1:.4f}")
            break

    checkpoint = torch.load(best_model_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_acc, test_f1, cm, report_dict, report_text = evaluate_with_details(model, test_loader, device)

    plot_curve(history["train_loss"], history["val_loss"], "Loss",
               "Training / Validation Loss", out_dir / "loss_curve.png")
    plot_curve(history["train_acc"], history["val_acc"], "Accuracy",
               "Training / Validation Accuracy", out_dir / "accuracy_curve.png")
    plot_curve(history["train_macro_f1"], history["val_macro_f1"], "Macro-F1",
               "Training / Validation Macro-F1", out_dir / "macro_f1_curve.png")
    plot_confusion_matrix(cm, out_dir / "test_confusion_matrix.png")

    summary = {
        "image_dir": str(image_dir),
        "label_dir": str(label_dir),
        "num_total": len(samples),
        "num_train": len(train_samples),
        "num_val": len(val_samples),
        "num_test": len(test_samples),
        "best_epoch": best_epoch,
        "best_val_macro_f1": best_val_f1,
        "test_accuracy": test_acc,
        "test_macro_f1": test_f1,
        "image_width": args.image_width,
        "image_height": args.image_height,
        "pretrained_backbone": args.pretrained_backbone,
        "freeze_backbone": args.freeze_backbone,
        "label_names": LABEL_NAMES,
    }
    save_json(summary, out_dir / "run_summary.json")
    save_json(report_dict, out_dir / "test_classification_report.json")

    with open(out_dir / "test_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report_text)

    print("\nFinished.")
    print(f"Best checkpoint:  {best_model_path}")
    print(f"Loss curve:       {out_dir / 'loss_curve.png'}")
    print(f"Accuracy curve:   {out_dir / 'accuracy_curve.png'}")
    print(f"Macro-F1 curve:   {out_dir / 'macro_f1_curve.png'}")
    print(f"Confusion matrix: {out_dir / 'test_confusion_matrix.png'}")
    print(f"Test accuracy:    {test_acc:.4f}")
    print(f"Test macro-F1:    {test_f1:.4f}")


if __name__ == "__main__":
    main()
