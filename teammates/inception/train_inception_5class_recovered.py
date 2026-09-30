import os
import gc
import pickle
import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, f1_score

# ============================================================
# CONFIG
# ============================================================

DATA_DIR = r"C:\Users\anany\Documents\major project\processed"

BATCH_SIZE = 32
EPOCHS = 20
LR = 1e-4

# FIX: Classes with fewer than this many samples are dropped entirely.
# Rationale: with 1-5 samples, a class cannot appear in train/val/test
# splits in a statistically meaningful way, and the model has no chance
# to learn a real pattern from so few examples. Keeping them only
# destabilizes the loss (see class_weight blow-up) without adding any
# learnable signal. This is a standard, explainable step for imbalanced
# medical classification baselines.
MIN_SAMPLES_PER_CLASS = 30

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

# ============================================================
# DATASET
# ============================================================

class ECGDataset(Dataset):

    def __init__(self, X, y, train=False):
        self.X = X
        self.y = y
        self.train = train

    def __len__(self):
        return len(self.X)

    def augment(self, signal):
        # Gaussian Noise
        if np.random.rand() < 0.15:
            signal = signal + np.random.normal(
                0, 0.002, signal.shape
            ).astype(np.float32)

        # Amplitude Scaling
        if np.random.rand() < 0.15:
            signal = signal * np.random.uniform(0.98, 1.02)

        return signal.astype(np.float32)

    def __getitem__(self, idx):
        # Data is already per-lead z-score normalized in preprocessing.
        # Do NOT re-normalize here with a single scalar mean/std across
        # all leads -- that destroys per-lead scaling.
        signal = np.asarray(self.X[idx], dtype=np.float32)

        if self.train:
            signal = self.augment(signal)

        return (
            torch.from_numpy(signal),
            torch.tensor(self.y[idx], dtype=torch.long)
        )


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("\nLoading Processed Dataset...")

    X = np.load(os.path.join(DATA_DIR, "X.npy"), mmap_mode="r")
    y = np.load(os.path.join(DATA_DIR, "y.npy"))

    with open(os.path.join(DATA_DIR, "label_encoder.pkl"), "rb") as f:
        old_encoder = pickle.load(f)

    print("X Shape :", X.shape)
    print("y Shape :", y.shape)
    print("Classes :", len(old_encoder.classes_))

    # FIX: filter by MIN_SAMPLES_PER_CLASS instead of the old ">= 2" cutoff.
    unique, counts = np.unique(y, return_counts=True)
    valid = unique[counts >= MIN_SAMPLES_PER_CLASS]
    mask = np.isin(y, valid)

    print(f"\nDropping {len(unique) - len(valid)} classes with "
          f"< {MIN_SAMPLES_PER_CLASS} samples")
    print(f"Classes kept: {len(valid)} / {len(unique)}")

    X = np.asarray(X[mask], dtype=np.float32)
    y = y[mask]

    encoder = LabelEncoder()
    y = encoder.fit_transform(y)

    print("\nRemaining Classes :", len(np.unique(y)))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.10,
        random_state=42,
        stratify=y_train
    )

    print("\nClass Distribution (Training Set):")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"Class {u}: {c} samples")

    del X
    del y
    gc.collect()

    class_weights_np = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train
    )
    class_weights = torch.tensor(class_weights_np, dtype=torch.float32, device=DEVICE)

    # FIX: WeightedRandomSampler for the train loader only.
    # class_weights_np already gives a per-CLASS weight; expand it to a
    # per-SAMPLE weight array so rare classes get drawn more often during
    # training. This complements (does not replace) the loss weighting --
    # the loss weighting corrects the gradient magnitude per class, while
    # the sampler corrects how often each class is even seen per batch.
    sample_weights = class_weights_np[y_train]

    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )

    train_loader = DataLoader(
        ECGDataset(X_train, y_train, train=True),
        batch_size=BATCH_SIZE,
        sampler=sampler,          # replaces shuffle=True
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    val_loader = DataLoader(
        ECGDataset(X_val, y_val),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    test_loader = DataLoader(
        ECGDataset(X_test, y_test),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=False,
        persistent_workers=False
    )

    return (
        train_loader,
        val_loader,
        test_loader,
        class_weights,
        len(np.unique(y_train))
    )

# ============================================================
# SQUEEZE AND EXCITATION BLOCK
# ============================================================

class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y


# ============================================================
# INCEPTION BLOCK
# ============================================================

class InceptionBlock(nn.Module):

    def __init__(self, in_channels, out_channels=64):
        super().__init__()

        self.branch1 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=9, padding=4, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch2 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=19, padding=9, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch3 = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=39, padding=19, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.branch4 = nn.Sequential(
            nn.MaxPool1d(kernel_size=3, stride=1, padding=1),
            nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.bn = nn.BatchNorm1d(out_channels * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(out_channels * 4)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)

        x = torch.cat([b1, b2, b3, b4], dim=1)
        x = self.bn(x)
        x = self.relu(x)
        x = self.se(x)
        return x


# ============================================================
# RESIDUAL BLOCK
# ============================================================

class ResidualInception(nn.Module):

    def __init__(self, channels):
        super().__init__()
        self.block1 = InceptionBlock(channels)
        self.block2 = InceptionBlock(channels)
        self.block3 = InceptionBlock(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = x + identity
        x = self.relu(x)
        return x


# ============================================================
# INCEPTIONTIME MODEL
# ============================================================

class InceptionTime(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.stem = InceptionBlock(12)
        self.res1 = ResidualInception(256)
        self.res2 = ResidualInception(256)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(256, num_classes)

    def forward(self, x):
        x = self.stem(x)
        x = self.res1(x)
        x = self.res2(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

# ============================================================
# BUILD MODEL
# ============================================================

def build_model(num_classes, class_weights):

    model = InceptionTime(num_classes).to(DEVICE)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.02)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=5e-5
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=EPOCHS,
        eta_min=1e-6
    )

    scaler = torch.amp.GradScaler(enabled=torch.cuda.is_available())

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(model)
    print(f"\nTotal Parameters : {total_params:,}")
    print(f"Trainable Params : {trainable_params:,}")

    return model, criterion, optimizer, scheduler, scaler

# ============================================================
# TRAIN ONE EPOCH
# ============================================================

def train_one_epoch(model, loader, criterion, optimizer, scaler):

    model.train()

    running_loss = 0.0
    predictions = []
    labels = []

    import time
    epoch_start = time.time()

    for batch_idx, (signals, targets) in enumerate(loader):

        if batch_idx % 10 == 0:
            elapsed = time.time() - epoch_start
            rate = (batch_idx + 1) / elapsed if elapsed > 0 else 0
            eta_min = (len(loader) - batch_idx) / rate / 60 if rate > 0 else 0
            print(f"Batch {batch_idx}/{len(loader)} | {rate:.2f} batch/s | ETA {eta_min:.1f} min", end="\r")

        signals = signals.to(DEVICE, non_blocking=True)
        targets = targets.to(DEVICE, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(
            device_type=DEVICE.type,
            dtype=torch.float16,
            enabled=torch.cuda.is_available()
        ):
            outputs = model(signals)
            loss = criterion(outputs, targets)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()

        preds = outputs.argmax(dim=1)
        predictions.extend(preds.detach().cpu().numpy())
        labels.extend(targets.detach().cpu().numpy())

    train_loss = running_loss / len(loader)
    train_acc = accuracy_score(labels, predictions)
    train_f1 = f1_score(labels, predictions, average="weighted")

    return train_loss, train_acc, train_f1


# ============================================================
# VALIDATION
# ============================================================

def validate(model, loader, criterion):

    model.eval()

    running_loss = 0.0
    predictions = []
    labels = []

    with torch.no_grad():
        for signals, targets in loader:

            signals = signals.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):
                outputs = model(signals)
                loss = criterion(outputs, targets)

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    val_loss = running_loss / len(loader)
    val_acc = accuracy_score(labels, predictions)
    val_f1 = f1_score(labels, predictions, average="weighted")

    return val_loss, val_acc, val_f1

# ============================================================
# TRAIN MODEL
# ============================================================

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, scaler):

    best_f1 = 0.0
    patience = 6
    wait = 0

    torch.cuda.empty_cache()

    for epoch in range(EPOCHS):

        print("\n" + "=" * 60)
        print(f"Epoch {epoch + 1}/{EPOCHS}")
        print("=" * 60)

        train_loss, train_acc, train_f1 = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler
        )

        val_loss, val_acc, val_f1 = validate(model, val_loader, criterion)

        scheduler.step()

        print()
        print(f"Train Loss : {train_loss:.4f} | Train Acc : {train_acc:.4f} | Train F1 : {train_f1:.4f}")
        print(f"Val Loss   : {val_loss:.4f} | Val Acc : {val_acc:.4f} | Val F1 : {val_f1:.4f}")

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"Learning Rate : {current_lr:.7f}")

        if val_f1 > best_f1:
            best_f1 = val_f1
            wait = 0

            torch.save(
                {
                    "epoch": epoch,
                    "best_f1": best_f1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict()
                },
                "best_inception.pth"
            )

            print()
            print("Best model saved.")
        else:
            wait += 1
            print()
            print(f"No Improvement ({wait}/{patience})")

        if wait >= patience:
            print()
            print("Early stopping triggered.")
            break

        torch.cuda.empty_cache()

    print()
    print("=" * 60)
    print("Training Finished")
    print("=" * 60)
    print(f"Best Validation F1 : {best_f1:.4f}")

    return best_f1

# ============================================================
# TEST MODEL
# ============================================================

def test_model(model, test_loader):

    checkpoint = torch.load("best_inception.pth", map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    predictions = []
    labels = []
    running_loss = 0.0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for signals, targets in test_loader:

            signals = signals.to(DEVICE, non_blocking=True)
            targets = targets.to(DEVICE, non_blocking=True)

            with torch.autocast(
                device_type=DEVICE.type,
                dtype=torch.float16,
                enabled=torch.cuda.is_available()
            ):
                outputs = model(signals)
                loss = criterion(outputs, targets)

            running_loss += loss.item()

            preds = outputs.argmax(dim=1)
            predictions.extend(preds.cpu().numpy())
            labels.extend(targets.cpu().numpy())

    test_loss = running_loss / len(test_loader)
    test_acc = accuracy_score(labels, predictions)
    test_f1 = f1_score(labels, predictions, average="weighted")

    print()
    print("=" * 60)
    print("FINAL TEST RESULTS")
    print("=" * 60)
    print(f"Test Loss    : {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Weighted F1  : {test_f1:.4f}")

    return test_loss, test_acc, test_f1


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print(torch.cuda.is_available())
    if torch.cuda.is_available():
        print(f"Total GPU Memory : {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"Allocated Memory : {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
        print(f"Reserved Memory  : {torch.cuda.memory_reserved() / 1024**2:.2f} MB")

    train_loader, val_loader, test_loader, class_weights, num_classes = load_data()

    model, criterion, optimizer, scheduler, scaler = build_model(
        num_classes=num_classes,
        class_weights=class_weights
    )

    print(f"\nModel is on device: {next(model.parameters()).device}")

    print()
    print("=" * 60)
    print("STARTING TRAINING")
    print("=" * 60)

    best_f1 = train_model(
        model, train_loader, val_loader, criterion, optimizer, scheduler, scaler
    )

    print()
    print("=" * 60)
    print("LOADING BEST MODEL")
    print("=" * 60)

    test_model(model, test_loader)

    print()
    print("=" * 60)
    print("DONE")
    print("=" * 60)
