"""Train the original 384px OOF ensemble pipeline on Kaggle."""

import os
import gc
import random
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
import torch.nn.functional as F

# ==========================================
# 1. Configuration & Paths
# ==========================================
DATA_ROOT = "/kaggle/input/datasets/hyunwoo11/chest-xray-ai"
WORK_DIR = "/kaggle/working"
SAMPLE_SUBMISSION_PATH = "/kaggle/input/datasets/hyunwoo11/submission/sample_submission.csv"

TRAIN_CSV = f"{DATA_ROOT}/train_split.csv"
VAL_CSV = f"{DATA_ROOT}/val_split.csv"
TEST_CSV = f"{DATA_ROOT}/test.csv"
IMAGE_DIR = f"{DATA_ROOT}/data/images"

# Hyperparameters
EPOCHS = 10                # 10 epochs is optimal for transfer learning with SWA
NUM_FOLDS = 5
SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Upgraded resolution to 384x384 to preserve micro-textures & vascular features of pneumonia
IMAGE_SIZE = 384
BATCH_SIZE_DENSE = 16
BATCH_SIZE_CONV = 16
BATCH_SIZE_EFF = 16

LR_HEAD = 1e-3
LR_BACKBONE = 1e-4
WEIGHT_DECAY = 1e-4

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed_everything(SEED)

# ==========================================
# 2. Advanced Preprocessing & Dataset (CLAHE)
# ==========================================
class CLAHETransform:
    def __init__(self, clip_limit=2.0, tile_grid_size=(8, 8)):
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)

    def __call__(self, img_pil):
        img_np = np.array(img_pil)
        lab = cv2.cvtColor(img_np, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        cl = self.clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        final_img = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
        return Image.fromarray(final_img)

class KaggleChestXrayDataset(Dataset):
    def __init__(self, df, image_dir, transform=None, is_test=False):
        self.df = df.reset_index(drop=True)
        self.image_dir = image_dir
        self.transform = transform
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def _resolve_path(self, file_name):
        paths_to_check = [
            os.path.join(self.image_dir, file_name),
            os.path.join(self.image_dir, "train", file_name),
            os.path.join(self.image_dir, "test", file_name)
        ]
        for p in paths_to_check:
            if os.path.exists(p):
                return p
        return paths_to_check[0]

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_name = row["file_name"]

        img_path = self._resolve_path(file_name)
        try:
            image = Image.open(img_path).convert("RGB")
        except FileNotFoundError:
            image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE))

        if self.transform is not None:
            image = self.transform(image)

        if self.is_test:
            return image, file_name
        else:
            return image, int(row["label"])

# --- Training Augmentation Transform (with 1.1x Resize -> CenterCrop) ---
train_transform = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Inference TTA Scale A (Margin Crop) ---
val_test_transform_crop = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((int(IMAGE_SIZE * 1.1), int(IMAGE_SIZE * 1.1))),
    transforms.CenterCrop(IMAGE_SIZE),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# --- Inference TTA Scale B (Direct Full Resize) ---
val_test_transform_full = transforms.Compose([
    CLAHETransform(),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# ==========================================
# 3. Model Architectures & Loss Functions
# ==========================================
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return x * self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        out = self.conv1(x_cat)
        return x * self.sigmoid(out)

class CBAM(nn.Module):
    def __init__(self, in_planes, ratio=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x

class DenseNet121_CBAM(nn.Module):
    def __init__(self):
        super().__init__()
        base = models.densenet121(weights=models.DenseNet121_Weights.IMAGENET1K_V1)
        self.features = base.features
        self.cbam = CBAM(in_planes=1024)
        self.classifier = nn.Sequential(nn.Dropout(0.4), nn.Linear(1024, 1))

    def forward(self, x):
        features = self.features(x)
        features = F.relu(features, inplace=True)
        features = self.cbam(features)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out

def get_densenet():
    return DenseNet121_CBAM().to(DEVICE)

def get_convnext():
    model = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
    in_features = model.classifier[2].in_features
    model.classifier[2] = nn.Linear(in_features, 1)
    return model.to(DEVICE)

# Swapped Swin-T for EfficientNet-B3 (Better convolutional inductive bias for 384x384 resolution)
def get_efficientnet():
    model = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.IMAGENET1K_V1)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, 1)
    return model.to(DEVICE)

# Balanced Smooth Focal Loss addressing the 3:1 class imbalance
class SmoothFocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, smoothing=0.05):
        """
        Since Pneumonia (Class 1) has 3100 samples and Normal (Class 0) has 1073,
        alpha=0.25 weights Class 1 by 0.25, and Class 0 by 0.75 (1 - alpha), perfectly balancing the loss.
        """
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.smoothing = smoothing

    def forward(self, inputs, targets):
        smoothed_targets = targets * (1.0 - self.smoothing) + 0.5 * self.smoothing
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, smoothed_targets, reduction='none')
        pt = torch.exp(-BCE_loss)

        # Apply alpha balancing conditionally based on actual targets
        alpha_factor = targets * self.alpha + (1.0 - targets) * (1.0 - self.alpha)

        F_loss = alpha_factor * (1.0 - pt)**self.gamma * BCE_loss
        return torch.mean(F_loss)

def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size()[0]
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

# SWA Checkpoint Averaging Utility (Averages state dicts of last 3 epochs)
def average_checkpoints(checkpoint_paths):
    state_dicts = [torch.load(p, map_location='cpu') for p in checkpoint_paths]
    averaged_state_dict = {}
    for key in state_dicts[0].keys():
        if torch.is_tensor(state_dicts[0][key]):
            if state_dicts[0][key].is_floating_point():
                averaged_state_dict[key] = torch.stack([sd[key] for sd in state_dicts]).mean(dim=0)
            else:
                averaged_state_dict[key] = state_dicts[0][key]
        else:
            averaged_state_dict[key] = state_dicts[0][key]
    return averaged_state_dict

# ==========================================
# 4. Training Engine (Cosine Annealing + Warmup + Mixup + SWA Checkpoint Saving)
# ==========================================
def train_fold(model_name, fold, train_loader, val_loader):
    print(f"\n>>> [Fold {fold+1}/{NUM_FOLDS}] Training {model_name}...")

    if model_name == "DenseNet121":
        model = get_densenet()
        backbone_params = model.features.parameters()
        head_params = list(model.classifier.parameters()) + list(model.cbam.parameters())
    elif model_name == "ConvNeXt":
        model = get_convnext()
        backbone_params = model.features.parameters()
        head_params = model.classifier.parameters()
    else:
        model = get_efficientnet()
        backbone_params = model.features.parameters()
        head_params = model.classifier.parameters()

    criterion = SmoothFocalLoss(alpha=0.25, gamma=2.0, smoothing=0.05)

    # Epoch 1-2: Freeze Backbone
    for param in backbone_params:
        param.requires_grad = False

    optimizer = optim.AdamW(head_params, lr=LR_HEAD, weight_decay=WEIGHT_DECAY)

    # Checkpoints explicitly named with _384
    best_model_path = os.path.join(WORK_DIR, f"best_{model_name.lower()}_384_fold{fold}.pt")
    last_checkpoints = []

    scheduler = None

    for epoch in range(EPOCHS):
        # Epoch 3: Unfreeze backbone with a 1-epoch Warmup learning rate
        if epoch == 2:
            print(">>> Unfreezing Backbone with 1-Epoch Warmup (Backbone LR: 1e-5)...")
            if model_name == "DenseNet121":
                backbone_params = model.features.parameters()
                head_params = list(model.classifier.parameters()) + list(model.cbam.parameters())
            elif model_name == "ConvNeXt":
                backbone_params = model.features.parameters()
                head_params = model.classifier.parameters()
            else:
                backbone_params = model.features.parameters()
                head_params = model.classifier.parameters()

            for param in backbone_params:
                param.requires_grad = True

            # Use smaller LR (1e-5) for backbone during warmup epoch to prevent weights corruption
            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': LR_BACKBONE * 0.1},
                {'params': head_params, 'lr': LR_HEAD}
            ], weight_decay=WEIGHT_DECAY)

        # Epoch 4: Start full learning rate and Cosine Annealing scheduler
        elif epoch == 3:
            print(">>> Transitioning to Full Backbone LR with Cosine Annealing (Backbone LR: 1e-4)...")
            if model_name == "DenseNet121":
                backbone_params = model.features.parameters()
                head_params = list(model.classifier.parameters()) + list(model.cbam.parameters())
            elif model_name == "ConvNeXt":
                backbone_params = model.features.parameters()
                head_params = model.classifier.parameters()
            else:
                backbone_params = model.features.parameters()
                head_params = model.classifier.parameters()

            optimizer = optim.AdamW([
                {'params': backbone_params, 'lr': LR_BACKBONE},
                {'params': head_params, 'lr': LR_HEAD}
            ], weight_decay=WEIGHT_DECAY)

            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(EPOCHS - 3), eta_min=1e-6)

        model.train()
        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS} [Train]", leave=False):
            images = images.to(DEVICE)
            labels = labels.to(DEVICE).float().unsqueeze(1)

            if random.random() < 0.5:
                mixed_images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2)
                optimizer.zero_grad()
                outputs = model(mixed_images)
                loss = lam * criterion(outputs, labels_a) + (1.0 - lam) * criterion(outputs, labels_b)
            else:
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)

            loss.backward()
            optimizer.step()

        if scheduler is not None:
            scheduler.step()

        # Save checkpoints of the last 3 epochs for Stochastic Weight Averaging (SWA)
        if epoch >= (EPOCHS - 3):
            epoch_ckpt_path = os.path.join(WORK_DIR, f"temp_{model_name.lower()}_384_fold{fold}_epoch{epoch}.pt")
            torch.save(model.state_dict(), epoch_ckpt_path)
            last_checkpoints.append(epoch_ckpt_path)

    # Perform SWA Weight Averaging
    print(f"Applying Stochastic Weight Averaging (SWA) for {model_name}...")
    swa_state_dict = average_checkpoints(last_checkpoints)
    torch.save(swa_state_dict, best_model_path)

    # Calculate SWA F1 validation score
    model.load_state_dict(swa_state_dict)
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(DEVICE)
            outputs = model(images)
            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).int()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())

    val_f1 = f1_score(all_labels, all_preds)
    print(f"-> Fold {fold+1} {model_name} SWA Finished. Val F1: {val_f1:.4f}")

    # Clean up temporary epoch files
    for cp in last_checkpoints:
        if os.path.exists(cp):
            os.remove(cp)

    del model, optimizer, scheduler
    gc.collect()
    torch.cuda.empty_cache()

    return best_model_path, val_f1

# ==========================================
# 5. Out-Of-Fold Evaluation & Threshold Optimization
# ==========================================
def find_optimal_oof_threshold(dense_models, conv_models, eff_models, df, folds, w_dense, w_conv, w_eff):
    print("\n[Searching for Optimal Threshold via 3-Model Out-of-Fold Predictions...]")

    oof_probs = np.zeros(len(df))
    all_labels = np.zeros(len(df))

    for fold, (train_idx, val_idx) in enumerate(folds):
        val_df = df.iloc[val_idx]

        # Validation utilizes TTA Scale A (Margin Crop) for optimal OOF search
        val_ds = KaggleChestXrayDataset(val_df, IMAGE_DIR, transform=val_test_transform_crop)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_DENSE, shuffle=False, num_workers=2)

        m_dense = get_densenet()
        m_dense.load_state_dict(torch.load(dense_models[fold]))
        m_dense.eval()

        m_conv = get_convnext()
        m_conv.load_state_dict(torch.load(conv_models[fold]))
        m_conv.eval()

        m_eff = get_efficientnet()
        m_eff.load_state_dict(torch.load(eff_models[fold]))
        m_eff.eval()

        dense_preds, conv_preds, eff_preds = [], [], []
        fold_labels = []

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(DEVICE)

                dense_preds.extend(torch.sigmoid(m_dense(images)).cpu().numpy())
                conv_preds.extend(torch.sigmoid(m_conv(images)).cpu().numpy())
                eff_preds.extend(torch.sigmoid(m_eff(images)).cpu().numpy())
                fold_labels.extend(labels.numpy())

        dense_preds = np.array(dense_preds).flatten()
        conv_preds = np.array(conv_preds).flatten()
        eff_preds = np.array(eff_preds).flatten()

        fold_oof = (w_dense * dense_preds) + (w_conv * conv_preds) + (w_eff * eff_preds)
        oof_probs[val_idx] = fold_oof
        all_labels[val_idx] = np.array(fold_labels)

        del m_dense, m_conv, m_eff
        gc.collect()
        torch.cuda.empty_cache()

    best_thresh = 0.5
    best_f1 = 0.0
    for thresh in np.arange(0.1, 0.9, 0.02):
        preds = (oof_probs > thresh).astype(int)
        score = f1_score(all_labels, preds)
        if score > best_f1:
            best_f1 = score
            best_thresh = thresh

    print(f">>> 3-Model Out-Of-Fold F1: {best_f1:.4f} at Optimal Threshold: {best_thresh:.3f}")
    return best_thresh

# ==========================================
# 6. Inference (15-Model Ensemble + 4-Way Multi-Scale TTA)
# ==========================================
def predict_with_tta(model, loader_crop, loader_full):
    """Generates TTA predictions: averages predictions across Margin-Crop, Margin-Crop-Flipped, Full-Resize, and Full-Resize-Flipped"""
    probs_list = []

    # Crucial Fix: Wrap inference in torch.no_grad() to prevent OOM
    with torch.no_grad():
        # Pack dataloaders to run synchronously
        for (images_crop, _), (images_full, _) in zip(loader_crop, loader_full):
            images_crop = images_crop.to(DEVICE)
            images_full = images_full.to(DEVICE)

            # 1. Scale A: CenterCrop Margin Removal (Original + Flipped)
            p_crop_orig = torch.sigmoid(model(images_crop))
            images_crop_flipped = torch.flip(images_crop, dims=[3])
            p_crop_flipped = torch.sigmoid(model(images_crop_flipped))

            # 2. Scale B: Full Rescale (Original + Flipped)
            p_full_orig = torch.sigmoid(model(images_full))
            images_full_flipped = torch.flip(images_full, dims=[3])
            p_full_flipped = torch.sigmoid(model(images_full_flipped))

            # Average all 4 spatial/scale variations
            batch_probs = (p_crop_orig + p_crop_flipped + p_full_orig + p_full_flipped) / 4.0
            probs_list.extend(batch_probs.cpu().numpy())

    return np.array(probs_list).flatten()

def generate_final_submission(dense_models, conv_models, eff_models, threshold, w_dense, w_conv, w_eff):
    print(f"\n[Running Final Inference: 15-Model Ensemble + 4-Way Multi-Scale TTA (Threshold: {threshold:.3f})]")

    test_df = pd.read_csv(TEST_CSV)

    # Dataloaders for TTA Scale A (Margin Crop)
    test_ds_crop = KaggleChestXrayDataset(test_df, IMAGE_DIR, transform=val_test_transform_crop, is_test=True)
    test_loader_crop = DataLoader(test_ds_crop, batch_size=BATCH_SIZE_DENSE, shuffle=False, num_workers=2)

    # Dataloaders for TTA Scale B (Full Resize)
    test_ds_full = KaggleChestXrayDataset(test_df, IMAGE_DIR, transform=val_test_transform_full, is_test=True)
    test_loader_full = DataLoader(test_ds_full, batch_size=BATCH_SIZE_DENSE, shuffle=False, num_workers=2)

    accumulated_probs = np.zeros(len(test_df))

    # Extract file names
    file_names = test_df["file_name"].tolist()

    # 1. Accumulate DenseNet-CBAM Folds (5 models)
    for fold in range(NUM_FOLDS):
        model = get_densenet()
        model.load_state_dict(torch.load(dense_models[fold]))
        model.eval()

        dense_preds = predict_with_tta(model, test_loader_crop, test_loader_full)
        accumulated_probs += w_dense * dense_preds / NUM_FOLDS

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 2. Accumulate ConvNeXt Folds (5 models)
    for fold in range(NUM_FOLDS):
        model = get_convnext()
        model.load_state_dict(torch.load(conv_models[fold]))
        model.eval()

        conv_preds = predict_with_tta(model, test_loader_crop, test_loader_full)
        accumulated_probs += w_conv * conv_preds / NUM_FOLDS

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # 3. Accumulate EfficientNet Folds (5 models)
    for fold in range(NUM_FOLDS):
        model = get_efficientnet()
        model.load_state_dict(torch.load(eff_models[fold]))
        model.eval()

        eff_preds = predict_with_tta(model, test_loader_crop, test_loader_full)
        accumulated_probs += w_eff * eff_preds / NUM_FOLDS

        del model
        gc.collect()
        torch.cuda.empty_cache()

    # Apply Threshold
    final_preds = (accumulated_probs > threshold).astype(int)

    results = [{"file_name": name, "label": pred} for name, pred in zip(file_names, final_preds)]
    submission_df = pd.DataFrame(results)

    try:
        sample_df = pd.read_csv(SAMPLE_SUBMISSION_PATH)
        submission_df = sample_df[['file_name']].merge(submission_df, on='file_name', how='left')
    except Exception as e:
        print(f"Could not align with sample submission: {e}")

    final_sub_path = os.path.join(WORK_DIR, "submission.csv")
    submission_df.to_csv(final_sub_path, index=False)
    print(f"Final 3-Model 15-Fold Ensemble Submission saved to: {final_sub_path}")

# ==========================================
# 7. Main Execution Flow
# ==========================================
if __name__ == "__main__":
    train_df = pd.read_csv(TRAIN_CSV)
    val_df = pd.read_csv(VAL_CSV)
    full_df = pd.concat([train_df, val_df], ignore_index=True)

    print(f"Grandmaster Mode 384x384: {len(full_df)} samples combined for 5-Fold Stratified Group CV.")

    sgkf = StratifiedGroupKFold(n_splits=NUM_FOLDS, shuffle=True, random_state=SEED)
    folds = list(sgkf.split(full_df, full_df['label'], groups=full_df['duplicate_group_id']))

    dense_paths, dense_f1s = [], []
    conv_paths, conv_f1s = [], []
    eff_paths, eff_f1s = [], []

    # 1. Train 5-Fold DenseNet121-CBAM
    for fold, (train_idx, val_idx) in enumerate(folds):
        f_train_df = full_df.iloc[train_idx]
        f_val_df = full_df.iloc[val_idx]

        train_ds = KaggleChestXrayDataset(f_train_df, IMAGE_DIR, transform=train_transform)
        val_ds = KaggleChestXrayDataset(f_val_df, IMAGE_DIR, transform=val_test_transform_crop)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_DENSE, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_DENSE, shuffle=False, num_workers=2)

        path, f1 = train_fold("DenseNet121", fold, train_loader, val_loader)
        dense_paths.append(path)
        dense_f1s.append(f1)

    # 2. Train 5-Fold ConvNeXt-Tiny
    for fold, (train_idx, val_idx) in enumerate(folds):
        f_train_df = full_df.iloc[train_idx]
        f_val_df = full_df.iloc[val_idx]

        train_ds = KaggleChestXrayDataset(f_train_df, IMAGE_DIR, transform=train_transform)
        val_ds = KaggleChestXrayDataset(f_val_df, IMAGE_DIR, transform=val_test_transform_crop)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_CONV, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_CONV, shuffle=False, num_workers=2)

        path, f1 = train_fold("ConvNeXt", fold, train_loader, val_loader)
        conv_paths.append(path)
        conv_f1s.append(f1)

    # 3. Train 5-Fold EfficientNet-B3
    for fold, (train_idx, val_idx) in enumerate(folds):
        f_train_df = full_df.iloc[train_idx]
        f_val_df = full_df.iloc[val_idx]

        train_ds = KaggleChestXrayDataset(f_train_df, IMAGE_DIR, transform=train_transform)
        val_ds = KaggleChestXrayDataset(f_val_df, IMAGE_DIR, transform=val_test_transform_crop)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE_EFF, shuffle=True, num_workers=2, drop_last=True)
        val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE_EFF, shuffle=False, num_workers=2)

        path, f1 = train_fold("EfficientNet", fold, train_loader, val_loader)
        eff_paths.append(path)
        eff_f1s.append(f1)

    # Step 4: Compute Model Weights
    avg_dense_f1 = np.mean(dense_f1s)
    avg_conv_f1 = np.mean(conv_f1s)
    avg_eff_f1 = np.mean(eff_f1s)

    total_f1 = avg_dense_f1 + avg_conv_f1 + avg_eff_f1 + 1e-8
    w_dense = avg_dense_f1 / total_f1
    w_conv = avg_conv_f1 / total_f1
    w_eff = avg_eff_f1 / total_f1

    # Step 5: Find Out-Of-Fold Optimal Threshold
    opt_thresh = find_optimal_oof_threshold(
        dense_paths, conv_paths, eff_paths, full_df, folds, w_dense, w_conv, w_eff
    )

    # Step 6: Run Inference & Generate Submission
    generate_final_submission(
        dense_paths, conv_paths, eff_paths, opt_thresh, w_dense, w_conv, w_eff
    )

    print("\n✅ Ultimate 10-Model Grandmaster Pipeline Completed!")
