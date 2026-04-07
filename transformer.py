import os
import numpy as np
import rasterio
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from sklearn.preprocessing import StandardScaler

# ---------- 设置随机种子 ----------
torch.manual_seed(42)
np.random.seed(42)

# ---------- 获取有效像素 ----------
def get_valid_pixels(data, nodata):
    if nodata is None:
        nodata = -3.4028235e+38
    data = data.astype('float32')
    data[(data == nodata) | (data < -1e20)] = np.nan
    mask = ~np.isnan(data)
    rows, cols = np.where(mask)
    return rows, cols, data[rows, cols]

# ---------- 加载单个文件字典并剔除无效像素 ----------
def load_raster_data(file_dict, verbose=True):
    feature_order = ['SRTM', 'Slope', 'LandCover', 'aspect', 'TPI', 'TRI']
    with rasterio.open(file_dict['Error']) as src:
        error_data = src.read(1)
        rows, cols, error_values = get_valid_pixels(error_data, src.nodata)

    all_features = []
    for key in feature_order:
        with rasterio.open(file_dict[key]) as src:
            all_features.append(src.read(1)[rows, cols])

    features = np.array(all_features).T
    targets = np.array(error_values)

    # 无效值过滤
    landcover_idx = feature_order.index('LandCover')
    tpi_idx = feature_order.index('TPI')
    tri_idx = feature_order.index('TRI')
    valid_mask = (features[:, landcover_idx] != 0) & (features[:, landcover_idx] != -128) & \
                 (features[:, tpi_idx] > -1e20) & (features[:, tri_idx] > -1e20)

    if verbose:
        print("将删除", np.sum(~valid_mask), "行 (LandCover/TPI/TRI 无效)")

    return features[valid_mask], targets[valid_mask], feature_order

# ---------- 加载所有训练数据 ----------
def load_all_rasters(file_dicts):
    all_features, all_targets = [], []
    feature_order = None
    for fd in file_dicts:
        feats, tars, f_order = load_raster_data(fd)
        all_features.append(feats)
        all_targets.append(tars)
        if feature_order is None:
            feature_order = f_order
    return np.vstack(all_features), np.concatenate(all_targets), feature_order

# ---------- 数据集类 ----------
class DEMDatasetEmbedding(Dataset):
    def __init__(self, features, targets, feature_names):
        self.feature_names = feature_names
        landcover_idx = feature_names.index('LandCover')

        landcover = features[:, landcover_idx].astype(int)
        cont_feats = np.delete(features, landcover_idx, axis=1)

        self.scaler = StandardScaler()
        cont_feats = self.scaler.fit_transform(cont_feats)

        self.cont_feats = torch.FloatTensor(cont_feats)
        self.landcover = torch.LongTensor(landcover)
        self.targets = torch.FloatTensor(targets)
        self.fitted_scaler = self.scaler

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        return self.cont_feats[idx], self.landcover[idx], self.targets[idx]

# ---------- Transformer 模型 ----------
class DEMTransformerWithEmbedding(nn.Module):
    def __init__(self, input_dim, num_classes=256, embed_dim=8, d_model=128, nhead=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.embedding = nn.Embedding(num_classes, embed_dim)
        self.input_fc = nn.Sequential(
            nn.Linear(input_dim - 1 + embed_dim, d_model),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead,
                                                   dim_feedforward=d_model * 4, dropout=dropout,
                                                   batch_first=True, activation='gelu')
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.regressor = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x_cont, x_cat):
        x_embed = self.embedding(x_cat)
        x = torch.cat([x_cont, x_embed], dim=1)
        x = self.input_fc(x).unsqueeze(1)
        x = self.encoder(x).squeeze(1)
        return self.regressor(x).squeeze(-1)

# ---------- 训练函数 ----------
def train_model(files, epochs=100, batch_size=1024):
    log_path = "train_log.txt"
    X, y, feature_names = load_all_rasters(files)
    dataset = DEMDatasetEmbedding(X, y, feature_names)

    global fitted_scaler
    fitted_scaler = dataset.fitted_scaler
    joblib.dump(fitted_scaler, "scaler.pkl")
    print("✅ 标准化器已保存为 scaler.pkl")

    total_len = len(dataset)
    test_len = int(0.2 * total_len)
    train_len = total_len - test_len
    train_set, test_set = random_split(dataset, [train_len, test_len])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = DEMTransformerWithEmbedding(input_dim=X.shape[1], num_classes=256).cuda()
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)

    best_mae_improvement = -float('inf')
    best_val_mae = best_val_rmse = best_raw_mae = best_raw_rmse = None
    best_train_mae = best_train_rmse = None

    with open(log_path, "w", encoding="utf-8") as log_file:
        log_file.write("Epoch,Train MAE,Train RMSE,Val MAE,Val RMSE,Raw MAE,Raw RMSE,MAE 提升\n")

    for epoch in range(epochs):
        model.train()
        for X_cont, X_cat, y_batch in train_loader:
            X_cont, X_cat, y_batch = X_cont.cuda(), X_cat.cuda(), y_batch.cuda()
            pred = model(X_cont, X_cat)
            loss = criterion(pred, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # 计算训练集指标
        model.eval()
        train_preds, train_gts = [], []
        with torch.no_grad():
            for X_cont, X_cat, y_batch in train_loader:
                pred = model(X_cont.cuda(), X_cat.cuda()).cpu()
                train_preds.append(pred)
                train_gts.append(y_batch)
        train_preds = torch.cat(train_preds).numpy()
        train_gts = torch.cat(train_gts).numpy()
        train_mae = mean_absolute_error(train_gts, train_preds)
        train_rmse = root_mean_squared_error(train_gts, train_preds)

        # 验证集指标
        preds, gts = [], []
        with torch.no_grad():
            for X_cont, X_cat, y_batch in test_loader:
                pred = model(X_cont.cuda(), X_cat.cuda()).cpu()
                preds.append(pred)
                gts.append(y_batch)
        preds = torch.cat(preds).numpy()
        gts = torch.cat(gts).numpy()

        val_mae = mean_absolute_error(gts, preds)
        val_rmse = root_mean_squared_error(gts, preds)
        raw_mae = np.mean(np.abs(gts))
        raw_rmse = np.sqrt(np.mean(np.square(gts)))
        mae_improvement = raw_mae - val_mae

        if mae_improvement > best_mae_improvement:
            best_mae_improvement = mae_improvement
            best_val_mae, best_val_rmse = val_mae, val_rmse
            best_raw_mae, best_raw_rmse = raw_mae, raw_rmse
            best_train_mae, best_train_rmse = train_mae, train_rmse
            torch.save(model.state_dict(), "best_transformer_model.pth")
            print(f"✅ 已保存最佳模型权重，Epoch: {epoch + 1}, MAE提升: {mae_improvement:.4f}")

        log_msg = (f"Epoch {epoch+1:03d}/{epochs} | Train MAE: {train_mae:.4f} | Train RMSE: {train_rmse:.4f} | "
                   f"Val MAE: {val_mae:.4f} | Val RMSE: {val_rmse:.4f} | "
                   f"测试集真实 Error MAE: {raw_mae:.4f} | 测试集真实 Error RMSE: {raw_rmse:.4f} | "
                   f"MAE提升: {mae_improvement:.4f}")
        print(log_msg)

        with open(log_path, "a", encoding="utf-8") as log_file:
            log_file.write(f"{epoch+1},{train_mae:.4f},{train_rmse:.4f},{val_mae:.4f},{val_rmse:.4f},{raw_mae:.4f},{raw_rmse:.4f},{mae_improvement:.4f}\n")

    # 最终结果总结
    summary_msg = (
        f"\n✅ 最佳模型指标:\n"
        f"训练集 MAE: {best_train_mae:.4f} | 训练集 RMSE: {best_train_rmse:.4f}\n"
        f"验证集 MAE: {best_val_mae:.4f} | 验证集 RMSE: {best_val_rmse:.4f}\n"
        f"测试集真实 Error MAE: {best_raw_mae:.4f} | 测试集真实 Error RMSE: {best_raw_rmse:.4f}\n"
        f"测试集 MAE 提升: {best_mae_improvement:.4f}"
    )
    print(summary_msg)
    with open(log_path, "a", encoding="utf-8") as log_file:
        log_file.write("\n" + summary_msg + "\n")

    return feature_names, model

# ---------- 主程序 ----------
if __name__ == "__main__":
    files = [
        {
            'SRTM': 'SRTM_train.tif',
            'Slope': 'Slope_train.tif',
            'LandCover': 'Landcover_train.tif',
            'aspect': 'aspect_train.tif',
            'TPI': 'TPI_train.tif',
            'TRI': 'TRI_train.tif',
            'Error': 'error_train.tif'
        },
        {
            'SRTM': 'SRTM_val.tif',
            'Slope': 'Slope_val.tif',
            'LandCover': 'Landcover_val.tif',
            'aspect': 'aspect_val.tif',
            'TPI': 'TPI_val.tif',
            'TRI': 'TRI_val.tif',
            'Error': 'error_val.tif'
        }
    ]

    feature_names, model = train_model(files, epochs=100, batch_size=1024)
