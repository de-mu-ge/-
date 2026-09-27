import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def PCA_256(X):
    # 原始数据
    # X = np.random.randn(30, 63, 1000)   # 你的数据
    n_samples, n_channels, n_features = X.shape
    # print(X.shape)  # (30, 63, 1000)

    # 1. 把前两维合并，最后一维作为特征
    X_2d = X.reshape(-1, n_features)     # (30*63, 1000) = (1890, 1000)

    # 2. 标准化（每个特征维度）
    scaler = StandardScaler()
    X_2d_scaled = scaler.fit_transform(X_2d)

    # 3. 先拟合全部，查看累计方差
    pca_full = PCA(n_components=min(X_2d_scaled.shape)).fit(X_2d_scaled)
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)

    # 4. 查看 256 维的累计方差
    # if len(cum_var) >= 256:
    #     print(f"前 256 维累计解释方差: {cum_var[255]:.4f}")
    # else:
    #     print(f"最多只有 {len(cum_var)} 维，累计方差: {cum_var[-1]:.4f}")

    # 5. 正式降到 256 维
    pca = PCA(n_components=256)
    X_2d_pca = pca.fit_transform(X_2d_scaled)   # (1890, 256)
    # print(f"256 维累计解释方差: {pca.explained_variance_ratio_.sum():.4f}")

    # 6. reshape 回 3D
    X_pca = X_2d_pca.reshape(n_samples, n_channels, 256)

    return X_pca
    # print(X_pca.shape)   # (30, 63, 256)

if "__main__" == __name__:
    arr = np.random.random([30, 63, 1000])
    pca = PCA_256(arr)

    print(pca.shape)