import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# 示例：规则强度矩阵（n法域 x m维度）
# 维度：[数据本地化, 问责机制, 执法协作, 例外条款]
regulatory_matrix = np.array([
    [0.1, 0.9, 0.8, 0.2],  # EU (GDPR)
    [0.8, 0.7, 0.3, 0.6],  # CN (中国)
    [0.3, 0.6, 0.7, 0.4],  # SG
    [0.4, 0.6, 0.6, 0.5],  # JP
    [0.2, 0.8, 0.7, 0.3]   # CA
])

jurisdictions = ['EU', 'CN', 'SG', 'JP', 'CA']

# PCA降维（简化：直接使用原矩阵或标准化后计算）
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

scaler = StandardScaler()
X_scaled = scaler.fit_transform(regulatory_matrix)

pca = PCA(n_components=2)
weights = pca.fit_transform(X_scaled)  # 得到二维监管权重向量

# 计算ICC矩阵
n = len(jurisdictions)
ICC = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        dist = np.linalg.norm(weights[i] - weights[j])
        max_dist = np.max([np.linalg.norm(weights[a] - weights[b]) for a in range(n) for b in range(n)])
        ICC[i, j] = 1 - (dist / max_dist) if max_dist > 0 else 1.0

# 可视化热力图
df_icc = pd.DataFrame(ICC, index=jurisdictions, columns=jurisdictions)
plt.figure(figsize=(8, 6))
sns.heatmap(df_icc, annot=True, cmap='coolwarm', center=0.5, square=True, cbar_kws={'label': 'ICC'})
plt.title('制度耦合系数（ICC）热力图')
plt.ylabel('源法域')
plt.xlabel('目标法域')
plt.tight_layout()
plt.savefig('icc_heatmap.png', dpi=300)
plt.show()

print("ICC矩阵：")
print(df_icc)
