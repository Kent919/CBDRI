# save as: generate_icc_heatmap.py
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# --- 1. 定义规则强度矩阵（示例数据）---
# 维度说明：[数据本地化, 问责机制, 执法协作, 例外条款, 数据可携]
regulatory_matrix = np.array([
    [0.1, 0.9, 0.8, 0.2, 0.85],  # EU (GDPR)
    [0.8, 0.7, 0.3, 0.6, 0.4],  # CN (中国)
    [0.3, 0.6, 0.7, 0.4, 0.7],  # SG (PDPA)
    [0.4, 0.6, 0.6, 0.5, 0.6],  # JP (APPI)
    [0.2, 0.8, 0.7, 0.3, 0.8],  # CA (PIPEDA)
    [0.35, 0.75, 0.8, 0.45, 0.7] # UK (UK GDPR)
])

jurisdictions = ['EU', 'CN', 'SG', 'JP', 'CA', 'UK']
dimensions = ['Data Localization', 'Accountability', 'Enforcement', 'Exceptions', 'Portability']

# --- 2. 数据标准化 + PCA 降维 ---
scaler = StandardScaler()
X_scaled = scaler.fit_transform(regulatory_matrix)

# 使用前2个主成分
pca = PCA(n_components=2)
weights = pca.fit_transform(X_scaled)  # 得到每个法域的二维权重向量

# --- 3. 计算 ICC 矩阵 ---
n = len(jurisdictions)
ICC = np.zeros((n, n))

# 计算最大可能距离（用于归一化）
all_dists = []
for i in range(n):
    for j in range(n):
        dist = np.linalg.norm(weights[i] - weights[j])
        all_dists.append(dist)
max_dist = max(all_dists) if max(all_dists) > 0 else 1.0

# 填充 ICC 矩阵
for i in range(n):
    for j in range(n):
        dist = np.linalg.norm(weights[i] - weights[j])
        ICC[i, j] = 1 - (dist / max_dist)

# --- 4. 创建 DataFrame 便于绘图 ---
df_icc = pd.DataFrame(ICC, index=jurisdictions, columns=jurisdictions)

# --- 5. 绘制热力图 ---
plt.figure(figsize=(10, 8))
sns.set(font_scale=1.1)
heatmap = sns.heatmap(
    df_icc,
    annot=True,
    fmt=".3f",
    cmap='RdYlGn',          # 红-黄-绿，直观表示耦合强弱
    center=0.5,
    square=True,
    cbar_kws={"shrink": 0.8, "label": "制度耦合系数 (ICC)"},
    linewidths=0.5,
    linecolor='gray'
)

plt.title('制度耦合系数 (ICC) 热力图\n基于国际数据治理规则比较', fontsize=16, pad=20)
plt.ylabel('源法域', fontsize=12)
plt.xlabel('目标法域', fontsize=12)
plt.xticks(rotation=0)
plt.yticks(rotation=0)

# --- 6. 保存为高分辨率 PNG ---
output_filename = 'icc_heatmap.png'
plt.tight_layout()
plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0.1)
print(f"✅ 热力图已保存为: {output_filename}")
print(f"   分辨率: 300 DPI, 格式: PNG")

# 显示（可选：在Jupyter或本地运行时）
plt.show()
