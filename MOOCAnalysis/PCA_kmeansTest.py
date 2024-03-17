# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

# 读取数据
data = pd.read_csv('../data11/MOOCdataset.csv', encoding='GBK')

# 数据预处理
'''# 处理缺失值
imputer = SimpleImputer(strategy='mean')
data[['Assignments Done', 'Pauses', 'Rewinds', 'Forwards', 'Quiz Score','Math Score', 'Reading Score', 'Writing Score']]\
    = imputer.fit_transform(data[['Assignments Done', 'Pauses', 'Rewinds', 'Forwards', 'Quiz Score',
      'Math Score', 'Reading Score', 'Writing Score']])'''

# 缺失值处理
# 假设使用均值填充数值型特征的缺失值
numeric_features = ['Assignments Done', 'Pauses', 'Rewinds', 'Forwards', 'Quiz Score','Math Score', 'Reading Score', 'Writing Score']
imputer = SimpleImputer(strategy='mean')
data[numeric_features] = imputer.fit_transform(data[numeric_features])


# 异常值处理和数据清洗，这里可以根据实际情况进行处理，比如删除异常值或者进行平滑处理
'''
# 特征选择
# 假设我们选择使用数学成绩和阅读成绩作为特征
#selected_features = data[['Math Score', 'Reading Score']]

# 特征选择
# 假设选择数学成绩、阅读成绩、写作成绩作为特征
selected_features = data[['Math Score', 'Reading Score', 'Writing Score']]


# 数据增强（如果需要）

# PCA降维分析
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features) # 标准化数据
pca = PCA(n_components=2)# 假设降到2维
principal_components = pca.fit_transform(scaled_features)
# 将降维后的数据加入原始数据表
data['PCA1'] = principal_components[:, 0]
data['PCA2'] = principal_components[:, 1]
# 输出处理后的数据
print(data.head())

# K-means聚类分析
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(principal_components)

# 可视化聚类结果
plt.scatter(principal_components[data['cluster'] == 0, 0], principal_components[data['cluster'] == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(principal_components[data['cluster'] == 1, 0], principal_components[data['cluster'] == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(principal_components[data['cluster'] == 2, 0], principal_components[data['cluster'] == 2, 1], s=100, c='green', label='Cluster 3')
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
'''


'''
在聚类之前使用决策树对特征重要性分析
从而了解哪个特征在区分簇时起到更重要的作用。
'''
# 假设选择数学成绩、阅读成绩、写作成绩作为特征
selected_features = data[['Math Score', 'Reading Score', 'Writing Score']]

# 数据增强（如果需要）

# 使用决策树对特征重要性进行分析
X = selected_features
y = data[['Math Score', 'Reading Score', 'Writing Score']]  # 假设data里面是你要预测的目标

# 创建并训练决策树模型
tree_model = DecisionTreeClassifier(random_state=42)
tree_model.fit(X, y)

# 获取特征重要性
feature_importance = tree_model.feature_importances_

# 输出特征重要性
for i, feature in enumerate(selected_features.columns):
    print(f"{feature}的重要性: {feature_importance[i]}")

# 绘制特征重要性图表
plt.bar(selected_features.columns, feature_importance)
plt.xlabel('Feature')
plt.ylabel('Importance')
plt.title('Feature Importance')
plt.show()

# PCA降维分析
scaler = StandardScaler()
scaled_features = scaler.fit_transform(selected_features) # 标准化数据
pca = PCA(n_components=2)# 假设降到2维
principal_components = pca.fit_transform(scaled_features)
# 将降维后的数据加入原始数据表
data['PCA1'] = principal_components[:, 0]
data['PCA2'] = principal_components[:, 1]

# 输出处理后的数据
print(data.head())

# K-means聚类分析
kmeans = KMeans(n_clusters=3, random_state=42)
data['cluster'] = kmeans.fit_predict(principal_components)

# 可视化聚类结果
plt.scatter(principal_components[data['cluster'] == 0, 0], principal_components[data['cluster'] == 0, 1], s=100, c='red', label='Cluster 1')
plt.scatter(principal_components[data['cluster'] == 1, 0], principal_components[data['cluster'] == 1, 1], s=100, c='blue', label='Cluster 2')
plt.scatter(principal_components[data['cluster'] == 2, 0], principal_components[data['cluster'] == 2, 1], s=100, c='green', label='Cluster 3')
plt.title('K-means Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()
