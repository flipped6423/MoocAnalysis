# 导入必要的库
import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns

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

'''# 特征选择
# 假设我们选择使用数学成绩和阅读成绩作为特征
selected_features = data[['Math Score', 'Reading Score']]'''

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
'''# 绘制年龄分布直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6))
plt.hist(data['Age'], bins=11, color='skyblue', edgecolor='black')  # 调整 bins 的数量以获得更适合的直方图
plt.xlabel('年龄')
plt.ylabel('人数')
plt.title('年龄分布直方图')
plt.grid(axis='y', alpha=0.75)
plt.show()
#绘制观看时长与测验成绩关系
# 示例代码，假设将观看时长分为四个等分组，统计每个分组的测验成绩平均值
data['Hours Viewed Group'] = pd.cut(data['Hours Viewed'], bins=5)
average_scores = data.groupby('Hours Viewed Group')['Quiz Score'].mean()

#plt.figure(figsize=(10, 6))
sns.barplot(x=average_scores.index, y=average_scores.values)
plt.title('观看时长与测验成绩关系')
plt.xlabel('观看时长分组')
plt.ylabel('平均测验成绩')
plt.show()'''

'''# 创建一个散点图，展示观看时长与测验成绩的关系，用颜色表示数学成绩，点的大小表示阅读成绩
过于密集
plt.figure(figsize=(10, 6))
scatter = plt.scatter(data['Hours Viewed'], data['Quiz Score'], c=data['Math Score'],
                      s=data['Reading Score']*10, cmap='viridis', alpha=0.6)
plt.colorbar(scatter, label='Math Score')
plt.xlabel('观看时长')
plt.ylabel('测验成绩')
plt.title('观看时长与测验成绩关系图')

plt.show()'''

'''# 创建观看时长和各科成绩的分组
过于密集
data['Hours Group'] = pd.cut(data['Hours Viewed'], bins=3)

# 将各科成绩分组
data['Math Score Group'] = pd.cut(data['Math Score'], bins=3, labels=['Low', 'Medium', 'High'])
data['Reading Score Group'] = pd.cut(data['Reading Score'], bins=3, labels=['Low', 'Medium', 'High'])

# 使用Seaborn进行可视化
plt.figure(figsize=(10, 6))
scatter = sns.scatterplot(x='Hours Viewed', y='Quiz Score', hue='Math Score Group',
                          size='Reading Score', sizes=(50, 200), data=data)
plt.xlabel('观看时长')
plt.ylabel('测验成绩')
plt.title('观看时长与测验成绩关系图')

plt.show()'''

#绘制不同类型图标结合图
# 创建数据的分组
data['Hours Viewed Group'] = pd.cut(data['Hours Viewed'], bins=5)
data['Math Score Group'] = pd.cut(data['Math Score'], bins=3, labels=['低', '中', '高'])
data['Reading Score Group'] = pd.cut(data['Reading Score'], bins=3, labels=['低', '中', '高'])
data['Writing Score Group'] = pd.cut(data['Writing Score'], bins=3, labels=['低', '中', '高'])
# 绘制不同类型图表的结合图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(15, 10))

plt.subplot(4, 2, 1)
sns.barplot(x='Hours Viewed Group', y='Quiz Score', data=data, estimator=sum, ci=None)
plt.xlabel('观看时长分组')
plt.ylabel('测验成绩')
plt.title('观看时长分组与测验成绩关系')

plt.subplot(4, 2, 2)
sns.lineplot(x='Math Score Group', y='Quiz Score', data=data, estimator=sum, ci=None)
plt.xlabel('数学成绩分组')
plt.ylabel('测验成绩总和')
plt.title('数学成绩分组与测验成绩总和关系')

plt.subplot(4, 2, 3)
sns.barplot(x='Hours Viewed Group', y='Reading Score', data=data, estimator=sum, ci=None)
plt.xlabel('观看时长分组')
plt.ylabel('阅读成绩总和')
plt.title('观看时长分组与阅读成绩总和关系')

plt.subplot(4, 2, 4)
sns.lineplot(x='Math Score Group', y='Reading Score', data=data, estimator=sum, ci=None)
plt.xlabel('数学成绩分组')
plt.ylabel('阅读成绩总和')
plt.title('数学成绩分组与阅读成绩总和关系')

plt.subplot(4, 2, 5)
sns.barplot(x='Hours Viewed Group', y='Math Score', data=data, estimator=sum, ci=None)
plt.xlabel('观看时长分组')
plt.ylabel('数学成绩总和')
plt.title('观看时长分组与阅读成绩总和关系')

plt.subplot(4, 2, 6)
sns.lineplot(x='Reading Score Group', y='Writing Score', data=data, estimator=sum, ci=None)
plt.xlabel('阅读成绩分组')
plt.ylabel('写作成绩总和')
plt.title('阅读成绩分组与写作成绩总和关系')


plt.subplot(4, 2, 7)
sns.barplot(x='Hours Viewed Group', y='Writing Score', data=data, estimator=sum, ci=None)
plt.xlabel('观看时长分组')
plt.ylabel('写作成绩总和')
plt.title('观看时长分组与阅读成绩总和关系')

plt.subplot(4, 2, 8)
sns.lineplot(x='Reading Score Group', y='Quiz Score', data=data, estimator=sum, ci=None)
plt.xlabel('阅读成绩分组')
plt.ylabel('测验成绩总和')
plt.title('阅读成绩分组与测验成绩总和关系')
plt.tight_layout()
plt.show()


