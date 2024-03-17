import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


'''
Student_ID：每个学生的唯一标识符。
Name：名字和姓氏的列表。
Age： 学生年龄
Gender：学生的性别，为“男性”或“女性”。
Education：学生达到的最高教育水平，从“高中”、“学士”或“硕士”。
Course：学生注册的课程类别，从“数学”、“写作”、“阅读”或“advance_skills”。
Course Name：学生正在修读的课程的具体名称，从相应类别的预定义课程名称中选择。
Hours Viewed：学生查看课程资料的总小时数。
Assignments Done：学生完成的作业数。
Quiz Score：学生在测验中获得的分数。
Lecture Pauses：学生在课程讲课期间停顿的次数。
Lecture Rewinds：学生倒带或重播课程讲义的次数。
Lecture Forwards：学生在课程讲座中快进的次数。
Math Score：学生数学成绩的分数
Writing Score：学生在写作任务中的表现分数
Reading Score：学生在阅读任务中的表现分数 
'''
#加载数据
df=pd.read_csv('../data11/text1.csv', encoding='GBK')
'''print(df.head())#输出默认头5行
print(df.info())    #输出movies_df的信息
print(df.describe())  #输出movies_df的基本统计量和分位数等值
#数据清洗
column_null_number = df.isnull().sum() #检查是否有缺省值
print('每列缺失值个数','\n',column_null_number)
df_nonull = df.dropna() #丢弃含空值的行或列
print('每列缺失值个数','\n',df_nonull.isnull().sum())
df_new = df_nonull.drop_duplicates(keep='first')#去除重复的项，并保留第一次出现的项
print(df_new.count())
print(df_new.head())        #输出默认头5行
print(df_new.describe())'''
#数据分析
plt.rcParams['font.sans-serif'] = ['SimHei']
# 绘制年龄分布直方图
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.figure(figsize=(10, 6))
plt.hist(df['Age'], bins=11, color='skyblue', edgecolor='black')  # 调整 bins 的数量以获得更适合的直方图
plt.xlabel('年龄')
plt.ylabel('人数')
plt.title('年龄分布直方图')
plt.grid(axis='y', alpha=0.75)
plt.show()

# 绘制性别比例饼图
gender_counts = df['Gender'].value_counts()
plt.figure(figsize=(6, 6))
plt.pie(gender_counts, labels=gender_counts.index, autopct='%1.1f%%', startangle=140)
plt.title('Gender Distribution')
plt.show()


# 统计每个学生的受教育水平的人数并绘制水平柱状图
edu_counts = df['Education'].value_counts()
plt.bar(edu_counts.index, edu_counts.values)
plt.xlabel('学生受教育程度')
plt.ylabel('人数')
plt.title('受教育程度柱状图')
plt.show()

"""# 创建散点图探索观看时长与测验成绩的关系
注：散点图过于密集
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Hours Viewed', y='Quiz Score', data=df)
plt.title('观看时长与测验成绩关系')
plt.xlabel('观看时长')
plt.ylabel('测验成绩')
plt.show()"""

"""# 创建热力图探索各变量之间的相关性
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
plt.title('变量相关性热力图')
plt.show()"""

#创建热力图探索各变量之间的相关性，除了id和姓名
plt.figure(figsize=(12, 8))
corr = df.iloc[:, 2:16].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title('各特征的相关热力图')
plt.show()


'''# 选择特定的科目（假设选择数学成绩）
注：图形大致相同，无法进行详细对比；说明数据比较集中
subject = 'Math Score'

# 使用seaborn绘制小提琴图
plt.figure(figsize=(12, 6))
sns.violinplot(x='Course', y=subject, data=df)
plt.title(f'不同课程类别学生在{subject}上的表现')
plt.xlabel('课程类别')
plt.ylabel(subject)
plt.show()

# 设置画布大小
plt.figure(figsize=(12, 8))
# 绘制小提琴图
sns.violinplot(x='Course', y='Hours Viewed', data=df)

# 添加标题和标签
plt.title('观看时长分布')
plt.xlabel('课程类名')
plt.ylabel('观看时长')

# 显示图形
plt.show()
'''
'''# 设置画布大小
# plt.figure(figsize=(12, 8))
# 绘制小提琴图
sns.violinplot(x='Course', y='Hours Viewed', hue='Education',data=df)

# 添加标题和标签
plt.title('不同课程类别的不同教育程度观看时长')
plt.xlabel('课程类名')
plt.ylabel('观看时长')

# 显示图形
plt.show()'''


'''
# 选择用于聚类的特征
features = ['Age', 'Hours Viewed', 'Quiz Score',
            'Pauses', 'Rewinds', 'Forwards','Assignments Done']

# 数据预处理
scaler = StandardScaler()
data_scaled = scaler.fit_transform(df[features])

# 选择K值
k = 3

# 训练K-means模型
kmeans = KMeans(n_clusters=k, random_state=0)
df['cluster'] = kmeans.fit_predict(data_scaled)

# 可视化聚类结果
plt.figure(figsize=(10, 6))
for cluster in df['cluster'].unique():
    plt.scatter(df.loc[df['cluster'] == cluster, 'Age'], df.loc[df['cluster'] == cluster, 'Pauses'], label=f'Cluster {cluster}')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s=300, c='red', label='Centroids')
plt.xlabel('Age')
plt.ylabel('Quiz Score')
plt.title('K-means Clustering')
plt.legend()
plt.show()

# 输出每个簇的统计信息
for cluster in df['cluster'].unique():
    cluster_data = df[df['cluster'] == cluster]
    print(f'Stats for Cluster {cluster}:')
    print(cluster_data.describe())

# 输出每个簇的平均值
cluster_means = df.groupby('cluster').mean()
print('Cluster Means:')
print(cluster_means)
'''


