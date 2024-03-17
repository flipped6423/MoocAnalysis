import ax as ax
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DayLocator, DateFormatter

#用户习惯分析
'''
# 读取CSV文件
login_df = pd.read_csv('../data22/cleaned_login.csv')
stu_study_time_df = pd.read_csv('../data22/cleaned_stuStudyTime.csv')
stu_study_schedule_df = pd.read_csv('../data22/cleaned_stuStudySchedule.csv')
class_df = pd.read_csv('../data22/cleaned_class.csv')
student_register_df = pd.read_csv('../data22/cleaned_studentRegister.csv')

# 合并数据
merged_df = login_df.merge(stu_study_time_df, on='user_id')\
    .merge(stu_study_schedule_df, on='user_id')\
    .merge(class_df, on='class_id')\
    .merge(student_register_df, on='user_id')
print(merged_df)
merged_df.to_csv('../data22/merged_data.csv', index=False)
'''

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = [u'SimHei']

#1.分时段学习人数

# 读取 CSV 文件
df = pd.read_csv('../data22/cleaned_stuStudyTime.csv')

# 将开始学习时间和最后学习时间转换为 datetime 类型
df['开始学习时间'] = pd.to_datetime(df['visit_time'])
df['最后学习时间'] = pd.to_datetime(df['last_visit_time'])


# 计算每个用户的学习总时长
df['学习时长'] = df['最后学习时间'] - df['开始学习时间']
print(df['学习时长'].count())

# 按用户ID分组并计算总学习时长
total_study_time = df.groupby('user_id')['学习时长'].sum().reset_index()

# 按照学习总时长降序排列
sorted_total_study_time = total_study_time.sort_values(by='学习时长', ascending=False)

print(sorted_total_study_time)
'''
#显示结果
1098
      user_id            学习时长
141  17682951 8 days 23:22:06
94   15258984 5 days 14:21:44
109  15259322 5 days 11:47:03
33   13049829 4 days 20:58:36
140  17639401 4 days 17:31:22
..        ...             ...
2     8084284 0 days 00:28:49
115  15932449 0 days 00:19:59
179  19939998 0 days 00:14:20
12   11995893 0 days 00:06:27
214  20388108 0 days 00:00:45

[239 rows x 2 columns]
'''

# 计算分时段的学习人数
df['学习时长'] = (df['最后学习时间'] - df['开始学习时间']).dt.total_seconds() / 3600  # 学习时长（小时）

# 根据学习时长进行分组统计
bins = [0, 1, 2, 4, 8, 24, 168]  # 分时段边界（小时）
labels = ['<1', '1-2', '2-4', '4-8', '8-24', '>24']  # 分时段标签
df['学习时长段'] = pd.cut(df['学习时长'], bins=bins, labels=labels)

# 统计每个分时段的学习人数
study_time_counts = df['学习时长段'].value_counts().sort_index()

# 可视化
plt.figure(figsize=(10, 6))
sns.barplot(x=study_time_counts.index, y=study_time_counts.values, palette="viridis")
plt.title('不同学习时长段的学习人数统计')
plt.xlabel('学习时长段（小时）')
plt.ylabel('学习人数')
plt.show()
#课程质量评估:
# 用户活跃度：通过查询zt_stu_study_schedule表来完成每日活跃学生人数的统计分析，这里设定每日至少进行3次学习行为的用户为活跃用户

# 读取CSV文件到DataFrame
df = pd.read_csv('../data22/cleaned_stuStudySchedule.csv')
# 按学习日期和用户ID进行分组，并计算每日活跃学生人数
active_students_per_day = df.groupby(['updated_date'])['user_id'].nunique().reset_index()
active_students_per_day.columns = ['学习日期', '活跃人数']
# 打印每日活跃学生人数
print(active_students_per_day)
# 可视化
active_students_per_day['学习日期'] = active_students_per_day['学习日期'].astype(str)
plt.plot(active_students_per_day['学习日期'], active_students_per_day['活跃人数'], marker='o')
plt.title('每日活跃学生人数')
plt.xlabel('日期')
plt.ylabel('活跃人数')
x_ticks = active_students_per_day['学习日期'][::5]  # 每隔两天显示一个日期
plt.xticks(x_ticks, rotation=45)
plt.grid(True)

plt.show()

import pandas as pd

# 读取CSV文件
df = pd.read_csv('../data22/cleaned_stuStudySchedule.csv')

# 按照学习日期和学习类型进行分组，并计算每日总学习类型次数
study_counts = df.groupby(['updated_date', 'content_type']).size().reset_index(name='学习次数')

# 使用透视表将数据重塑为适合绘图的格式
pivot_table = study_counts.pivot(index='updated_date', columns='content_type', values='学习次数').fillna(0)

# 绘制叠加柱状图
ax=pivot_table.plot(kind='bar', stacked=True, figsize=(12, 6))
# 设置x轴日期格式
ax.xaxis.set_major_locator(DayLocator(interval=5))  # 设置日期间隔为5天
ax.xaxis.set_major_formatter(DateFormatter("%Y-%m-%d"))  # 设置日期显示格式为年-月-日
plt.title('每日学习类型次数')
plt.xlabel('日期')
plt.ylabel('学习次数')
plt.legend(title='学习类型')
plt.show()
