import bisect
import os

import mplcursors as mplcursors
import pandas as pd
import pymysql
from matplotlib import pyplot as plt
import seaborn as sns

'''
#将MySQL文件转为csv文件
# 连接到 MySQL 数据库
connection = pymysql.connect(
    host='localhost',
    port=3306,
    user='root',
    password='123456',
    database='stuanalysis',
    charset='utf8'  # 设置字符集为utf8
)

# 要导出的表名列表
table_names = ['zt_class', 'zt_login', 'zt_stu_study_schedule','zt_stu_study_time_beihang','zt_student_num']

# 导出的 CSV 文件存储路径
output_directory = 'data11'

# 逐个表导出为 CSV 文件
for table_name in table_names:
    # 从数据库中读取数据到 DataFrame
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql(query, con=connection)

    # 设置 CSV 文件路径
    csv_file_path = f"{output_directory}{table_name}.csv"

    # 将 DataFrame 写入 CSV 文件
    df.to_csv(csv_file_path, index=False)

#手动修改转换后的文件名
# 关闭数据库连接
connection.close()'''
'''
login是“登陆信息表”，包含：用户ID，登录日期，登录具体时间
stuStudyTime是“当日人均学习时长表”，包含：用户ID，用户开始学习时间，用户结束学习时间
stuStudySchedule 是“学习行为活跃情况”，包含：用户ID，班次ID，学习类型，学习日期
class是"班级表"，包含：班次ID，班次名称
studentRegister是"学生注册表"，包含：用户ID,创建时间

学习行为类型;
page；仅点击网页
video:观看学习视频
Topic:完成课题
course;完成课程内容
assignment:完成课程作业
quiz:完成测试

分析思路：
从平台使用情况，学生习惯分析，课程质量评估三个方面
平台使用情况：每日登录次数，用户活跃度
习惯分析:分时段学习人数、学习行为次数、平均学习时长
课程质量评估:用户活跃度，学习行为次数，平均学习时长
'''

#数据清洗
# 原CSV文件夹路径
csv_folder_path = '../data1'

# 获取文件夹中所有CSV文件的文件名
csv_files = [file for file in os.listdir(csv_folder_path) if file.endswith('.csv')]

# 循环处理每个CSV文件
for file in csv_files:
    file_path = os.path.join(csv_folder_path, file)
    # 读取CSV文件为DataFrame
    df = pd.read_csv(file_path)

    # 检查缺失值并处理
    missing_values = df.isnull().sum().sum()
    if missing_values > 0:
        print(f'发现 {missing_values} 个缺失值')
        # 用平均值填充缺失值
        df.fillna(df.mean(), inplace=True)

    # 检查重复值并处理
    #duplicate_rows = df.duplicated().sum()
    #if duplicate_rows > 0:
     #   print(f'发现 {duplicate_rows} 个重复值')
        # 删除重复行
      #  df.drop_duplicates(inplace=True)

    # 处理异常值，例如移除超出3倍标准差的异常值
    std = df.std()
    mean = df.mean()
    outlier_threshold = 3
    df = df[~((df - mean).abs() > outlier_threshold * std).any(axis=1)]

    # 保存清洗后的结果为新的CSV文件
    cleaned_file_path = os.path.join('../data2', 'cleaned_' + file)
    df.to_csv(cleaned_file_path, index=False)

    print(f'文件 {file} 数据清洗完成')
'''
# 读取CSV文件
#df = pd.read_csv("../data22/cleaned_studentRegister.csv")
#print(df.count())

# 数据可视化
# 登录次数和人数柱状图：登录次数在5次及其以下的居多，达到了将近2000个，而5-10次的就减为了一半
zt_login = pd.read_csv("../data22/cleaned_login.csv")

user_info = zt_login['user_id'].value_counts().reset_index()
user_info.columns = ['user', 'times']

breakpoints = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60]
count_list = [0 for _ in range(len(breakpoints) + 1)]

for item in user_info['times']:
    level_index = bisect.bisect_right(breakpoints, item)
    count_list[level_index] += 1

breakpoints.insert(0, 0)

plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.rcParams['axes.unicode_minus'] = False
plt.bar([x + 5 for x in breakpoints], count_list)
plt.title("学生登录人次表")
plt.xlabel("登录次数")
plt.ylabel("登录人数")
plt.show()

# 按照登录日期统计每日登录人数：登录人数在2月15日达到了顶峰，2月5日次之。而2月的平均登录人数在739人次。
# 读取CSV文件
df = pd.read_csv('../data22/cleaned_login.csv')
# 按照登录日期统计每日登录人数
daily_login_count = df['login_date'].value_counts().sort_index()
# 绘制折线图
plt.figure(figsize=(12, 6))
sns.lineplot(x=daily_login_count.index, y=daily_login_count.values, marker='o')
plt.title('每日登录人数')
plt.xlabel('日期')
plt.ylabel('人数')
plt.xticks(rotation=45)
# 标注数据点
for x, y in zip(daily_login_count.index, daily_login_count.values):
    plt.text(x, y, f'{y}', ha='right', va='bottom')

            
                交互失败
                # 添加鼠标悬停交互
                cursor = mplcursors.cursor(hover=True)
                
                @cursor.connect("add")
                def on_add(sel):
                    date = sel.annotation.get_text().split('\n')[0].split(': ')[1]
                    count = sel.annotation.get_text().split('\n')[1].split(': ')[1]
                    sel.annotation.set_text(f"Date: {date}\nLogin Count: {count}")
            
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import bisect

# 读取原始数据
data = pd.read_csv("../data22/cleaned_stuStudyTime.csv")

# 创建空字典存储学习时间和学习次数
study_time_info = {}
study_count_info = {}

# 遍历原始数据，计算每个用户的学习时间和次数
for i in range(len(data["user_id"])):
    user_id = data["user_id"][i]
    if user_id not in study_time_info:
        last_visit_time = pd.to_datetime(data["last_visit_time"][i])
        visit_time = pd.to_datetime(data["visit_time"][i])
        delta_minutes = (last_visit_time - visit_time).total_seconds() / 60  # 将秒数转换为分钟
        study_time_info[user_id] = delta_minutes
        study_count_info[user_id] = 1
    else:
        last_visit_time = pd.to_datetime(data["last_visit_time"][i])
        visit_time = pd.to_datetime(data["visit_time"][i])
        delta_minutes = (last_visit_time - visit_time).total_seconds() / 60  # 将秒数转换为分钟
        study_time_info[user_id] += delta_minutes
        study_count_info[user_id] += 1

# 计算每个用户的平均学习时间
user_id_list = []
count_list = []
total_time_list = []
avg_time_list = []

for key, value in study_time_info.items():
    user_id_list.append(key)
    total_time_list.append(value)
    count = study_count_info[key]
    count_list.append(count)
    avg_time_list.append(value / count)

# 创建DataFrame并保存为CSV文件
result = pd.DataFrame({
    "user_id": user_id_list,
    "total_time": total_time_list,
    "count": count_list,
    "avg_time": avg_time_list
})
result.to_csv("用户平均登录时间表.csv", index=False)

# 对用户平均登录时间进行可视化
avg_time_count = {}

for item in avg_time_list:
    if int(item) not in avg_time_count:
        avg_time_count[int(item)] = 1
    else:
        avg_time_count[int(item)] += 1

avg_time_list = list(avg_time_count.keys())
avg_time_list.sort()

count_list = [avg_time_count[x] for x in avg_time_list]

break_points = [60, 120, 180, 240, 300, 360, 420, 480, 540, 600, 660, 720, 780, 840, 900, 960]
group_count = [0 for _ in break_points]

for item in avg_time_list:
    level_index = bisect.bisect_right(break_points, item)
    group_count[level_index] += 1

# 绘制可视化图表
plt.rcParams['font.sans-serif'] = [u'SimHei']
plt.title("学生平均登录时间时间与对应人数")
plt.xlabel("平均时长(分钟)")
plt.ylabel("人数")
plt.plot(break_points, group_count)
plt.show()
'''