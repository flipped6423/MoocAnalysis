
import openpyxl
'''# 打开Excel文件
workbook = openpyxl.load_workbook('data/219211168网页设计_43493777_21软03.xlsx')

# 选择要操作的工作表
sheet = workbook.active

# 获取A1单元格的数据
merged_cell_value = sheet['A1'].value

# 如果A1单元格是合并的单元格，获取合并单元格的起始单元格
if sheet.merged_cells.ranges:
    for merged_cell in sheet.merged_cells.ranges:
        if sheet['A1'] in merged_cell:
            start_cell = merged_cell.coord.split(':')[0]
            break

# 获取合并单元格的起始单元格所在列的数据
column_data = [cell.value for cell in sheet[start_cell[0] + '1:' + start_cell[0] + str(sheet.max_row)]]

print(column_data)

# 关闭Excel文件
workbook.close()'''


