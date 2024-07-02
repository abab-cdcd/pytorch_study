import csv

# 定义函数用于修改路径格式
def modify_path(original_path):
    # 将原始路径中的部分替换为目标路径
    modified_path = original_path.replace("C:\\Users\\19124\\Desktop\\final_works\\Prototypical\\", "C:\\Users\\19124\\Desktop\\study\\pytorch_study\\")
    return modified_path



#读取CSV文件
with open('val.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    data = list(reader)
# 修改路径列
for row in data:
    row[0] = modify_path(row[0])

#写回CSV文件
with open('val.csv', 'w', newline='') as file:
    writer = csv.writer(file)
    writer.writerows(data)
