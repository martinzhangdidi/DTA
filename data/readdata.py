import pandas as pd

# 定义文件路径
file_path = 'data/kiba.txt'

# 定义列名
columns = ['Compound_ID', 'Protein_ID', 'Compound_SMILES', 'Protein_Sequence', 'Affinity']

# 初始化一个空列表来存储数据
data = []

# 读取文件
with open(file_path, 'r') as file:
    for line in file:
        # 去掉行尾的换行符，并按照空格分割
        parts = line.strip().split(' ')
        # 将分割后的部分添加到数据列表中
        data.append(parts)

# 将数据列表转换为DataFrame
df = pd.DataFrame(data, columns=columns)

# 打印前几行数据进行验证
print(df.head())

# 如果需要保存为新的 CSV 文件
##df.to_csv('kiba_data.csv', index=False)