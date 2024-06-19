from rdkit import Chem
from rdkit.Chem import rdmolops
import torch
import numpy as np

# 处理分子图，将 SMILES 字符串转换为分子图表示
def mol_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)  # 从 SMILES 字符串生成分子对象
    if mol is None:
        return None
    atom_features = []
    for atom in mol.GetAtoms():  # 遍历分子中的每个原子，提取特征
        atom_features.append([
            atom.GetAtomicNum(),         # 原子序数
            atom.GetDegree(),            # 原子的键数
            atom.GetTotalNumHs(),        # 原子连接的氢原子数
            atom.GetImplicitValence(),   # 隐含价
            atom.GetIsAromatic()         # 是否芳香性
        ])
    adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)  # 获取分子的邻接矩阵
    return {'atoms': atom_features, 'adjacency_matrix': adjacency_matrix}

# 将蛋白质序列转换为 N-Gram 序列
def sequence_to_ngram(sequence, n=3):
    ngrams = [sequence[i:i+n] for i in range(len(sequence)-n+1)]  # 生成长度为 n 的 N-Gram 序列
    return ngrams

# 处理蛋白质序列，将其转换为 N-Gram 分子图表示
def process_sequence(sequence, max_length=1000, ngram_size=3):
    if len(sequence) > max_length:  # 如果序列长度超过最大长度，进行截断；否则进行填充
        sequence = sequence[:max_length]
    else:
        sequence = sequence.ljust(max_length, '0')
    
    ngram_graphs = []
    ngrams = sequence_to_ngram(sequence, ngram_size)
    for ngram in ngrams:  # 遍历每个 N-Gram 序列，生成分子对象并提取特征
        mol = Chem.MolFromSequence(ngram)  # 使用 Chem.MolFromSequence 处理蛋白质片段
        if mol:
            atom_features = []
            for atom in mol.GetAtoms():
                atom_features.append([
                    atom.GetAtomicNum(),        # 原子序数
                    atom.GetDegree(),           # 原子的键数
                    atom.GetTotalNumHs(),       # 原子连接的氢原子数
                    atom.GetImplicitValence(),  # 隐含价
                    atom.GetIsAromatic()        # 是否芳香性
                ])
            adjacency_matrix = rdmolops.GetAdjacencyMatrix(mol)  # 获取分子的邻接矩阵
            ngram_graphs.append({'atoms': atom_features, 'adjacency_matrix': adjacency_matrix})
    return ngram_graphs

# 处理 SMILES 字符串和蛋白质序列，生成分子图和 N-Gram 图
def process_data(smiles, sequence, affinity, max_length=1000, ngram_size=3):
    mol_graph = mol_to_graph(smiles)  # 处理分子图
    ngram_graphs = process_sequence(sequence, max_length, ngram_size)  # 处理 N-Gram 图
    affinity = float(affinity)  # 将标签转换为浮点数
    return mol_graph, ngram_graphs, affinity

# 将图转换为张量格式
def tensors_from_graph(graph):
    atoms = torch.tensor(graph['atoms'], dtype=torch.float)  # 将原子特征转换为张量
    adjacency_matrix = torch.tensor(graph['adjacency_matrix'], dtype=torch.float)  # 将邻接矩阵转换为张量
    return atoms, adjacency_matrix

# 处理并转换 SMILES 字符串和蛋白质序列为张量格式
def process_and_convert_to_tensors(smiles, sequence, affinity, max_length=1000, ngram_size=3):
    mol_graph, ngram_graphs, affinity = process_data(smiles, sequence, affinity, max_length, ngram_size)  # 处理数据，生成分子图和 N-Gram 图
    if mol_graph:
        mol_atoms, mol_adj = tensors_from_graph(mol_graph)  # 将分子图转换为张量格式
    else:
        mol_atoms, mol_adj = None, None
    
    ngram_atoms_adj = []
    for ngram_graph in ngram_graphs:  # 将 N-Gram 图转换为张量格式
        if ngram_graph:
            atoms, adj = tensors_from_graph(ngram_graph)
            ngram_atoms_adj.append((atoms, adj))
        else:
            ngram_atoms_adj.append((None, None))
       # 清理中间结果，释放内存
    del mol_graph, ngram_graphs
    return (mol_atoms, mol_adj), ngram_atoms_adj, affinity

if __name__ == "__main__":
    sample_smiles = "CCO"
    sample_sequence = "MTVKTEAAKGTL"
    sample_affinity = "11.1"
    mol_graph_tensors, ngram_graphs_tensors, affinity = process_and_convert_to_tensors(sample_smiles, sample_sequence, sample_affinity)
    print("Molecular Graph Tensors:", mol_graph_tensors)
    print("N-Gram Graphs Tensors:", ngram_graphs_tensors)
    print("Affinity:", affinity)