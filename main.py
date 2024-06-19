import torch
from memory_profiler import profile 
import torch.optim as optim
import torch.nn as nn
import logging
import datetime  # 导入 datetime 模块以获取当前时间戳
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader  # 使用新的 DataLoader
from torch.utils.tensorboard import SummaryWriter
from data import load_data  # 导入数据加载函数
from utility import process_and_convert_to_tensors  # 导入数据处理和转换函数
from GNN import GNN, DTA  # 导入 GNN 和 DTA 模型
import gc 

 
# # 获取当前时间戳
timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

# 检查 MPS 是否可用
device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

# 设置日志记录
logging.basicConfig(filename=f'training_{timestamp}.log', level=logging.INFO, format='%(asctime)s:%(levelname)s:%(message)s')

# 定义批量数据生成器
def data_generator(data, batch_size=10):  # 减小 batch size
    import gc  # 导入垃圾收集模块
    data_size = len(data)
    for start_idx in range(0, data_size, batch_size):
        end_idx = min(start_idx + batch_size, data_size)
        batch_data = data.iloc[start_idx:end_idx]
        dataset = []
        for _, row in batch_data.iterrows():
            smiles, sequence, affinity = row['Compound_SMILES'], row['Protein_Sequence'], row['Affinity']
            
            # 调试信息
            logging.info(f"Processing SMILES: {smiles}, Sequence: {sequence}, Affinity: {affinity}")
            
            # 处理和转换数据
            mol_graph_tensors, ngram_graphs_tensors, affinity = process_and_convert_to_tensors(smiles, sequence, affinity)
            
            # 检查数据有效性
            if mol_graph_tensors[0] is None or any(ng[0] is None for ng in ngram_graphs_tensors):
                logging.warning(f"Invalid data for SMILES: {smiles}, Sequence: {sequence}")
                continue  # 跳过无效数据
            
            mol_atoms, mol_adj = mol_graph_tensors  # 提取分子图张量
            # 提取 N-Gram 图张量
            ngram_atoms, ngram_adj = zip(*[(ng[0], ng[1]) for ng in ngram_graphs_tensors if ng[0] is not None])
            
            # 创建药物数据对象
            drug_data = Data(
                x=mol_atoms,
                edge_index=mol_adj.nonzero().t()
            )

            protein_data_list = []  # 初始化蛋白质数据列表
            for atoms, adj in zip(ngram_atoms, ngram_adj):  # 遍历 N-Gram 图
                protein_data_list.append(Data(
                    x=atoms,
                    edge_index=adj.nonzero().t()
                ))
            
            # 将药物数据、蛋白质数据列表和标签添加到数据集
            dataset.append((drug_data, protein_data_list, torch.tensor([affinity], dtype=torch.float)))

        yield DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)
        gc.collect()  # 清理垃圾
             # 清理批次数据，释放内存
      # 强制垃圾收集

# 自定义 collate_fn，用于将批次数据组合在一起
def collate_fn(batch):
    drug_data_list, protein_data_lists, affinities = zip(*batch)  # 解压批次数据
    drug_batch = Batch.from_data_list(drug_data_list)  # 创建药物批次
    protein_batches = [Batch.from_data_list(protein_data) for protein_data in zip(*protein_data_lists)]  # 创建蛋白质批次
    return drug_batch, protein_batches, torch.stack(affinities)  # 返回药物批次、蛋白质批次和标签


@profile  # 添加内存监控
# 定义训练函数
def train(model, data, epochs=1000, lr=0.0005, batch_size=10):  # 减小 batch size
     # 导入垃圾收集模块
    writer = SummaryWriter(log_dir=f'runs/DTA_experiment_{timestamp}')  # 初始化 TensorBoard 编写器，添加时间戳
    optimizer = optim.Adam(model.parameters(), lr=lr)  # 初始化优化器
    criterion = nn.MSELoss()  # 初始化损失函数

    global_batch_count = 0  # 用于记录全局批次计数

    for epoch in range(epochs):  # 迭代训练
        model.train()  # 设定模型为训练模式
        total_loss = 0  # 初始化总损失
        data_gen = data_generator(data, batch_size=batch_size)
        batch_count = 0
        for data_loader in data_gen:
            for drug_batch, protein_batches, affinities in data_loader:  # 遍历数据加载器中的批次
                optimizer.zero_grad()  # 清除梯度
                drug_batch = drug_batch.to(device).detach()
                protein_batches = [batch.to(device).detach() for batch in protein_batches]
                affinities = affinities.to(device).detach()
                
                output = model(drug_batch, protein_batches)  # 通过模型计算输出
                loss = criterion(output, affinities)  # 计算损失
                loss.backward()  # 反向传播
                optimizer.step()  # 更新参数
                total_loss += loss.item()  # 累加损失
                
                batch_count += 1
                global_batch_count += 1
                avg_loss = total_loss / batch_count  # 计算平均损失
               

#Scalars：展示训练过程中的标量指标，比如训练误差、验证误差、学习率等。
#Graphs：展示计算图，可以看到每一层的输入输出，以及参数的维度和数值。
#Distributions：展示数据分布情况，可以查看权重、梯度、激活值等的分布情况，有助于诊断过拟合或欠拟合等问题。
#Histograms：展示数据分布的直方图，类似于Distributions，但更详细。
#Images：展示图像数据，可以查看输入图片、卷积层的输出等。
#Projector：展示高维数据的嵌入情况，可以对数据进行降维可视化。


                writer.add_scalar('Loss/train', avg_loss, global_batch_count)  # 记录损失到 TensorBoardtensorboard 观察
             

                writer.flush()  # 确保数据写入文件
                logging.info(f'Processed batch {batch_count} in epoch {epoch + 1}')
                print(f'Processed batch {batch_count} in epoch {epoch + 1}')
                gc.collect()
            
        avg_loss = total_loss / batch_count  # 计算平均损失，使用 batch_count 而不是 len(data)
        logging.info(f'Epoch {epoch + 1}, Loss: {avg_loss}')  # 记录日志
        print(f'Epoch {epoch + 1}, Loss: {avg_loss}')  # 打印当前轮次的损失
            
   

    writer.close()  # 确保写入所有数据到 TensorBoard

if __name__ == "__main__":
    # 获取当前时间戳
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # 加载数据
    df = load_data('data/kiba.txt')  # 从指定路径加载数据
    logging.info("Data loaded successfully")
    
    # 初始化药物 GNN 和蛋白质 GNN
    drug_gnn = GNN(num_features=5, hidden_dim=128, output_dim=128).to(device)  # 初始化药物 GNN 模型，并移动到设备（MPS 或 CPU）
    protein_gnn = GNN(num_features=5, hidden_dim=128, output_dim=128).to(device)  # 初始化蛋白质 GNN 模型，并移动到设备（MPS 或 CPU）
    logging.info("GNN models initialized successfully")
    
    # 初始化 DTA 模型
    model = DTA(drug_gnn, protein_gnn, hidden_dim=128, num_conformer_blocks=3).to(device)  # 初始化 DTA 模型，并移动到设备（MPS 或 CPU）
    logging.info("DTA model initialized successfully")
    
    # 开始训练
    train(model, df, epochs=1000, lr=0.0005, batch_size=10)  # 使用数据加载器和指定的超参数开始训练
    logging.info("Training started")