import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.nn as pyg_nn
import gc  # 导入垃圾回收模块


# 定义一个图卷积层，结合卷积操作和线性变换
class LEConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(LEConvLayer, self).__init__()
        # 使用 PyG 的 GCNConv 进行图卷积操作
        self.conv = pyg_nn.GCNConv(in_channels, out_channels)
        # 线性变换层
        self.lin = nn.Linear(out_channels, out_channels)

    def forward(self, x, edge_index):
        # 执行图卷积操作
        x = self.conv(x, edge_index)
        # 执行线性变换
        x = self.lin(x)
        # 使用 ReLU 激活函数
        return F.relu(x)

# 定义一个通用的图神经网络（GNN）模块，用于药物分子图和蛋白质 N-Gram 图的处理
class GNN(nn.Module):
    def __init__(self, num_features, hidden_dim, output_dim):
        super(GNN, self).__init__()
        # 三层 LEConvLayer，分别用于特征提取和降维
        self.layer1 = LEConvLayer(num_features, hidden_dim)
        self.layer2 = LEConvLayer(hidden_dim, hidden_dim)
        self.layer3 = LEConvLayer(hidden_dim, output_dim)
        # 全局平均池化层，用于将节点特征聚合为图特征
        self.pool = pyg_nn.global_mean_pool

    def forward(self, data):
        # 获取节点特征和边索引
        x, edge_index = data.x, data.edge_index
        # 三层图卷积操作
        x = self.layer1(x, edge_index)
        x = self.layer2(x, edge_index)
        x = self.layer3(x, edge_index)
        # 全局平均池化，将节点特征聚合为图特征
        x = self.pool(x, data.batch)
        return x

# 定义一个 Conformer 块，用于处理蛋白质 N-Gram 图的特征
class ConformerBlock(nn.Module):
    def __init__(self, input_dim, num_heads=4, dropout=0.1):
        super(ConformerBlock, self).__init__()
        # 多头自注意力机制
        self.attention = nn.MultiheadAttention(input_dim, num_heads, dropout=dropout)
        # 前馈神经网络
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, 4 * input_dim),
            nn.ReLU(),
            nn.Linear(4 * input_dim, input_dim)
        )
        # 层归一化
        self.norm1 = nn.LayerNorm(input_dim)
        self.norm2 = nn.LayerNorm(input_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自注意力机制
        attn_output, _ = self.attention(x, x, x)
        # 层归一化和残差连接
        x = self.norm1(x + self.dropout(attn_output))
        # 前馈神经网络
        ffn_output = self.ffn(x)
        # 层归一化和残差连接
        x = self.norm2(x + self.dropout(ffn_output))
        return x

# 定义药物-靶点亲和性预测模型
class DTA(nn.Module):
    def __init__(self, drug_gnn, protein_gnn, hidden_dim, num_conformer_blocks=3):
        super(DTA, self).__init__()
        self.drug_gnn = drug_gnn  # 药物 GNN 模型
        self.protein_gnn = protein_gnn  # 蛋白质 GNN 模型
        # 多个 Conformer 块
        self.conformer_blocks = nn.ModuleList([ConformerBlock(hidden_dim, num_heads=4) for _ in range(num_conformer_blocks)])
        # 注意力权重参数
        self.attention_weights = nn.Parameter(torch.randn(hidden_dim))
        # 全连接层
        self.fc1_drug = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_drug = nn.Linear(hidden_dim, hidden_dim)
        self.fc1_protein = nn.Linear(hidden_dim, hidden_dim)
        self.fc2_protein = nn.Linear(hidden_dim, hidden_dim)
        self.fc1 = nn.Linear(hidden_dim * 2, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 1)

    def forward(self, drug_data, protein_data_list):
        # 药物分子图嵌入
        drug_embedding = self.drug_gnn(drug_data)  # 使用药物 GNN 提取嵌入
        drug_embedding = F.relu(self.fc1_drug(drug_embedding))  # 第一个全连接层，ReLU 激活
        drug_embedding = F.relu(self.fc2_drug(drug_embedding))  # 第二个全连接层，ReLU 激活

        # 蛋白质 N-Gram 图嵌入并通过 Conformer 块
        protein_embeddings = []
        for protein_data in protein_data_list:
            protein_embedding = self.protein_gnn(protein_data)  # 使用蛋白质 GNN 提取嵌入
            protein_embedding = F.relu(self.fc1_protein(protein_embedding))  # 第一个全连接层，ReLU 激活
            protein_embedding = F.relu(self.fc2_protein(protein_embedding))  # 第二个全连接层，ReLU 激活
            protein_embeddings.append(protein_embedding)
        # 堆叠所有 N-Gram 嵌入
        protein_embeddings = torch.stack(protein_embeddings, dim=1)

        # 通过多个 Conformer 块处理 N-Gram 嵌入
        for conformer_block in self.conformer_blocks:
            protein_embeddings = conformer_block(protein_embeddings)
        
        # 加权求和注意力机制
        attn_scores = torch.matmul(protein_embeddings, self.attention_weights)  # 计算注意力得分
        attn_weights = F.softmax(attn_scores, dim=1)  # 计算注意力权重
        protein_embedding = torch.sum(attn_weights.unsqueeze(-1) * protein_embeddings, dim=1)  # 计算加权求和的蛋白质嵌入

        # 将药物嵌入加入到每个 N-Gram 嵌入的注意力权重中
        attn_scores_with_drug = torch.matmul(protein_embeddings + drug_embedding.unsqueeze(1), self.attention_weights)  # 计算包含药物嵌入的注意力得分
        attn_weights_with_drug = F.softmax(attn_scores_with_drug, dim=1)  # 计算包含药物嵌入的注意力权重
        protein_embedding_with_drug = torch.sum(attn_weights_with_drug.unsqueeze(-1) * protein_embeddings, dim=1)  # 计算包含药物嵌入的加权求和的蛋白质嵌入

        # 合并药物和蛋白质嵌入
        x = torch.cat((drug_embedding, protein_embedding_with_drug), dim=1)  # 合并药物和蛋白质嵌入
        # 通过全连接层进行预测
        x = F.relu(self.fc1(x))  # 第一个全连接层，ReLU 激活
        x = F.relu(self.fc2(x))  # 第二个全连接层，ReLU 激活
        x = self.fc3(x)  # 最终预测层
        return x

if __name__ == "__main__":
    # 初始化药物 GNN 和蛋白质 GNN
    drug_gnn = GNN(num_features=5, hidden_dim=128, output_dim=128)  # 初始化药物 GNN
    protein_gnn = GNN(num_features=5, hidden_dim=128, output_dim=128)  # 初始化蛋白质 GNN
    # 初始化 DTA 模型
    model = DTA(drug_gnn, protein_gnn, hidden_dim=128, num_conformer_blocks=3)  # 初始化 DTA 模型
    print(model)  # 打印模型结构