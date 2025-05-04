import torch.nn as nn
import torch
from torch_geometric.data import Data, Batch
from torchinfo import summary
from mamba_ssm import Mamba
from torch_geometric.nn import GATConv


class GATLayer(nn.Module):
    def __init__(self, model_dim, heads, edge_index):
        super(GATLayer, self).__init__()
        self.gat_conv = GATConv(in_channels=model_dim, out_channels=model_dim, heads=heads,concat=False)
        self.ln = nn.LayerNorm(model_dim)
        self.edge_index = edge_index

    def forward(self, x):
        batch_size, in_steps, num_nodes, model_dim = x.shape
        input_data = x.view(-1, num_nodes, model_dim)  # (batch_size * in_steps, num_nodes, model_dim)

        # 创建图数据列表
        data_list = [Data(x=data, edge_index=self.edge_index) for data in input_data]

        input_data = Batch.from_data_list(data_list)

        output, attention_weights = self.gat_conv(input_data.x, input_data.edge_index, return_attention_weights = True)

        output = output.view(batch_size, in_steps, num_nodes, model_dim)  # 还原为 (batch_size, in_steps, num_nodes, model_dim)

        return  self.ln(x + output)



class MambaLayer(nn.Module):
    def __init__(self, model_dim,dim):
        super().__init__()
        # 初始化 Mamba 模块，注意不要传递未定义的参数
        self.mamba = Mamba(d_model=model_dim)
        self.ln = nn.LayerNorm(model_dim)
        self.dim = dim

    def forward(self, x):
        # 将指定的维度移动到第二个维度（batch 之后）
        x = x.transpose(self.dim, 1)
        # 获取输入的形状
        x_shape = x.shape
        batch_size, seq_len = x_shape[0], x_shape[1]
        rest_dims = x_shape[2:-1]
        model_dim = x_shape[-1]
        # 将 rest_dims 展平到 batch 维度
        x = x.reshape(-1, seq_len, model_dim)
        residual = x
        out = self.mamba(x)
        out = self.ln(residual + out)
        # 恢复原始形状
        out = out.reshape(x_shape)
        # 将维度顺序转换回去
        out = out.transpose(1, self.dim)
        return out.contiguous()


class SpatialTemporalAttention(nn.Module):
    def __init__(self, num_layers,model_dim, edge_index,spatial_attention,temporal_attention , gat ):
        super().__init__()
        self.attn_layers_t = nn.ModuleList()
        for i in range(num_layers):
            self.attn_layers_t.append(GATLayer(model_dim,4, edge_index)) if gat and temporal_attention else nn.Identity()
            self.attn_layers_t.append(MambaLayer(model_dim,1)) if temporal_attention else nn.Identity()

        self.attn_layers_s = nn.ModuleList()
        for i in range(num_layers):
            self.attn_layers_s.append(GATLayer(model_dim,4, edge_index)) if gat and spatial_attention else nn.Identity()
            self.attn_layers_s.append(MambaLayer(model_dim,2)) if spatial_attention else nn.Identity()

    def forward(self, x):
        # (batch_size, in_steps, num_nodes, model_dim)
        # 时间维度上的 Mamba 注意力层
        for layer in self.attn_layers_t:
            x = layer(x)

        # 空间维度上的 Mamba 注意力层
        for layer in self.attn_layers_s:
            x = layer(x)

        return x

class GAMMANet(nn.Module):
    def __init__(
        self,
        num_nodes,
        in_steps=12,
        out_steps=12,
        steps_per_day=288,
        input_dim=3,
        output_dim=1,
        input_embedding_dim=24,
        tod_embedding_dim=24,
        dow_embedding_dim=24,
        spatial_embedding_dim=0,
        adaptive_embedding_dim=80,
        feed_forward_dim=256,
        num_heads=4,  # 虽然不再需要 num_heads，但保留以保持接口一致性
        num_layers=3,
        dropout=0.1,
        use_mixed_proj=True,
        edge_index_path = None,
        spatial_attention = True,
        temporal_attention = True,
        gat= True
    ):
        super().__init__()

        self.num_nodes = num_nodes
        self.in_steps = in_steps
        self.out_steps = out_steps
        self.steps_per_day = steps_per_day
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.input_embedding_dim = input_embedding_dim
        self.tod_embedding_dim = tod_embedding_dim
        self.dow_embedding_dim = dow_embedding_dim
        self.spatial_embedding_dim = spatial_embedding_dim # alway 0
        self.adaptive_embedding_dim = adaptive_embedding_dim
        self.model_dim = (
            input_embedding_dim
            + tod_embedding_dim
            + dow_embedding_dim
            + spatial_embedding_dim
            + adaptive_embedding_dim
        )
        self.num_layers = num_layers
        self.use_mixed_proj = use_mixed_proj
        self.edge_index = torch.load(edge_index_path).cuda()

        self.input_proj = nn.Linear(input_dim, input_embedding_dim)
        if tod_embedding_dim > 0:
            self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        if dow_embedding_dim > 0:
            self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        if spatial_embedding_dim > 0:
            self.node_emb = nn.Parameter(
                torch.empty(self.num_nodes, self.spatial_embedding_dim)
            )
            nn.init.xavier_uniform_(self.node_emb)
        if adaptive_embedding_dim > 0:
            self.adaptive_embedding = nn.Parameter(
                torch.empty(in_steps, num_nodes, adaptive_embedding_dim)
            )
            nn.init.xavier_uniform_(self.adaptive_embedding)

        self.spatial_temporal_attention = SpatialTemporalAttention(
            num_layers=self.num_layers,
            model_dim=self.model_dim,
            edge_index = self.edge_index,
            spatial_attention = spatial_attention,
            temporal_attention = temporal_attention,
            gat = gat
        )

        if use_mixed_proj:
            self.output_proj = nn.Linear(
                in_steps * self.model_dim, out_steps * output_dim
            )
        else:
            self.temporal_proj = nn.Linear(in_steps, out_steps)
            self.output_proj = nn.Linear(self.model_dim, self.output_dim)


    def forward(self, x):
        # x: (batch_size, in_steps, num_nodes, input_dim+tod+dow=3)
        batch_size = x.shape[0]

        if self.tod_embedding_dim > 0:
            tod = x[..., 1] # 提取 time of date
        if self.dow_embedding_dim > 0:
            dow = x[..., 2] # 提取 day of week
        x = x[..., : self.input_dim] # 车流数据

        x = self.input_proj(x)  # (batch_size, in_steps, num_nodes, input_embedding_dim)
        features = [x]
        if self.tod_embedding_dim > 0:
            tod_emb = self.tod_embedding(
                (tod * self.steps_per_day).long()
            )  # (batch_size, in_steps, num_nodes, tod_embedding_dim)
            features.append(tod_emb)
        if self.dow_embedding_dim > 0:
            dow_emb = self.dow_embedding(
                dow.long()
            )  # (batch_size, in_steps, num_nodes, dow_embedding_dim)
            features.append(dow_emb)
        if self.spatial_embedding_dim > 0:
            spatial_emb = self.node_emb.expand(
                batch_size, self.in_steps, *self.node_emb.shape
            )
            features.append(spatial_emb)
        if self.adaptive_embedding_dim > 0:
            adp_emb = self.adaptive_embedding.expand(
                size=(batch_size, *self.adaptive_embedding.shape)
            )
            features.append(adp_emb)
        x = torch.cat(features, dim=-1)  # (batch_size, in_steps, num_nodes, model_dim)

        x = self.spatial_temporal_attention(x)

        # (batch_size, in_steps, num_nodes, model_dim)

        if self.use_mixed_proj:
            out = x.transpose(1, 2)  # (batch_size, num_nodes, in_steps, model_dim)
            out = out.reshape(
                batch_size, self.num_nodes, self.in_steps * self.model_dim
            )
            out = self.output_proj(out).view(
                batch_size, self.num_nodes, self.out_steps, self.output_dim
            )
            out = out.transpose(1, 2)  # (batch_size, out_steps, num_nodes, output_dim)
        else:
            out = x.transpose(1, 3)  # (batch_size, model_dim, num_nodes, in_steps)
            out = self.temporal_proj(
                out
            )  # (batch_size, model_dim, num_nodes, out_steps)
            out = self.output_proj(
                out.transpose(1, 3)
            )  # (batch_size, out_steps, num_nodes, output_dim)

        return out

if __name__ == "__main__":
    model = GAMMANet(207, 12, 12)
    summary(model, [64, 12, 207, 3])
