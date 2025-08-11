
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module
from functools import partial



from DACN import transformer

def sce_loss(x, y, alpha=3):
    x = F.normalize(x, p=2, dim=-1)
    y = F.normalize(y, p=2, dim=-1)

    # loss =  - (x * y).sum(dim=-1)
    # loss = (x_h - y_h).norm(dim=1).pow(alpha)

    loss = (1 - (x * y).sum(dim=-1)).pow_(alpha)

    loss = loss.mean()
    return loss


def full_block(in_features, out_features, p_drop):
    return nn.Sequential(
        nn.Linear(in_features, out_features),
        nn.BatchNorm1d(out_features, momentum=0.01, eps=0.001),
        nn.ELU(),
        nn.Dropout(p=p_drop),
    )


class SelfAttention(nn.Module):
    def __init__(self, dim):
        super(SelfAttention, self).__init__()
        self.fc_q = nn.Linear(dim, dim)  # Query
        self.fc_k = nn.Linear(dim, dim)  # Key
        self.fc_v = nn.Linear(dim, dim)  # Value

    def forward(self, x):
        q = self.fc_q(x)
        k = self.fc_k(x)
        v = self.fc_v(x)

        # 计算注意力权重
        scores = torch.matmul(q, k.transpose(-2, -1)) / (x.size(-1) ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)

        # 生成加权特征
        attn_output = torch.matmul(attn_weights, v)
        return attn_output, attn_weights

class WeightedConcatFusionWithAttention(nn.Module):
    def __init__(self, dim):
        super(WeightedConcatFusionWithAttention, self).__init__()
        self.attention = SelfAttention(dim)

    def forward(self, feat_x, gnn_z):
        # 计算 feat_x 的注意力加权
        attn_feat_x, attn_weights_x = self.attention(feat_x.unsqueeze(1))  # 增加维度以符合输入格式
        attn_feat_x = attn_feat_x.squeeze(1)  # 去除增加的维度

        # 计算 gnn_z 的注意力加权
        attn_gnn_z, attn_weights_z = self.attention(gnn_z.unsqueeze(1))  # 增加维度以符合输入格式
        attn_gnn_z = attn_gnn_z.squeeze(1)  # 去除增加的维度

        z_fused = torch.cat((attn_feat_x, attn_gnn_z), dim=1)

        return z_fused, attn_weights_x, attn_weights_z

class ComplexEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dims, p_drop):
        super(ComplexEncoder, self).__init__()

        self.num_heads = 1
        self.layers = nn.ModuleList()
        self.layer = nn.ModuleList()
        in_features = input_dim
        self.layer.append(nn.MultiheadAttention(embed_dim=input_dim, num_heads=self.num_heads))
        for out_features in hidden_dims:
            self.layers.append(full_block(in_features, out_features, p_drop))
            in_features = out_features

        self.adjust_layer = nn.Linear(in_features, input_dim)

    def forward(self, x):
        identity = x  # 残差连接的输入
        x = x.unsqueeze(1)  # 增加一个维度，形状变为 (batch_size, 1, feature_dim)
        for layer in self.layer:
            x, _ = layer(x, x, x)  # 使用相同的输入作为查询、键和值
        x = x.squeeze(1)  # 移除多余的维度，形状变为 (batch_size, feature_dim)
        x = x + identity
        for layer in self.layers:
            x = layer(x)
        # print(x.shape)
        # x = self.adjust_layer(x)  # 可选的输出层
        # for layer in self.layers:
        #     x = layer(x)
        return x

# GCN Layer
class GraphConvolution(Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.xavier_uniform_(self.weight)

    def forward(self, input, adj):
        input = F.dropout(input, self.dropout, self.training)
        support = torch.mm(input, self.weight)
        output = torch.spmm(adj, support)
        output = self.act(output)
        return output


class GraphTransformer(Module):

    def __init__(self, in_features, out_features, dropout=0., act=F.relu):
        super(GraphTransformer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.act = act

        # Query, Key, Value transformation matrices for Attention mechanism
        self.query = Parameter(torch.FloatTensor(in_features, out_features))
        self.key = Parameter(torch.FloatTensor(in_features, out_features))
        self.value = Parameter(torch.FloatTensor(in_features, out_features))

        # Dropout layer
        self.dropout_layer = nn.Dropout(p=self.dropout)

        self.reset_parameters()

    def reset_parameters(self):
        # Xavier initialization for the query, key, and value matrices
        torch.nn.init.xavier_uniform_(self.query)
        torch.nn.init.xavier_uniform_(self.key)
        torch.nn.init.xavier_uniform_(self.value)

    def forward(self, input, adj):
        # Apply dropout to input
        input = self.dropout_layer(input)

        # Linear projections to get query, key, value
        query = torch.mm(input, self.query)  # (N, out_features)
        key = torch.mm(input, self.key)  # (N, out_features)
        value = torch.mm(input, self.value)  # (N, out_features)

        # Attention scores (N, N)
        attention_scores = torch.mm(query, key.t())  # (N, N)
        # attention_scores = attention_scores * adj.to_dense()

        # Apply softmax to get attention weights (N, N)
        attention_weights = torch.softmax(attention_scores, dim=1)
        # Aggregate values using the attention weights (N, out_features)
        attention_output = torch.mm(attention_weights, value)
        # Optionally apply dropout to the output
        attention_output = self.dropout_layer(attention_output)

        # Apply the activation function
        output = self.act(attention_output)
        return output


# class InnerProductDecoder1(nn.Module):
#     """Decoder for using inner product for prediction."""
#
#     def __init__(self, dropout, act=torch.sigmoid):
#         super(InnerProductDecoder1, self).__init__()
#         self.dropout = dropout
#         self.act = act
#
#     def forward(self, z, mask):
#         col = mask.coalesce().indices()[0]
#         row = mask.coalesce().indices()[1]
#         values = mask.coalesce().values()
#
#         idx = torch.where(values > 0)[0]
#         col = col[idx]
#         row = row[idx]
#
#
#         result = self.act(torch.sum(z[col] * z[row], axis=1))
#         return result
#
#         # z = F.dropout(z, self.dropout, training=self.training)
#         # adj = self.act(torch.mm(z, z.t()))
#         # return adj




class InnerProductDecoder(nn.Module):
    """Decoder for using inner product for prediction."""

    def __init__(self, dropout, act=torch.sigmoid):
        super(InnerProductDecoder, self).__init__()
        self.dropout = dropout
        self.act = act

    def forward(self, z, mask):
        col = mask.coalesce().indices()[0]
        row = mask.coalesce().indices()[1]
        result = self.act(torch.sum(z[col] * z[row], axis=1))

        return result

class Generator(nn.Module):
    def __init__(self, latent_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, output_dim),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z)




class module(nn.Module):
    def __init__(
            self,
            input_dim,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=64,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_clsuter_n=10,
            generate_dim =200,
    ):
        super(module, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_clsuter_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2
        self.feature_fusion = WeightedConcatFusionWithAttention(dim=16)

        self.generator = Generator(latent_dim=self.feat_hidden2, output_dim=self.input_dim)
        self.linear = nn.Linear(feat_hidden2, 1)
        self.encoder = ComplexEncoder(input_dim, [feat_hidden1, feat_hidden2], p_drop)

        #feature autoencoder
        # self.encoder = nn.Sequential()
        # self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        # self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))




        self.decoder = GraphConvolution(self.latent_dim, self.input_dim, self.p_drop, act=lambda x: x)
        # self.decoder = nn.Sequential()
        # self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, self.p_drop))

        # GCN layers
        self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)

        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)

        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.gcn_hidden2 + self.feat_hidden2))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        #########################
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self._mask_rate = 0.8
        self.criterion = self.setup_loss_fn(loss_fn='sce')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return criterion

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,train_mode = None):
        if train_mode == "gan":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mu, logvar, feat_x = self.encode(x, adj)
            z_fake = torch.randn(x.size(0), feat_x.size(1)).to(device)  # 随机噪声
            fake_x = self.generator(z_fake)  # 生成伪造数据

            real_feat = feat_x
            fake_feat = self.encoder(fake_x.detach())  # 假数据的编码器特征（不回传梯度）

            batch_size = x.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 实例化 BCEWithLogitsLoss 损失函数
            criterion = nn.BCEWithLogitsLoss()

            real_pre = self.linear(real_feat)
            fake_pre = self.linear(fake_feat)
            loss_real = criterion(real_pre, real_labels)
            loss_fake = criterion(fake_pre, fake_labels)
            loss_D = (loss_real + loss_fake) / 2

            fake_feat = self.encoder(fake_x)  # 生成器希望“欺骗”判别器
            fake_pre = self.linear(fake_feat)
            loss_G = criterion(fake_pre, real_labels)
            return loss_G, loss_D

        else:
            adj, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, x, self._mask_rate)  #

            mu, logvar, feat_x = self.encode(x, adj)
            gnn_z = self.reparameterize(mu, logvar)
            #z = torch.cat((feat_x, gnn_z), 1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            feat_x, gnn_z = feat_x.to(device), gnn_z.to(device)
            z, attn_weights_x, attn_weights_z = self.feature_fusion(feat_x, gnn_z)

            de_feat = self.decoder(z, adj)

            # DEC clustering
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

            # self-construction loss
            recon = de_feat.clone()  #
            x_init = x[mask_nodes]  #
            x_rec = recon[mask_nodes]  #

            loss = self.criterion(x_rec, x_init) #
            # loss = 0
            return z, mu, logvar, de_feat, q, feat_x, gnn_z, loss

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)
        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        # print("mask_nodes:", mask_nodes)
        # print("x shape:", x.shape)
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        #out_x[mask_nodes] = 0.0

        token_nodes = token_nodes[token_nodes < out_x.shape[0]]
        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()


        return use_adj, out_x, (mask_nodes, keep_nodes)


class impute_module(nn.Module):
    def __init__(
            self,
            input_dim,
            feat_hidden1=64,
            feat_hidden2=16,
            gcn_hidden1=64,
            gcn_hidden2=16,
            p_drop=0.2,
            alpha=1.0,
            dec_clsuter_n=10,
    ):
        super(impute_module, self).__init__()
        self.input_dim = input_dim
        self.feat_hidden1 = feat_hidden1
        self.feat_hidden2 = feat_hidden2
        self.gcn_hidden1 = gcn_hidden1
        self.gcn_hidden2 = gcn_hidden2
        self.p_drop = p_drop
        self.alpha = alpha
        self.dec_cluster_n = dec_clsuter_n
        self.latent_dim = self.gcn_hidden2 + self.feat_hidden2
        self.feature_fusion = WeightedConcatFusionWithAttention(dim=16)

        self.generator = Generator(latent_dim=self.feat_hidden2, output_dim=self.input_dim)
        self.linear = nn.Linear(feat_hidden2, 1)
        self.encoder = ComplexEncoder(input_dim, [feat_hidden1, feat_hidden2], p_drop)

        # # feature autoencoder
        # self.encoder = nn.Sequential()
        # self.encoder.add_module('encoder_L1', full_block(self.input_dim, self.feat_hidden1, self.p_drop))
        # self.encoder.add_module('encoder_L2', full_block(self.feat_hidden1, self.feat_hidden2, self.p_drop))

        self.decoder = nn.Sequential()
        self.decoder.add_module('decoder_L0', full_block(self.latent_dim, input_dim, self.p_drop))

        # GCN layers
        self.gc1 = GraphConvolution(self.feat_hidden2, self.gcn_hidden1, self.p_drop, act=F.relu)
        self.gc2 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.gc3 = GraphConvolution(self.gcn_hidden1, self.gcn_hidden2, self.p_drop, act=lambda x: x)
        self.dc = InnerProductDecoder(self.p_drop, act=lambda x: x)

        # DEC cluster layer
        self.cluster_layer = Parameter(torch.Tensor(self.dec_cluster_n, self.latent_dim))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)

        #########################
        self.enc_mask_token = nn.Parameter(torch.zeros(1, input_dim))
        self._mask_rate = 0.8
        self.criterion = self.setup_loss_fn(loss_fn='sce')

    def setup_loss_fn(self, loss_fn, alpha_l=3):
        if loss_fn == "mse":
            criterion = nn.MSELoss()
        elif loss_fn == "sce":
            criterion = partial(sce_loss, alpha=3)
        else:
            raise NotImplementedError
        return criterion

    def encode(self, x, adj):
        feat_x = self.encoder(x)
        hidden1 = self.gc1(feat_x, adj)
        return self.gc2(hidden1, adj), self.gc3(hidden1, adj), feat_x

    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(logvar)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x, adj,train_mode = None):

        if train_mode == "gan":
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            mu, logvar, feat_x = self.encode(x, adj)
            z_fake = torch.randn(x.size(0), feat_x.size(1)).to(device)  # 随机噪声
            fake_x = self.generator(z_fake)  # 生成伪造数据

            real_feat = feat_x
            fake_feat = self.encoder(fake_x.detach())  # 假数据的编码器特征（不回传梯度）

            batch_size = x.size(0)
            real_labels = torch.ones(batch_size, 1).to(device)
            fake_labels = torch.zeros(batch_size, 1).to(device)

            # 实例化 BCEWithLogitsLoss 损失函数
            criterion = nn.BCEWithLogitsLoss()

            real_pre = self.linear(real_feat)
            fake_pre = self.linear(fake_feat)
            loss_real = criterion(real_pre, real_labels)
            loss_fake = criterion(fake_pre, fake_labels)
            loss_D = (loss_real + loss_fake) / 2

            fake_feat = self.encoder(fake_x)  # 生成器希望“欺骗”判别器
            fake_pre = self.linear(fake_feat)
            loss_G = criterion(fake_pre, real_labels)
            return loss_G, loss_D

        else:
            adj, x, (mask_nodes, keep_nodes) = self.encoding_mask_noise(adj, x, self._mask_rate)  #

            mu, logvar, feat_x = self.encode(x, adj)
            gnn_z = self.reparameterize(mu, logvar)
            # z = torch.cat((feat_x, gnn_z), 1)
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            feat_x, gnn_z = feat_x.to(device), gnn_z.to(device)
            z, attn_weights_x, attn_weights_z = self.feature_fusion(feat_x, gnn_z)
            de_feat = self.decoder(z)


            # DEC clustering
            q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.alpha)
            q = q.pow((self.alpha + 1.0) / 2.0)
            q = (q.t() / torch.sum(q, 1)).t()

            # self-construction loss
            recon = de_feat.clone()  #
            x_init = x[mask_nodes]  #
            x_rec = recon[mask_nodes]  #

            loss = self.criterion(x_rec, x_init)  #
            # loss = 0

            return z, mu, logvar, de_feat, q, feat_x, gnn_z, loss

    def encoding_mask_noise(self, adj, x, mask_rate=0.3):
        num_nodes = adj.shape[0]
        perm = torch.randperm(num_nodes, device=x.device)

        # random masking
        num_mask_nodes = int(mask_rate * num_nodes)
        mask_nodes = perm[: num_mask_nodes]
        keep_nodes = perm[num_mask_nodes:]

        out_x = x.clone()
        token_nodes = mask_nodes
        #out_x[mask_nodes] = 0.0

        out_x[token_nodes] += self.enc_mask_token
        use_adj = adj.clone()

        return use_adj, out_x, (mask_nodes, keep_nodes)