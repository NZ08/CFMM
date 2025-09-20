
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from modules import CrossAttention
from utility.parser import parse_args
from utility.norm import build_sim, build_knn_normalized_graph

args = parse_args()

class CFMM(nn.Module):
    def __init__(self, n_users, n_items, embedding_dim, weight_size, dropout_list, image_feats, text_feats):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.weight_size = weight_size
        self.n_ui_layers = len(self.weight_size)
        self.weight_size = [self.embedding_dim] + self.weight_size
        self.user_embedding = nn.Embedding(n_users, self.embedding_dim)
        self.item_id_embedding = nn.Embedding(n_items, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_id_embedding.weight)
        
        self.crossAtt = CrossAttention()
        self.image_embedding = nn.Embedding.from_pretrained(torch.Tensor(image_feats), freeze=False)
        self.text_embedding = nn.Embedding.from_pretrained(torch.Tensor(text_feats), freeze=False)
        image_adj = build_sim(self.image_embedding.weight.detach())
        image_adj = build_knn_normalized_graph(image_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)

        text_adj = build_sim(self.text_embedding.weight.detach())
        text_adj = build_knn_normalized_graph(text_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)
        self.text_original_adj = text_adj.cuda()
        self.image_original_adj = image_adj.cuda()
        
        self.image_trs = nn.Linear(image_feats.shape[1], embedding_dim)
        self.text_trs = nn.Linear(text_feats.shape[1], embedding_dim)
        nn.init.xavier_uniform_(self.image_trs.weight)
        nn.init.xavier_uniform_(self.text_trs.weight)

        self.softmax = nn.Softmax(dim=-1)


        self.query = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.Tanh(),
            nn.Linear(self.embedding_dim, 1, bias=False)
        )

        self.tau = 0.5

    def mm(self, x, y):
        if args.sparse:
            return torch.sparse.mm(x, y)
        else:
            return torch.mm(x, y)
    def sim(self, z1, z2):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())
    def init_weights(self):
        for layer in self.in_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        for layer in self.out_layers:
            size = layer.weight.size()
            std = np.sqrt(2.0 / (size[0] + size[1]))
            layer.weight.data.normal_(0.0, std)
            layer.bias.data.normal_(0.0, 0.001)

        size = self.emb_layer.weight.size()
        std = np.sqrt(2.0 / (size[0] + size[1]))
        self.emb_layer.weight.data.normal_(0.0, std)
        self.emb_layer.bias.data.normal_(0.0, 0.001)

    def batched_contrastive_loss(self, z1, z2, batch_size=4096):
        device = z1.device
        num_nodes = z1.size(0)
        num_batches = (num_nodes - 1) // batch_size + 1
        f = lambda x: torch.exp(x / self.tau)
        indices = torch.arange(0, num_nodes).to(device)
        losses = []

        for i in range(num_batches):
            mask = indices[i * batch_size:(i + 1) * batch_size]
            refl_sim = f(self.sim(z1[mask], z1))  # [B, N]
            between_sim = f(self.sim(z1[mask], z2))  # [B, N]

            losses.append(-torch.log(
                between_sim[:, i * batch_size:(i + 1) * batch_size].diag()
                / (refl_sim.sum(1) + between_sim.sum(1)
                   - refl_sim[:, i * batch_size:(i + 1) * batch_size].diag())))
                   
        loss_vec = torch.cat(losses)
        return loss_vec.mean()

    def forward(self, adj, build_item_graph=False):
        image_feats = self.image_trs(self.image_embedding.weight)
        text_feats = self.text_trs(self.text_embedding.weight)

        if build_item_graph:
            self.image_adj = build_sim(image_feats)
            self.image_adj = build_knn_normalized_graph(self.image_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)
            self.image_adj = (1 - args.lambda_coeff) * self.image_adj + args.lambda_coeff * self.image_original_adj

            self.text_adj = build_sim(text_feats)
            self.text_adj = build_knn_normalized_graph(self.text_adj, topk=args.topk, is_sparse=args.sparse, norm_type=args.norm_type)
            self.text_adj = (1 - args.lambda_coeff) * self.text_adj + args.lambda_coeff * self.text_original_adj
        else:
            self.image_adj = self.image_adj.detach()
            self.text_adj = self.text_adj.detach()

        image_item_embeds = self.item_id_embedding.weight
        text_item_embeds = self.item_id_embedding.weight
        
        for i in range(args.layers):
            image_item_embeds = self.mm(self.image_adj, image_item_embeds)
        for i in range(args.layers):
            text_item_embeds = self.mm(self.text_adj, text_item_embeds)
        att = torch.cat([self.query(image_item_embeds), self.query(text_item_embeds)], dim=-1)
        weight = self.softmax(att)
        h = weight[:, 0].unsqueeze(dim=1) * image_item_embeds + weight[:, 1].unsqueeze(dim=1) * text_item_embeds
        
        # Att_lightgcn model implementation
        ego_embeddings = torch.cat((self.user_embedding.weight, self.item_id_embedding.weight), dim=0)
        all_embeddings = [ego_embeddings]

        for i in range(self.n_ui_layers):
            side_embeddings = torch.sparse.mm(adj, ego_embeddings)
            ego_embeddings = side_embeddings
            all_embeddings += [ego_embeddings]
        all_embeddings = torch.stack(all_embeddings, dim=1)
        all_embeddings = all_embeddings.mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)
        h_norm = F.normalize(h, p=2, dim=1)
        att = self.crossAtt(i_g_embeddings.unsqueeze(0), h_norm.unsqueeze(0))

        weight = self.softmax(att.squeeze())
        i_g_embeddings = weight[:, 0].unsqueeze(dim=1) * i_g_embeddings + weight[:, 1].unsqueeze(dim=1) * h_norm

        return u_g_embeddings, i_g_embeddings, image_item_embeds, text_item_embeds, h

