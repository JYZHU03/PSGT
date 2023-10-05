import argparse
import time
from tqdm import tqdm
import torch.nn.functional as F
from torch.utils.data import random_split
from torch_geometric.data import DataLoader, DataListLoader
from torch_geometric.nn import DataParallel
from torch.nn import Linear
from torch_geometric.nn import global_mean_pool, GATConv

from utils.data_loader import *
from utils.eval_helper import *
import torch as th
from layers import Atten_transformer
from utils.mask_graph import *
from torch_geometric.utils import to_dense_adj

class Net(torch.nn.Module):
	def __init__(self, concat=False):
		super(Net, self).__init__()

		self.num_features = dataset.num_features
		self.num_classes = args.num_classes
		self.nhid = args.nhid
		self.concat = concat
		self.usegnn = False

		self.conv1 = GATConv(self.num_features, self.nhid * 2)
		self.conv2 = GATConv(self.nhid * 2, self.nhid * 2)

		self.liner1 = Linear(self.num_features, self.nhid * 2)
		self.liner2 = Linear(self.nhid * 2, self.nhid * 2)
		self.liner3 = Linear(self.nhid * 2, 1)



		self.Atten_transformer1 = Atten_transformer(self.nhid* 2)
		self.Atten_transformer2 = Atten_transformer(self.nhid* 2)


		self.fc1 = Linear(self.nhid * 2, self.nhid)

		if self.concat:
			self.fc0 = Linear(self.num_features, self.nhid)
			self.fc1 = Linear(self.nhid * 2, self.nhid)

		self.fc2 = Linear(self.nhid, self.num_classes)
		self.attn_mask = None
		#
		self.fix_r = False
		self.decay_interval = 10
		self.decay_r = 0.1
		self.final_r = 0.8
		self.init_r = 0.9


	def forward(self, data, training):
		x, edge_index, batch = data.x, data.edge_index, data.batch

		if self.usegnn:
			x_att = F.selu(self.conv1(x, edge_index))
			x_att = F.selu(self.conv2(x_att, edge_index))

			att_log_logits = th.sigmoid(F.selu(self.liner3(x_att)))
			att = self.sampling(att_log_logits, epoch, training)
			edge_att = self.lift_node_att_to_edge_att_gnn(att, edge_index, batch)

			x = F.selu(self.conv1(x, edge_index, edge_att))
			x = F.selu(self.conv2(x, edge_index, edge_att))
		else:
			x_att = F.selu(self.liner1(x))
			x_att = self.Atten_transformer1(x_att, data.batch, None)
			x_att = F.selu(self.liner2(x_att))
			x_att = self.Atten_transformer2(x_att, data.batch, None)

			att_log_logits = th.sigmoid(F.selu(self.liner3(x_att)))
			att = self.sampling(att_log_logits, epoch, training)
			atten_full = self.get_full_attention(batch)
			edge_att = self.lift_node_att_to_edge_att(att, atten_full, batch)
			edge_att = edge_att.cpu()

			self.attn_mask = get_attention_mask(data, edge_att).to(args.device)

			x = F.selu(self.liner1(x))
			x = self.Atten_transformer1(x, data.batch, self.attn_mask)
			x = F.selu(self.liner2(x))
			x = self.Atten_transformer2(x, data.batch, self.attn_mask)

		x = F.selu(global_mean_pool(x, batch))
		x = F.selu(self.fc1(x))
		x = F.dropout(x, p=0.5, training=self.training)

		if self.concat:
			news = torch.stack([data.x[(data.batch == idx).nonzero().squeeze()[0]] for idx in range(data.num_graphs)])
			news = F.relu(self.fc0(news))
			x = torch.cat([x, news], dim=1)
			x = F.relu(self.fc1(x))

		x = F.log_softmax(self.fc2(x), dim=-1)

		r = self.fix_r if self.fix_r else self.get_r(self.decay_interval, self.decay_r, epoch, final_r=self.final_r,init_r=self.init_r)
		info_loss = (att * torch.log(att / r + 1e-6) + (1 - att) * torch.log((1 - att) / (1 - r + 1e-6) + 1e-6)).mean()


		return x, info_loss

	def sampling(self, att_log_logits, epoch, training):
		att = self.concrete_sample(att_log_logits, temp=1, training=training)
		return att

	@staticmethod
	def lift_node_att_to_edge_att(node_att, edge_index, batch):
		src_lifted_att = node_att[edge_index[0]]
		dst_lifted_att = node_att[edge_index[1]]
		edge_att = src_lifted_att * dst_lifted_att
		edge_att_adj = to_dense_adj(edge_index, batch, edge_attr=edge_att).squeeze()
		return edge_att_adj

	@staticmethod
	def lift_node_att_to_edge_att_gnn(node_att, edge_index, batch):
		src_lifted_att = node_att[edge_index[0]]
		dst_lifted_att = node_att[edge_index[1]]
		edge_att = src_lifted_att * dst_lifted_att
		return edge_att

	@staticmethod
	def concrete_sample(att_log_logit, temp, training):
		if training:
			random_noise = torch.empty_like(att_log_logit).uniform_(1e-10, 1 - 1e-10)
			random_noise = torch.log(random_noise) - torch.log(
				1.0 - random_noise)
			att_bern = ((att_log_logit + random_noise) / temp).sigmoid()
		else:
			att_bern = (att_log_logit).sigmoid()
		return att_bern

	def get_r(self, decay_interval, decay_r, current_epoch, init_r=0.9, final_r=0.5):
		r = init_r - current_epoch // decay_interval * decay_r
		if r < final_r:
			r = final_r
		return r

	def get_full_attention(self, batch):
		edge_index_full = torch.empty((2, 0), dtype=torch.long).to(args.device)
		for graph_id in batch.unique():
			nodes = (batch == graph_id).nonzero(as_tuple=True)[0]

			edge_index = torch.combinations(nodes, r=2).to(args.device)

			edge_index_full = torch.cat([edge_index_full, edge_index.t(), edge_index.flip([1]).t()], dim=1)

			self_loops = torch.stack([nodes, nodes])
			edge_index_full = torch.cat([edge_index_full, self_loops], dim=1)
		return edge_index_full

@torch.no_grad()
def compute_test(loader, verbose=False):
	model.eval()
	loss_test = 0.0
	out_log = []
	for data in loader:
		if not args.multi_gpu:
			data = data.to(args.device)
		out, info_loss = model(data, training=False)
		if args.multi_gpu:
			y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
		else:
			y = data.y
		if verbose:
			print(F.softmax(out, dim=1).cpu().numpy())
		out_log.append([F.softmax(out, dim=1), y])
		loss_test += F.nll_loss(out, y).item()
	return eval_deep(out_log, loader), loss_test


parser = argparse.ArgumentParser()

# original model parameters
parser.add_argument('--seed', type=int, default=777, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0', help='specify cuda devices')

# hyper-parameters
parser.add_argument('--dataset', type=str, default='politifact', help='[politifact, gossipcop]')
parser.add_argument('--batch_size', type=int, default=128, help='batch size')
parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0.01, help='weight decay')
parser.add_argument('--nhid', type=int, default=128, help='hidden size')
parser.add_argument('--epochs', type=int, default=600, help='maximum number of epochs')
parser.add_argument('--concat', type=bool, default=False, help='whether concat news embedding and graph embedding')
# parser.add_argument('--usegnn', type=bool, default=False, help='whether use gnn as encoder')
parser.add_argument('--multi_gpu', type=bool, default=False, help='multi-gpu mode')
parser.add_argument('--feature', type=str, default='bert', help='feature type, [profile, spacy, bert, content]')

args = parser.parse_args()
torch.manual_seed(args.seed)
if torch.cuda.is_available():
	torch.cuda.manual_seed(args.seed)

dataset = FNNDataset(root='../data', feature=args.feature, empty=False, name=args.dataset, transform=ToUndirected())

args.num_classes = dataset.num_classes
args.num_features = dataset.num_features

print(args)

num_training = int(len(dataset) * 0.75)
num_val = int(len(dataset) * 0.1)
num_test = len(dataset) - (num_training + num_val)
training_set, validation_set, test_set = random_split(dataset, [num_training, num_val, num_test])

if args.multi_gpu:
	loader = DataListLoader
else:
	loader = DataLoader

train_loader = loader(training_set, batch_size=args.batch_size, shuffle=True)
val_loader = loader(validation_set, batch_size=args.batch_size, shuffle=False)
test_loader = loader(test_set, batch_size=args.batch_size, shuffle=False)

model = Net(concat=args.concat).to(args.device)
if args.multi_gpu:
	model = DataParallel(model)
model = model.to(args.device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


if __name__ == '__main__':
	# Model training
	t = time.time()
	model.train()
	for epoch in tqdm(range(args.epochs)):
		out_log = []
		loss_train = 0.0
		for i, data in enumerate(train_loader):


			optimizer.zero_grad()
			if not args.multi_gpu:
				data = data.to(args.device)
			out, info_loss = model(data, training=True)
			if args.multi_gpu:
				y = torch.cat([d.y.unsqueeze(0) for d in data]).squeeze().to(out.device)
			else:
				y = data.y
			loss = F.nll_loss(out, y)
			loss = loss + info_loss
			loss.backward()
			optimizer.step()
			loss_train += loss.item()

			out_log.append([F.softmax(out, dim=1), y])
		acc_train, _, _, _, recall_train, auc_train, _ = eval_deep(out_log, train_loader)
		[acc_val, _, _, _, recall_val, auc_val, _], loss_val = compute_test(val_loader)
		print(f'loss_train: {loss_train:.4f}, acc_train: {acc_train:.4f},'
			  f' recall_train: {recall_train:.4f}, auc_train: {auc_train:.4f},'
			  f' loss_val: {loss_val:.4f}, acc_val: {acc_val:.4f},'
			  f' recall_val: {recall_val:.4f}, auc_val: {auc_val:.4f}')

		[acc, f1_macro, f1_micro, precision, recall, auc, ap], test_loss = compute_test(test_loader, verbose=False)
		print(f'Test set results: acc: {acc:.4f}, f1_macro: {f1_macro:.4f}, f1_micro: {f1_micro:.4f}, '
			  f'precision: {precision:.4f}, recall: {recall:.4f}, auc: {auc:.4f}, ap: {ap:.4f}')
