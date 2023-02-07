# transformer: https://peterbloem.nl/blog/transformers
import torch
from torch import nn
import torch.nn.functional as F

# self-attention
class SelfAttention(nn.Module):

	def __init__(self, k, heads):
		super(SelfAttention, self).__init__()
		self.k = k
		self.heads = heads

	    # These compute the queries, keys and values for all heads (as a single concatenated vector)
	    self.tokeys    = nn.Linear(k, k * heads, bias=False)
	    self.toqueries = nn.Linear(k, k * heads, bias=False)
		self.tovalues  = nn.Linear(k, k * heads, bias=False)

		# This unifies the outputs of the different heads into a single k-vector
		self.unifyheads = nn.Linear(heads * k, k)

	def forward(self, x):
		b, t, k = x.size()
		h = self.heads

		queries = self.toqueries(x).view(b, t, h, k)
		keys    = self.tokeys(x).view(b, t, h, k)
		values  = self.tovalues(x).view(b, t, h, k)

		# - fold heads into the batch dimension
	    keys    = keys.transpose(1, 2).contiguous().view(b * h, t, k)
	    queries = queries.transpose(1, 2).contiguous().view(b * h, t, k)
	    values  = values.transpose(1, 2).contiguous().view(b * h, t, k)

	    queries = queries / (k ** (1/4))
    	keys    = keys / (k ** (1/4))

    	dot = torch.bmm(queries, keys.transpose(1, 2)) # dot has size (b*h, t, t) containing raw weights
    	dot = F.softmax(dot, dim=2)

    	out = torch.bmm(dot, values).view(b, h, t, k)

    	out = out.transpose(1, 2).contiguous().view(b, t, h * k)
    	return self.unifyheads(out)


class TransformerBlock(nn.Module):

	def __init__(self, k, heads):
		super(TransformerBlock, self).__init__()
		
		self.attention = SelfAttention(k, heads)

		self.norm1 = nn.LayerNorm(k)
		self.norm2 = nn.LayerNorm(k)

		self.mlp = nn.Sequential(
			nn.Linear(k, 4 * k),
      		nn.ReLU(),
      		nn.Linear(4 * k, k))

	def forward(self, x):
		x = self.attention(x)
		x = self.norm1(x)
		x = self.norm2(x + self.mlp(x))
		
		return x

class Transformer(nn.Module):

	def __init__(self, k, heads, depth, seq_length, num_tokens, num_classes):
		super(Transformer, self).__init__()

		self.num_tokens = num_tokens
		self.token_embedding = nn.Embedding(num_tokens, k)
		self.position_embedding = nn.Embedding(seq_length, k)
		
		tblocks = []
        for i in range(depth):
            tblocks.append(TransformerBlock(k, heads))
        self.tblocks = nn.Sequential(*tblocks)

        self.toprobs = nn.Linear(k, num_classes)

    def forward(self, x):

    	tokens = self.token_embedding(x)
    	b, t, k = tokens.size()

    	positions = torch.arange(t)
        positions = self.pos_emb(positions)[None, :, :].expand(b, t, k)

        x = tokens + positions
        x = self.tblocks(x)

        # Average-pool over the t dimension and project to class probabilities
        x = self.toprobs(x.mean(dim=1))
        return F.log_softmax(x, dim=1)
				
