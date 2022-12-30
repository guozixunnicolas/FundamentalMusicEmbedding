import copy
from typing import Optional, Any, Union, Callable
import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F
from torch.nn import ModuleList, Dropout, Linear, LayerNorm #MultiheadAttention,
import math
from torch.nn.init import xavier_uniform_

class TransformerEncoder(nn.Module):
	r"""TransformerEncoder is a stack of N encoder layers
	Args:
		encoder_layer: an instance of the TransformerEncoderLayer() class (required).
		num_layers: the number of sub-encoder-layers in the encoder (required).
		norm: the layer normalization component (optional).
	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
		>>> src = torch.rand(10, 32, 512)
		>>> out = transformer_encoder(src)
	"""
	__constants__ = ['norm']

	def __init__(self, encoder_layer, num_layers, norm=None):
		super(TransformerEncoder, self).__init__()
		self.layers = _get_clones(encoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, src: Tensor, mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None, pitch_rel=None,pitch_rel_mask=None, dur_rel=None, dur_rel_mask=None) -> Tensor:
		r"""Pass the input through the encoder layers in turn.
		Args:
			src: the sequence to the encoder (required).
			mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).
		Shape:
			see the docs in Transformer class.
		"""
		output = src
		attention_weights = []
		# print(f"check src:{src.shape}, pitch_rel:{pitch_rel.shape},pitch_rel_mask:{pitch_rel_mask.shape}, dur_rel:{dur_rel.shape}, dur_rel_mask:{dur_rel_mask.shape} ") 

		for mod in self.layers:
			# output = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask) #I changed here 
			output, attn_weight = mod(output, src_mask=mask, src_key_padding_mask=src_key_padding_mask, pitch_rel=pitch_rel,pitch_rel_mask=pitch_rel_mask, dur_rel=dur_rel, dur_rel_mask=dur_rel_mask)
			attention_weights.append(attn_weight)
			# print(f"attn_weight:{attn_weight.shape}, output:{output.shape}")
		attention_weights_cat = torch.cat(attention_weights, dim = -1) 
		# print(f"attention_weights_cat:{attention_weights_cat.shape}")
		if self.norm is not None: #PENDING CHANGE, SHOULD BE NONE
			output = self.norm(output)

		return output, attention_weights_cat

class TransformerDecoder(nn.Module):
	r"""TransformerDecoder is a stack of N decoder layers
	Args:
		decoder_layer: an instance of the TransformerDecoderLayer() class (required).
		num_layers: the number of sub-decoder-layers in the decoder (required).
		norm: the layer normalization component (optional).
	Examples::
		>>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
		>>> transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=6)
		>>> memory = torch.rand(10, 32, 512)
		>>> tgt = torch.rand(20, 32, 512)
		>>> out = transformer_decoder(tgt, memory)
	"""
	__constants__ = ['norm']

	def __init__(self, decoder_layer, num_layers, norm=None):
		super(TransformerDecoder, self).__init__()
		self.layers = _get_clones(decoder_layer, num_layers)
		self.num_layers = num_layers
		self.norm = norm

	def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None,
				memory_mask: Optional[Tensor] = None, tgt_key_padding_mask: Optional[Tensor] = None,
				memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the inputs (and mask) through the decoder layer in turn.
		Args:
			tgt: the sequence to the decoder (required).
			memory: the sequence from the last layer of the encoder (required).
			tgt_mask: the mask for the tgt sequence (optional).
			memory_mask: the mask for the memory sequence (optional).
			tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
			memory_key_padding_mask: the mask for the memory keys per batch (optional).
		Shape:
			see the docs in Transformer class.
		"""
		output = tgt
		#pending change! lack embedding & pos encoding &dropout!

		for mod in self.layers:
			output = mod(output, memory, tgt_mask=tgt_mask,
						 memory_mask=memory_mask,
						 tgt_key_padding_mask=tgt_key_padding_mask,
						 memory_key_padding_mask=memory_key_padding_mask)

		if self.norm is not None:
			output = self.norm(output)

		return output

class TransformerEncoderLayer(nn.Module):
	r"""TransformerEncoderLayer is made up of self-attn and feedforward network.
	This standard encoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.
	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).
		activation: the activation function of the intermediate layer, can be a string
			("relu" or "gelu") or a unary callable. Default: relu
		layer_norm_eps: the eps value in layer normalization components (default=1e-5).
		batch_first: If ``True``, then the input and output tensors are provided
			as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
		norm_first: if ``True``, layer norm is done prior to attention and feedforward
			operations, respectivaly. Otherwise it's done after. Default: ``False`` (after).
	Examples::
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8)
		>>> src = torch.rand(10, 32, 512)
		>>> out = encoder_layer(src)
	Alternatively, when ``batch_first`` is ``True``:
		>>> encoder_layer = nn.TransformerEncoderLayer(d_model=512, nhead=8, batch_first=True)
		>>> src = torch.rand(32, 10, 512)
		>>> out = encoder_layer(src)
	"""
	__constants__ = ['batch_first', 'norm_first']

	def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
				 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
				 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
				 device=None, dtype=None) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super(TransformerEncoderLayer, self).__init__()
		# self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
		#                                     **factory_kwargs) #why does this need drop out? pending change
		# self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first) #why does this need drop out? pending change
		self.self_attn = RelativeGlobalAttention_provided(d_model, nhead, dropout=dropout, batch_first=batch_first)
		# Implementation of Feedforward model
		self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
		self.dropout = Dropout(dropout)
		self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

		self.norm_first = norm_first
		self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
		self.dropout3 = Dropout(dropout)

		# Legacy string support for activation function.
		if isinstance(activation, str):
			self.activation = _get_activation_fn(activation)
		else:
			self.activation = activation

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerEncoderLayer, self).__setstate__(state)

	def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the input through the encoder layer.
		Args:
			src: the sequence to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).
		Shape:
			see the docs in Transformer class.
		"""

		# see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

		x = src

		attn_logits, attn_weights = self._sa_block(self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask)
		if self.norm_first:
			x = x + attn_logits
			x = x + self._ff_block(self.norm2(x))
		else: #choose this as ori!  pending change
			x = self.norm1(x + attn_logits)
			x = self.norm2(x + self._ff_block(x))

		return x, attn_weights # I changed to return weight too!

	# self-attention block
	def _sa_block(self, x: Tensor,
				  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
		# x = self.self_attn(x, x, x,
		#                    attn_mask=attn_mask,
		#                    key_padding_mask=key_padding_mask,
		#                    need_weights=False)[0]
		x, attn_weight = self.self_attn(x, x, x,
						   attn_mask=attn_mask,
						   key_padding_mask=key_padding_mask)
		return self.dropout1(x), self.dropout3(attn_weight)

	# feed forward block
	def _ff_block(self, x: Tensor) -> Tensor:
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout2(x)

class TransformerDecoderLayer(nn.Module):
	r"""TransformerDecoderLayer is made up of self-attn, multi-head-attn and feedforward network.
	This standard decoder layer is based on the paper "Attention Is All You Need".
	Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
	Lukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. In Advances in
	Neural Information Processing Systems, pages 6000-6010. Users may modify or implement
	in a different way during application.
	Args:
		d_model: the number of expected features in the input (required).
		nhead: the number of heads in the multiheadattention models (required).
		dim_feedforward: the dimension of the feedforward network model (default=2048).
		dropout: the dropout value (default=0.1).
		activation: the activation function of the intermediate layer, can be a string
			("relu" or "gelu") or a unary callable. Default: relu
		layer_norm_eps: the eps value in layer normalization components (default=1e-5).
		batch_first: If ``True``, then the input and output tensors are provided
			as (batch, seq, feature). Default: ``False`` (seq, batch, feature).
		norm_first: if ``True``, layer norm is done prior to self attention, multihead
			attention and feedforward operations, respectivaly. Otherwise it's done after.
			Default: ``False`` (after).
	Examples::
		>>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8)
		>>> memory = torch.rand(10, 32, 512)
		>>> tgt = torch.rand(20, 32, 512)
		>>> out = decoder_layer(tgt, memory)
	Alternatively, when ``batch_first`` is ``True``:
		>>> decoder_layer = nn.TransformerDecoderLayer(d_model=512, nhead=8, batch_first=True)
		>>> memory = torch.rand(32, 10, 512)
		>>> tgt = torch.rand(32, 20, 512)
		>>> out = decoder_layer(tgt, memory)
	"""
	__constants__ = ['batch_first', 'norm_first']

	def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
				 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
				 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
				 device=None, dtype=None) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super(TransformerDecoderLayer, self).__init__()
		self.self_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
											**factory_kwargs)
		self.multihead_attn = MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=batch_first,
												 **factory_kwargs)
		# Implementation of Feedforward model
		self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
		self.dropout = Dropout(dropout)
		self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

		self.norm_first = norm_first
		self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm3 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
		self.dropout3 = Dropout(dropout)

		# Legacy string support for activation function.
		if isinstance(activation, str):
			self.activation = _get_activation_fn(activation)
		else:
			self.activation = activation

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerDecoderLayer, self).__setstate__(state)

	def forward(self, tgt: Tensor, memory: Tensor, tgt_mask: Optional[Tensor] = None, memory_mask: Optional[Tensor] = None,
				tgt_key_padding_mask: Optional[Tensor] = None, memory_key_padding_mask: Optional[Tensor] = None) -> Tensor:
		r"""Pass the inputs (and mask) through the decoder layer.
		Args:
			tgt: the sequence to the decoder layer (required).
			memory: the sequence from the last layer of the encoder (required).
			tgt_mask: the mask for the tgt sequence (optional).
			memory_mask: the mask for the memory sequence (optional).
			tgt_key_padding_mask: the mask for the tgt keys per batch (optional).
			memory_key_padding_mask: the mask for the memory keys per batch (optional).
		Shape:
			see the docs in Transformer class.
		"""
		# see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf

		x = tgt
		if self.norm_first:
			x = x + self._sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask)
			x = x + self._mha_block(self.norm2(x), memory, memory_mask, memory_key_padding_mask)
			x = x + self._ff_block(self.norm3(x))
		else:
			x = self.norm1(x + self._sa_block(x, tgt_mask, tgt_key_padding_mask))
			x = self.norm2(x + self._mha_block(x, memory, memory_mask, memory_key_padding_mask))
			x = self.norm3(x + self._ff_block(x))

		return x

	# self-attention block
	def _sa_block(self, x: Tensor,
				  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
		x = self.self_attn(x, x, x,
						   attn_mask=attn_mask,
						   key_padding_mask=key_padding_mask,
						   need_weights=False)[0]
		return self.dropout1(x)

	# multihead attention block
	def _mha_block(self, x: Tensor, mem: Tensor,
				   attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor]) -> Tensor:
		x = self.multihead_attn(x, mem, mem,
								attn_mask=attn_mask,
								key_padding_mask=key_padding_mask,
								need_weights=False)[0]
		return self.dropout2(x)

	# feed forward block
	def _ff_block(self, x: Tensor) -> Tensor:
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout3(x)

class RelativeGlobalAttention_my_version(nn.Module):

	def __init__(self, embed_dim=256, num_heads=4, dropout = 0.2, add_emb=False ,max_seq=245, batch_first=True):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.len_k = None
		self.max_seq = max_seq
		self.h = num_heads
		self.d = embed_dim
		self.dh = embed_dim // num_heads
		self.Wq = torch.nn.Linear(self.d, self.d)
		self.Wk = torch.nn.Linear(self.d, self.d)
		self.Wv = torch.nn.Linear(self.d, self.d)
		self.fc = torch.nn.Linear(embed_dim, embed_dim)
		self.additional = add_emb
		E_init = torch.rand((self.max_seq, int(self.dh)), dtype = torch.float32)
		E = nn.Parameter(E_init, requires_grad=True)
		self.register_parameter("E", E)

	def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, need_weights=True, pitch_rel=None,pitch_rel_mask=None, dur_rel=None, dur_rel_mask=None):
		"""
		:param inputs: a list of tensors. i.e) [Q, K, V]
		:param mask: mask tensor
		:param kwargs:
		:return: final tensor ( output of attention )
		"""

		batch_size, seq_length, embed_dim = q.size()

		q, k, v = self.Wq(q), self.Wk(k), self.Wv(v) #batch, seq, dim

		q, k, v = q.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3),\
				k.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3),\
				v.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3)

		self.len_k = k.size(2)
		self.len_q = q.size(2)

		E = self.E[self.max_seq-q.shape[-2]:, :] 

		QE = torch.matmul(q, E.permute(1, 0))#batch, n_head, len_q, dim * len_k, dim ==> batch, n_head, len_q, len_k

		QE = self._qe_masking(QE)

		Srel = self._skewing(QE)

		Kt = k.permute(0, 1, 3, 2)
		QKt = torch.matmul(q, Kt)
		logits = QKt + Srel
		logits = logits / math.sqrt(self.dh)

		if attn_mask is not None or key_padding_mask is not None:
			attn_mask = attn_mask[None, None, ...]
			key_padding_mask = key_padding_mask[:, None, :, None]
			mask = torch.logical_or(key_padding_mask, attn_mask!=0)
			logits = logits.masked_fill(mask , -9e15)

		attention_weights = F.softmax(logits, -1)
		attention = torch.matmul(attention_weights, v)

		out = attention.permute(0, 2, 1, 3)
		out = torch.reshape(out, (out.size(0), -1, self.d))

		out = self.fc(out)
		out = self.dropout(out)
		return out, attention_weights


	def _skewing(self, tensor: torch.Tensor):
		padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0])
		reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
		Srel = reshaped[:, :, 1:, :]
		if self.len_k > self.len_q:
			Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
		elif self.len_k < self.len_q:
			Srel = Srel[:, :, :, :self.len_k]

		return Srel

	@staticmethod
	def _qe_masking(qe):
		mask = sequence_mask(
			torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
			qe.size()[-1])
		mask = ~mask.to(mask.device)
		return mask.to(qe.dtype) * qe

class MultiheadAttention_myversion(nn.Module):

	def __init__(self, embed_dim=256, num_heads=4, dropout = 0.2, add_emb=False ,max_seq=245, batch_first=True):
		super().__init__()
		print(f"using vanilla transformer with multi-head attn!")

		self.dropout = nn.Dropout(p=dropout)
		self.len_k = None
		self.max_seq = max_seq
		self.h = num_heads
		self.d = embed_dim
		self.dh = embed_dim // num_heads
		self.Wq = torch.nn.Linear(self.d, self.d)
		self.Wk = torch.nn.Linear(self.d, self.d)
		self.Wv = torch.nn.Linear(self.d, self.d)
		self.fc = torch.nn.Linear(embed_dim, embed_dim)
		self.additional = add_emb

	def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, need_weights=True, pitch_rel=None,pitch_rel_mask=None, dur_rel=None, dur_rel_mask=None):
		"""
		:param inputs: a list of tensors. i.e) [Q, K, V]
		:param mask: mask tensor
		:param kwargs:
		:return: final tensor ( output of attention )
		"""

		batch_size, seq_length, embed_dim = q.size()

		q, k, v = self.Wq(q), self.Wk(k), self.Wv(v) #batch, seq, dim

		q, k, v = q.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3),\
				k.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3),\
				v.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3)

		self.len_k = k.size(2)
		self.len_q = q.size(2)

		Kt = k.permute(0, 1, 3, 2)
		QKt = torch.matmul(q, Kt)
		logits = QKt 
		logits = logits / math.sqrt(self.dh)


		if attn_mask is not None or key_padding_mask is not None:
			attn_mask = attn_mask[None, None, ...]
			key_padding_mask = key_padding_mask[:, None, :, None]
			mask = torch.logical_or(key_padding_mask, attn_mask!=0)
			logits = logits.masked_fill(mask , -9e15)

		attention_weights = F.softmax(logits, -1)
		attention = torch.matmul(attention_weights, v)

		out = attention.permute(0, 2, 1, 3)
		out = torch.reshape(out, (out.size(0), -1, self.d))

		out = self.fc(out)
		out = self.dropout(out)
		return out, attention_weights

class RelativeGlobalAttention_relative_index_pitch_onset(nn.Module):
	def __init__(self, embed_dim=256, num_heads=4, dropout = 0.2, add_emb=False ,max_seq=246, if_add_relative_pitch=True , if_add_relative_duration=True, if_add_relative_idx = False,if_add_relative_idx_no_mask = False, batch_first=True):
		super().__init__()
		self.dropout = nn.Dropout(p=dropout)
		self.len_k = None
		self.max_seq = max_seq
		self.h = num_heads
		self.d = embed_dim
		self.dh = embed_dim // num_heads
		self.Wq = torch.nn.Linear(self.d, self.d)
		self.Wk = torch.nn.Linear(self.d, self.d)
		self.Wv = torch.nn.Linear(self.d, self.d)
		self.fc = torch.nn.Linear(embed_dim, embed_dim)
		self.additional = add_emb
		self.if_add_relative_pitch=if_add_relative_pitch
		self.if_add_relative_duration = if_add_relative_duration
		self.if_add_relative_idx = if_add_relative_idx
		self.if_add_relative_idx_no_mask = if_add_relative_idx_no_mask
		if self.if_add_relative_idx:
			E_init = torch.rand((self.max_seq, int(self.dh)), dtype = torch.float32)
			E = nn.Parameter(E_init, requires_grad=True)
			self.register_parameter("E", E)
		if self.if_add_relative_idx_no_mask:
			E_init = torch.rand((2*self.max_seq-1, int(self.dh)), dtype = torch.float32)
			E = nn.Parameter(E_init, requires_grad=True)
			self.register_parameter("E", E)

		self.pitch_relnn = torch.nn.Linear(self.d, self.d) 
		self.dur_relnn = torch.nn.Linear(self.d, self.d)

	def forward(self, q, k, v, attn_mask=None, key_padding_mask=None, need_weights=True, pitch_rel=None,pitch_rel_mask=None, dur_rel=None, dur_rel_mask=None):
		batch_size, seq_length, embed_dim = q.size()
		q, k, v = self.Wq(q), self.Wk(k), self.Wv(v) #batch, seq, dim

		q, k, v = q.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3),\
				k.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3),\
				v.reshape(batch_size, seq_length, self.h, self.dh).permute(0, 2, 1, 3)
		batch_size = q.size(0)
		self.len_k = k.size(2)
		self.len_q = q.size(2)

		Kt = k.permute(0, 1, 3, 2) #batch, head, len, dim --> batch, head, dim, len
		QKt = torch.matmul(q, Kt) # batch, head, len_q, dim_q * batch, head, dim_k, len_k
		Srel = 0
		if self.if_add_relative_idx:
			E = self.E[self.max_seq-q.shape[-2]:, :] # #be careful with the self.max_seq! 
			QE = torch.matmul(q, E.permute(1, 0))#batch, n_head, len_q, dim * len_k, dim ==> batch, n_head, len_q, len_k
			QE = self._qe_masking(QE)
			Srel_idx = self._skewing(QE)
			Srel += Srel_idx

		if self.if_add_relative_idx_no_mask:
			
			if self.max_seq-q.shape[-2]!=0:
				E = self.E[self.max_seq-q.shape[-2]:-self.max_seq+q.shape[-2], :]  #be careful with the self.max_seq! 
			else:
				E = self.E #here assume max_len == input_seq_len #original
			
			QE = torch.matmul(q, E.permute(1, 0))#batch, n_head, len_q, dim * len_k, dim ==> batch, n_head, len_q, len_k
			Srel_idx = self._skewing_no_mask(QE)
			Srel += Srel_idx

		if self.if_add_relative_pitch==True:
			pitch_rel = self.pitch_relnn(pitch_rel)
			pitch_rel_perm = pitch_rel.reshape(batch_size,self.len_q, self.len_k, self.h, self.dh).permute(0, 3, 1, 4, 2) ##batch, len_q, len_k, head, dim_rel--> batch, head, len_q, dim, len_k
			q_add_dim = q[:, :, :, None, :]# batch, head, len_q, 1, dim_q 
			Srel_pitch = torch.matmul(q_add_dim, pitch_rel_perm) # batch, head, len_q, 1, dim_q * batch, head, len_q, dim, len_k ==> batch, head, len_q, 1, len_k
			Srel_pitch = Srel_pitch[:, :, :, 0, :]
			Srel+=Srel_pitch

		if self.if_add_relative_duration==True:
			dur_rel = self.dur_relnn(dur_rel)
			dur_rel_perm = dur_rel.reshape(batch_size,self.len_q, self.len_k, self.h, self.dh).permute(0, 3, 1, 4, 2)
			q_add_dim = q[:, :, :, None, :]# batch, head, len_q, 1, dim_q 
			Srel_dur = torch.matmul(q_add_dim, dur_rel_perm) # batch, head, len_q, 1, dim_q * batch, head, len_q, dim, len_k ==> batch, head, len_q, 1, len_k
			Srel_dur = Srel_dur[:, :, :, 0, :]
			Srel+=Srel_dur

		logits = QKt + Srel
		logits = logits / math.sqrt(self.dh)

		if self.if_add_relative_idx_no_mask: #dont use look ahead mask
			mask = key_padding_mask[:, None, :, None]
			logits = logits.masked_fill(mask , -9e15)
		elif attn_mask is not None or key_padding_mask is not None: #use both look ahead mask and pad mask
			attn_mask = attn_mask[None, None, ...]
			key_padding_mask = key_padding_mask[:, None, :, None]
			mask = torch.logical_or(key_padding_mask, attn_mask!=0)
			logits = logits.masked_fill(mask , -9e15)


		attention_weights = F.softmax(logits, -1)
		attention = torch.matmul(attention_weights, v)

		out = attention.permute(0, 2, 1, 3)
		out = torch.reshape(out, (out.size(0), -1, self.d))

		out = self.fc(out)
		out = self.dropout(out)
		return out, attention_weights

	def _skewing(self, tensor: torch.Tensor):
		padded = F.pad(tensor, [1, 0, 0, 0, 0, 0, 0, 0]) #batch, head, len, len 
		reshaped = torch.reshape(padded, shape=[padded.size(0), padded.size(1), padded.size(-1), padded.size(-2)])
		Srel = reshaped[:, :, 1:, :]
		if self.len_k > self.len_q:
			Srel = F.pad(Srel, [0, 0, 0, 0, 0, 0, 0, self.len_k-self.len_q])
		elif self.len_k < self.len_q:
			Srel = Srel[:, :, :, :self.len_k]

		return Srel
	def _skewing_no_mask(self, tensor: torch.Tensor):
		padded = F.pad(tensor, [0, 1, 0, 0, 0, 0, 0, 0]) #batch, head, len, 2*len-1 ==> batch, head, len, 2*len
		flattened = torch.reshape(padded, shape=[padded.size(0), padded.size(1), -1]) #==> batch, head, 2*len^2 

		zero_pad = torch.zeros(padded.size(0), padded.size(1), padded.size(2)-1).to(padded.device) # batch, head, len-1 
		flattened_zero_pad = torch.cat((flattened, zero_pad), dim = -1) #batch, head, 2*len^2+len-1 
		reshaped = torch.reshape(flattened_zero_pad, (padded.size(0), padded.size(1), padded.size(2)+1, 2*padded.size(2)-1))

		Srel = reshaped[:, :, :padded.size(2), -padded.size(2):]
		return Srel

	@staticmethod
	def _qe_masking(qe):
		mask = sequence_mask(
			torch.arange(qe.size()[-1] - 1, qe.size()[-1] - qe.size()[-2] - 1, -1).to(qe.device),
			qe.size()[-1])
		mask = ~mask.to(mask.device)
		return mask.to(qe.dtype) * qe

class TransformerEncoderLayer_type_selection(nn.Module):

	__constants__ = ['batch_first', 'norm_first']

	def __init__(self, d_model: int, nhead: int, dim_feedforward: int = 2048, dropout: float = 0.1,
				 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu,
				 layer_norm_eps: float = 1e-5, batch_first: bool = False, norm_first: bool = False,
				 device=None, dtype=None,  attention_type= "rgl_rel_pitch_dur", if_add_relative_pitch= True, if_add_relative_duration= True, if_add_relative_idx =False, if_add_relative_idx_no_mask = False) -> None:
		factory_kwargs = {'device': device, 'dtype': dtype}
		super(TransformerEncoderLayer_type_selection, self).__init__()
		self.attention_type = attention_type
		if attention_type=="mha":
			print("attention type: multihead_vanilla")
			self.self_attn = MultiheadAttention_myversion(d_model, nhead, dropout=dropout, batch_first=batch_first) #why does this need drop out? pending change
		elif attention_type=="rgl_vanilla":
			print("attention type: rgl_vanilla--> music transformer")
			self.self_attn = RelativeGlobalAttention_my_version(d_model, nhead, dropout=dropout, batch_first=batch_first) #night safari
		elif attention_type=="rgl_rel_pitch_dur":
			print("attention type: relative index pitch onset --> RIPO transformer")
			self.self_attn = RelativeGlobalAttention_relative_index_pitch_onset(d_model, nhead, dropout=dropout, if_add_relative_pitch = if_add_relative_pitch, if_add_relative_duration = if_add_relative_duration,  if_add_relative_idx=if_add_relative_idx,if_add_relative_idx_no_mask = if_add_relative_idx_no_mask, batch_first=batch_first)

		# Implementation of Feedforward model
		self.linear1 = Linear(d_model, dim_feedforward, **factory_kwargs)
		self.dropout = Dropout(dropout)
		self.linear2 = Linear(dim_feedforward, d_model, **factory_kwargs)

		self.norm_first = norm_first
		self.norm1 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.norm2 = LayerNorm(d_model, eps=layer_norm_eps, **factory_kwargs)
		self.dropout1 = Dropout(dropout)
		self.dropout2 = Dropout(dropout)
		self.dropout3 = Dropout(dropout)

		# Legacy string support for activation function.
		if isinstance(activation, str):
			self.activation = _get_activation_fn(activation)
		else:
			self.activation = activation

	def __setstate__(self, state):
		if 'activation' not in state:
			state['activation'] = F.relu
		super(TransformerEncoderLayer_type_selection, self).__setstate__(state)

	def forward(self, src: Tensor, src_mask: Optional[Tensor] = None, src_key_padding_mask: Optional[Tensor] = None , pitch_rel=None,pitch_rel_mask=None, dur_rel=None, dur_rel_mask=None) -> Tensor:
		r"""Pass the input through the encoder layer.
		Args:
			src: the sequence to the encoder layer (required).
			src_mask: the mask for the src sequence (optional).
			src_key_padding_mask: the mask for the src keys per batch (optional).
		Shape:
			see the docs in Transformer class.
		"""

		

		x = src

		attn_logits, attn_weights = self._sa_block(self.norm1(x), attn_mask = src_mask, key_padding_mask = src_key_padding_mask, pitch_rel=pitch_rel,
																																pitch_rel_mask=pitch_rel_mask, 
																																dur_rel=dur_rel, 
																																dur_rel_mask=dur_rel_mask)
		if self.norm_first:
			x = x + attn_logits
			x = x + self._ff_block(self.norm2(x))
		else: #choose this as ori! see Fig. 1 of https://arxiv.org/pdf/2002.04745v1.pdf
			x = self.norm1(x + attn_logits)
			x = self.norm2(x + self._ff_block(x))

		return x, attn_weights # I changed to return weight too!

	# self-attention block
	def _sa_block(self, x: Tensor,
				  attn_mask: Optional[Tensor], key_padding_mask: Optional[Tensor], pitch_rel=None,pitch_rel_mask=None, dur_rel=None, dur_rel_mask=None) -> Tensor:
		
		x, attn_weight = self.self_attn(x, x, x,
						   attn_mask=attn_mask,
						   key_padding_mask=key_padding_mask,
						   pitch_rel=pitch_rel,
						   pitch_rel_mask=pitch_rel_mask, 
						   dur_rel=dur_rel, 
						   dur_rel_mask=dur_rel_mask)
		return self.dropout1(x), self.dropout3(attn_weight)

	# feed forward block
	def _ff_block(self, x: Tensor) -> Tensor:
		x = self.linear2(self.dropout(self.activation(self.linear1(x))))
		return self.dropout2(x)

def scaled_dot_product(q, k, v, attn_mask= None, key_padding_mask=None):
	#attn_mask: len, len, pad_mask: batch, len
	d_k = q.size()[-1]
	attn_logits = torch.matmul(q, k.transpose(-2, -1))
	attn_logits = attn_logits / math.sqrt(d_k)
	if attn_mask is not None or key_padding_mask is not None:
		attn_mask = attn_mask[None, None, ...]
		key_padding_mask = key_padding_mask[:, None, :, None]
		mask = torch.logical_or(key_padding_mask, attn_mask!=0)
		attn_logits = attn_logits.masked_fill(mask, -9e15)

	attention = F.softmax(attn_logits, dim=-1)
	values = torch.matmul(attention, v)
	return values, attention

def _get_clones(module, N):
	return ModuleList([copy.deepcopy(module) for i in range(N)])

def _get_activation_fn(activation):
	if activation == "relu":
		return F.relu
	elif activation == "gelu":
		return F.gelu

	raise RuntimeError("activation should be relu/gelu, not {}".format(activation))

def sequence_mask(length, max_length=None):
	if max_length is None:
		max_length = length.max()
	x = torch.arange(max_length, dtype=length.dtype, device=length.device)
	return x.unsqueeze(0) < length.unsqueeze(1)

if __name__ == "__main__":
	torch.set_printoptions(precision = 2)
	max_seq = 10

	rgl = RelativeGlobalAttention_relative_index_pitch_onset(max_seq=max_seq,if_add_relative_pitch=True , if_add_relative_duration=True, if_add_relative_idx = False,if_add_relative_idx_no_mask = True)
	E = rgl.E 
	q = torch.ones(1, 8, max_seq, 64)
	q2 = torch.ones(1, 8, max_seq-3, 64)
	QE = torch.matmul(q, E.permute(1, 0))#batch, n_head, len_q, dim * len_k, dim ==> batch, n_head, len_q, len_k
	print(rgl.max_seq-q2.shape[-2], -rgl.max_seq+q2.shape[-2])
	QE2 = torch.matmul(q2, E[rgl.max_seq-q2.shape[-2]:-rgl.max_seq+q2.shape[-2], :].permute(1, 0))#batch, n_head, len_q, dim * len_k, dim ==> batch, n_head, len_q, len_k

	print(f"E:{E.shape, QE.shape, QE2.shape}QE:{QE[0,0,:, :]} QE2:{QE2[0,0,:, :]}")
	Srel_idx = rgl._skewing_no_mask(QE)
	Srel_idx2 = rgl._skewing_no_mask(QE2)
	print(f"check srel{Srel_idx.shape, Srel_idx2.shape}:{Srel_idx[0,0,:, :]},check srel2:{Srel_idx2[0,0,:, :]}")