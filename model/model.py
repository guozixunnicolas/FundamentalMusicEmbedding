import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.layers import TransformerEncoder, TransformerEncoderLayer, TransformerEncoderLayer_type_selection

from model.FME_music_positional_encoding import PositionalEncoding, Fundamental_Music_Embedding, Music_PositionalEncoding
import math
from fast_transformers.builders import TransformerEncoderBuilder
from fast_transformers.masking import TriangularCausalMask
##used to be in the baseline model.py
def loss_function_baseline(output, target):
	padding_msk = target['pitch']!=0 #batch, len
	tfmer_lossp_fn = nn.CrossEntropyLoss(reduction='none')
	tfmer_lossd_fn = nn.CrossEntropyLoss(reduction='none')
	pitch_loss = tfmer_lossp_fn(torch.swapaxes(output['pitch_pred'], -1, -2), target['pitch'])* padding_msk
	dur_loss = tfmer_lossd_fn(torch.swapaxes(output['dur_pred'], -1, -2), target['dur_p']) * padding_msk
	loss_dict = {}
	loss_dict['pitch_loss'] = torch.sum(pitch_loss)/torch.sum(padding_msk)
	loss_dict['dur_loss'] = torch.sum(dur_loss)/torch.sum(padding_msk)
	loss_dict['total_loss'] = loss_dict['pitch_loss']+loss_dict['dur_loss']
	return loss_dict

class FusionLayer_baseline(nn.Module):
	def __init__(self, out_dim = 128, pitch_dim =128, dur_dim = 17):
		super().__init__()
		self.out_dim = out_dim
		self.fusion = nn.Linear(pitch_dim+dur_dim, out_dim)
	def forward(self, pitch, dur_p):
		pitch_dur_tpe_tps = torch.cat((pitch,dur_p), dim = -1)
		return self.fusion(pitch_dur_tpe_tps)

class RIPO_transformer(nn.Module):
	def __init__(self,d_model,nhead,  dim_feedforward, dropout, nlayers,pitch_embedding_conf, dur_embedding_conf, position_encoding_conf, attention_conf ,pitch_dim=128, dur_dim=17, emb_size = 128,device='cuda:0'):
		super().__init__()
		self.model_type = 'Transformer'
		self.pos_encoder = Music_PositionalEncoding(d_model, dropout, **position_encoding_conf) #night safari
		print(f"checking model config:{d_model, nhead, dim_feedforward, dropout, attention_conf}")
		
		if attention_conf['attention_type']!="linear_transformer":
			self.use_linear = False
			encoder_layers = TransformerEncoderLayer_type_selection(d_model, nhead, dim_feedforward, dropout, batch_first = True, **attention_conf) #night safari
			self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		else:
			self.use_linear = True
			self.transformer_encoder = TransformerEncoderBuilder.from_kwargs(
				n_layers=nlayers,
				n_heads=nhead,
				query_dimensions=d_model//nhead,
				value_dimensions=d_model//nhead,
				feed_forward_dimensions=dim_feedforward,
				activation='gelu',
				dropout=dropout,
				attention_type="causal-linear" #causal-linear
			).get()


		self.fusion_layer = FusionLayer_baseline(out_dim = d_model, pitch_dim = d_model, dur_dim = d_model)     #combines multiple types of input
		self.pitch_embedding_conf = pitch_embedding_conf
		self.dur_embedding_conf = dur_embedding_conf

		if self.pitch_embedding_conf['type']=="nn":
			print("end2end trainable embedding for pitch")
			self.pitch_embedding = nn.Embedding(pitch_dim,d_model) #pending change for rest and sustain! #changed from embsize to d model
		elif self.pitch_embedding_conf['type']=="se":
			print("FME for pitch")
			self.pitch_embedding = Fundamental_Music_Embedding(**pitch_embedding_conf) 
			self.pitch_embedding_supplement = nn.Embedding(3,d_model) #pad:0, rest:1, sustain:2
			self.relative_pitch_embedding_supplement = nn.Embedding(1,d_model) #hosting non-quantifiable relative pitch (to pad, rest, sustain)
			self.pitch_senn = nn.Linear(d_model, d_model)
		elif self.pitch_embedding_conf['type']=="nn_pretrain":
			print("end2end trainable embedding (pretrained) for pitch")
			self.pitch_embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.load(self.pitch_embedding_conf['pretrain_emb_path'])), freeze = self.pitch_embedding_conf['freeze_pretrain'])
		
		if self.dur_embedding_conf['type']=="nn":
			print("end2end trainable embedding for duration")
			self.dur_embedding = nn.Embedding(dur_dim, d_model)
		elif self.dur_embedding_conf['type']=="se":
			print("FME for duration")
			self.dur_embedding = Fundamental_Music_Embedding(**dur_embedding_conf) 
			self.dur_embedding_supplement = nn.Embedding(1,d_model) #pad:0
			self.dur_senn = nn.Linear(d_model, d_model)

		elif self.dur_embedding_conf['type']=="nn_pretrain":
			print("end2end trainable embedding (pretrained) for duration")
			self.dur_embedding = nn.Embedding.from_pretrained(torch.from_numpy(np.load(self.dur_embedding_conf['pretrain_emb_path'])), freeze = self.dur_embedding_conf['freeze_pretrain'])
		
		self.pitch_ffn = nn.Linear(d_model, pitch_dim)
		self.dur_ffn = nn.Linear(d_model, dur_dim)

		self.d_model = d_model
		self.device = device
		self.init_weights()

	def init_weights(self) -> None:
		initrange = 0.1
		# self.encoder.weight.data.uniform_(-initrange, initrange)
		if self.pitch_embedding_conf['type']=="nn":
			self.pitch_embedding.weight.data.uniform_(-initrange, initrange)
		elif self.pitch_embedding_conf['type']=="se":
			self.pitch_embedding_supplement.weight.data.uniform_(-initrange, initrange)
			self.relative_pitch_embedding_supplement.weight.data.uniform_(-initrange, initrange)

		
		if self.dur_embedding_conf['type']=="nn":
			self.dur_embedding.weight.data.uniform_(-initrange, initrange)
		elif self.dur_embedding_conf['type']=="se":
			self.dur_embedding_supplement.weight.data.uniform_(-initrange, initrange)

		# self.decoder.bias.data.zero_()
		# self.decoder.weight.data.uniform_(-initrange, initrange)

		self.pitch_ffn.bias.data.zero_()
		self.pitch_ffn.weight.data.uniform_(-initrange, initrange)	
		self.dur_ffn.bias.data.zero_()
		self.dur_ffn.weight.data.uniform_(-initrange, initrange)	
	
	def get_mask(self, inp, type_ = "lookback"):
		#inp shape:(batch, len)
		#https://zhuanlan.zhihu.com/p/353365423
		#https://andrewpeng.dev/transformer-pytorch/
		length = inp.shape[1]
		#lookback & padding
		if type_ == "lookback": #additive mask
			mask = torch.triu(torch.ones(length, length) * float('-inf'), diagonal=1)#https://pytorch.org/docs/stable/generated/torch.nn.Transformer.html#torch.nn.Transformer
			# print("lookback mask", mask)
		elif type_ == "pad":#boolean mask
			mask = (inp == 0)
			# print("pad mask", mask)
		return mask.to(self.device)
	
	def forward(self, inp_dict):
		"""
		Args:
			src: Tensor, shape [seq_len, batch_size]
			src_mask: Tensor, shape [seq_len, seq_len]
		Returns:
			output Tensor of shape [seq_len, batch_size, ntoken]
		"""
		pitch,dur,pitch_rel,pitch_rel_mask, dur_rel, dur_rel_mask, dur_onset_cumsum = inp_dict['pitch'], inp_dict['dur_p'], inp_dict['pitch_rel'], inp_dict['pitch_rel_mask'], inp_dict['dur_rel'], inp_dict['dur_rel_mask'],inp_dict['dur_onset_cumsum']
		
		lookback_mask = self.get_mask(pitch, "lookback")
		pad_mask = self.get_mask(pitch, "pad")

		#emb pitch and dur 
		if self.pitch_embedding_conf['type']=="nn":
			pitch_enc = self.pitch_embedding(pitch) #batch, len, emb_dim
			pitch_rel_enc = None

		elif self.pitch_embedding_conf['type']=="se":
			#when token is 012 use supplement embedding 
			# pitch_enc = torch.where( (pitch==0) | (pitch==1) | (pitch==2), self.pitch_embedding_supplement(pitch), self.pitch_embedding(pitch))
			pitch_sup= torch.where( (pitch==0) | (pitch==1) | (pitch==2), pitch, 0)

			pitch_sup_emb = self.pitch_embedding_supplement(pitch_sup)
			pitch_norm_emb = self.pitch_embedding(pitch)

			pitch = pitch[...,None]
			pitch_enc = torch.where((pitch==0) | (pitch==1) | (pitch==2), pitch_sup_emb, pitch_norm_emb)
			
			pitch_rel_enc = self.pitch_embedding.FMS(pitch_rel) #batch, len,len, dim
			rel_pitch_sup_emb = self.relative_pitch_embedding_supplement(torch.tensor(0).to(self.device))[None, None, None, :]
			pitch_rel_enc = torch.where(pitch_rel_mask[..., None], rel_pitch_sup_emb, pitch_rel_enc)
		
			if self.pitch_embedding_conf['emb_nn']==True:
				pitch_rel_enc = self.pitch_senn(pitch_rel_enc)

		elif self.pitch_embedding_conf['type']=="one_hot":
			pitch_enc = F.one_hot(pitch, num_classes = self.d_model).to(torch.float32) #batch, len, emb_dim
			pitch_rel_enc = None

		elif self.pitch_embedding_conf['type']=="nn_pretrain":
			pitch_enc = self.pitch_embedding(pitch) #batch, len, emb_dim
			pitch_rel_enc = None
		
		if self.dur_embedding_conf['type']=="nn":
			dur_enc = self.dur_embedding(dur) #batch, len, emb_dim
			dur_rel_enc = None
		elif self.dur_embedding_conf['type']=="se":
			#when token is 012 use supplement embedding 
			# dur_enc = torch.where(dur==0, self.dur_embedding_supplement(dur), self.dur_embedding(dur))
			dur_sup = torch.where(dur==0, dur, 0)
			dur_sup_emb = self.dur_embedding_supplement(dur_sup)
			dur_norm_emb = self.dur_embedding(dur)
			dur = dur[..., None]
			dur_enc = torch.where(dur==0, dur_sup_emb, dur_norm_emb)
			dur_rel_enc = self.dur_embedding.FMS(dur_rel)
			if self.dur_embedding_conf['emb_nn']==True:
				dur_rel_enc = self.dur_senn(dur_rel_enc)
		elif self.dur_embedding_conf['type']=="one_hot":
			dur_enc = F.one_hot(dur, num_classes = self.d_model).to(torch.float32) #batch, len, emb_dim
			dur_rel_enc = None
		if self.dur_embedding_conf['type']=="nn_pretrain":
			dur_enc = self.dur_embedding(dur) #batch, len, emb_dim
			dur_rel_enc = None
		
		fused_music_info = self.fusion_layer(pitch_enc, dur_enc)#should include adj as well? pending change
		fused_music_info = fused_music_info * math.sqrt(self.d_model)

		src = self.pos_encoder(fused_music_info, dur_onset_cumsum) #night safari: change here to add duration cumsum too 

		if not self.use_linear:
			latent, _ = self.transformer_encoder(src, mask = lookback_mask, src_key_padding_mask = pad_mask, pitch_rel=pitch_rel_enc, dur_rel=dur_rel_enc)
		else:
			attn_mask = TriangularCausalMask(src.size(1), device=src.device)
			latent = self.transformer_encoder(src,attn_mask) # y: b x s x d_model
		
		#norm pred_using transformer
		pitch_pred = self.pitch_ffn(latent) ###-->insert CE loss here
		dur_pred = self.dur_ffn(latent) ###-->insert CE loss here
		output = {"pitch_pred":pitch_pred, "dur_pred":dur_pred}
		return output
