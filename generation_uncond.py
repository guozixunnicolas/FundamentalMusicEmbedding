
import numpy as np
from data_processing.data_loading import tokenizer_plain, detokenizer_plain
from data_processing.data_loading import MotifDataset 
from model.model import RIPO_transformer, loss_function_baseline
from data_processing.utils import Sampler_torch, write_midi_monophonic
from utils.eval_utils import get_rep_seq, get_unique_tokens, get_unique_intervals

import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

import yaml
import time
from datetime import datetime
import os
import glob
import json
from tqdm import tqdm
import matplotlib.pyplot as plt
class generator():
	def __init__(self, tokenizer, detokenizer, sampler,model):
		self.tokenizer = tokenizer
		self.detokenizer = detokenizer
		self.sampler = sampler
		self.model = model
	
	def cal_sum_dur(self, dur):
		dur = dur.clone().cpu().detach().numpy()
		dur_decoded = [self.detokenizer.decode_dur[x] for x in dur[0]]
		return sum(dur_decoded)

	def cal_cumsum_dur(self, dur):
		dur_cpu = dur.clone().cpu().detach().numpy()
		dur_decoded = torch.tensor([0.]+[self.detokenizer.decode_dur[x] for x in dur_cpu[0]][:-1]).to(dur.device)
		out = torch.cumsum(dur_decoded, dim = -1)[None, ...]
		return out

	def get_rel_mask(self, inp, tpe="pitch"):
		if tpe =="pitch":
			mask0 = inp==0
			mask1 = inp==1
			mask2 = inp==2
			mask_01= torch.logical_or(mask0, mask1)
			mask_012= torch.logical_or(mask_01, mask2)
			mask_matrix = torch.logical_or(mask_012[..., None], mask_012[None, ...])
		if tpe =="dur":
			mask0 = inp==0
			mask_matrix = torch.logical_or(mask0[..., None], mask0[None, ...])

		return mask_matrix
	
	def generate(self, inp_dict, target_bar_num):
		max_len = 999
		sum_dur = self.cal_sum_dur(inp_dict['dur_p']) #batch, len
		i = len(inp_dict['pitch'][0])-1

		#model recursive generate
		while sum_dur<=target_bar_num*4 and inp_dict['pitch'].shape[-1]<=245: 
			pred_dict = self.model(inp_dict)
			pitch_pred_logits = pred_dict["pitch_pred"][:, -1, :]
			dur_pred_logits = pred_dict["dur_pred"][:, -1, :]
			pitch_pred = self.sampler(pitch_pred_logits).to(inp_dict['pitch'].device)
			dur_pred = self.sampler(dur_pred_logits).to(inp_dict['pitch'].device)

			#get the probabilty
			pitch_prob = F.softmax(pitch_pred_logits, dim = -1)[0, pitch_pred[0]]
			dur_prob = F.softmax(dur_pred_logits, dim = -1)[0, dur_pred[0]]

			inp_dict["pitch_prob"].append(pitch_prob)
			inp_dict["dur_prob"].append(dur_prob)

			inp_dict["pitch"] = torch.cat((inp_dict["pitch"], pitch_pred), dim = -1)
			inp_dict["dur_p"] = torch.cat((inp_dict["dur_p"], dur_pred), dim = -1)
			sum_dur = self.cal_sum_dur(inp_dict['dur_p']) 
			
			#calculate pitch_rel, dur_rel, pitch_rel_mask, dur_rel_mask, dur_cumsum
			inp_dict["pitch_rel"] = inp_dict["pitch"][:, :, None] - inp_dict["pitch"][:, None, :]
			inp_dict["dur_rel"] = inp_dict["dur_p"][:, :, None] - inp_dict["dur_p"][:, None, :]

			inp_dict["pitch_rel_mask"] = self.get_rel_mask(inp_dict["pitch"], tpe="pitch")
			inp_dict["dur_rel_mask"] = self.get_rel_mask(inp_dict["dur_p"], tpe="dur")
			inp_dict["dur_onset_cumsum"] = self.cal_cumsum_dur(inp_dict["dur_p"])

		#detokenizer output 
		out_pitch = inp_dict["pitch"].cpu().detach().numpy()[0]
		out_dur = inp_dict["dur_p"].cpu().detach().numpy()[0]
		out_note = [[self.detokenizer.decode_pitch[p], self.detokenizer.decode_dur[d]] for p,d in zip(out_pitch, out_dur)]
		inp_dict["pitch_prob"] = torch.stack(inp_dict["pitch_prob"], dim = -1)
		inp_dict["dur_prob"] = torch.stack(inp_dict["dur_prob"], dim = -1)
		out_dict = {}
		for key, value in inp_dict.items():
			out_dict[key] = inp_dict[key].cpu().detach().numpy()[0].tolist()
		return out_note, out_dict

def trim_seed_music(inp, dur, seed_bar_num, detokenizer):
	dur_decoded = [detokenizer.decode_dur[x] for x in dur[0]] #batch, len ==> len
	cum_sum = np.cumsum([0]+[x for x in dur_decoded])
	if cum_sum[-1]<seed_bar_num*4:
		where_until = len(cum_sum)
	else:
		where_until = np.where(cum_sum>=seed_bar_num*4)[0][0]
	return inp[:, :where_until]

if __name__ =="__main__":
	#config file should contain the folder path where the ckpt is stored
	gen_config_path = "config/generation_uncond.yaml"
	with open (gen_config_path, 'r') as f:
		gen_cfg = yaml.safe_load(f)
	ckpt_path = gen_cfg["ckpt_path"]
	device = gen_cfg["device"] 
	seed_bar_num = gen_cfg['seed_bar_num']
	target_bar_num = gen_cfg['target_bar_num']
	
	#model config & load model 
	with open (glob.glob(ckpt_path+"/*.yaml")[0], 'r') as f:
		model_cfg = yaml.safe_load(f)
		model_cfg['device'] = device
		model_cfg['relative_pitch_attention']['device'] = device
		print("model config: ",model_cfg)
	
	model = RIPO_transformer(**model_cfg['relative_pitch_attention']).to(device)
	model.device = device
	model.pos_encoder.global_time_embedding.device = device
	model.pos_encoder.modulo_time_embedding.device = device
	model.pos_encoder.modulo_time_embedding.angles = model.pos_encoder.modulo_time_embedding.angles.to(device)
	model.pos_encoder.global_time_embedding.angles = model.pos_encoder.global_time_embedding.angles.to(device)

	if model_cfg['relative_pitch_attention']['dur_embedding_conf']['type']=="se": #if FME
		model.dur_embedding.device = device
		model.dur_embedding.to(device)
		model.dur_embedding.angles = model.dur_embedding.angles.to(device)
	if model_cfg['relative_pitch_attention']['pitch_embedding_conf']['type']=="se": #if FME
		model.pitch_embedding.device = device
		model.pitch_embedding.to(device)
		model.pitch_embedding.angles = model.pitch_embedding.angles.to(device)

	model.load_state_dict(torch.load(ckpt_path+"/best_loss.pth"))
	model.eval()

	#tokenizer & detokenizer
	tokenizer = tokenizer_plain() #gen_cfg['tknz']
	detokenizer = detokenizer_plain()
	sampler = Sampler_torch(**gen_cfg['sampling'])
	generator = generator(tokenizer, detokenizer, sampler, model)

	#define saving path and save generation config
	save_dir = os.path.join(ckpt_path, f"results_{gen_cfg['sampling']['decoder_choice']}_{gen_cfg['sampling']['temperature']}_{gen_cfg['sampling']['top_k']}_{gen_cfg['sampling']['top_p']}_{gen_cfg['seed_bar_num']}_{gen_cfg['target_bar_num']}")
	print(f"saving to:{save_dir}")
	os.makedirs(save_dir, exist_ok=True)
	with open(os.path.join(save_dir, "gen_config.yaml"), 'w') as f:
		documents = yaml.dump(gen_cfg, f)

	#load data 
	dataset = MotifDataset(**model_cfg["dataset"])	
	train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)], generator=torch.Generator().manual_seed(0)) #using the same seed to prevent data leaking (use valid set which is not seen during training for generation)
	print(f"valid:{len(valid_dataset)}")
	valid_dataloader = DataLoader(valid_dataset, batch_size=1, shuffle=False)

	#generation
	for i,data in tqdm(enumerate(valid_dataloader)):
		try:
			pitch,dur= data[0]['pitch'].numpy(),data[0]['dur_p'].numpy()
			pitch_rel, pitch_rel_mask, dur_rel, dur_rel_mask, dur_cumsum = data[0]['pitch_rel'].numpy(), data[0]['pitch_rel_mask'].numpy(), data[0]['dur_rel'].numpy(), data[0]['dur_rel_mask'].numpy(), data[0]['dur_onset_cumsum'].numpy()
			pitch_drop_pad, dur_drop_pad = detokenizer.drop_pad_new(lst = pitch), detokenizer.drop_pad_new(lst = dur)	
			pitch_drop_pad_trim, dur_drop_pad_trim = trim_seed_music(pitch_drop_pad, dur_drop_pad, seed_bar_num, detokenizer),\
																										trim_seed_music(dur_drop_pad, dur_drop_pad, seed_bar_num, detokenizer)

			trim_len = len(pitch_drop_pad_trim[0])
			pitch_rel_trim, pitch_rel_mask_trim, dur_rel_trim, dur_rel_mask_trim, dur_cumsum_trim = pitch_rel[:, :trim_len, :trim_len], pitch_rel_mask[:, :trim_len, :trim_len], dur_rel[:, :trim_len, :trim_len], dur_rel_mask[:, :trim_len, :trim_len], dur_cumsum[:, :trim_len]
			pitch_drop_pad_trim, dur_drop_pad_trim,pitch_rel_trim, pitch_rel_mask_trim, dur_rel_trim, dur_rel_mask_trim, dur_cumsum_trim = torch.tensor(pitch_drop_pad_trim).to(device),\
																											torch.tensor(dur_drop_pad_trim).to(device),\
																											torch.tensor(pitch_rel_trim).to(device),\
																											torch.tensor(pitch_rel_mask_trim).to(device),\
																											torch.tensor(dur_rel_trim).to(device),\
																											torch.tensor(dur_rel_mask_trim).to(device),\
																											torch.tensor(dur_cumsum_trim).to(device)

			inp_dict = {"pitch":pitch_drop_pad_trim,"dur_p":dur_drop_pad_trim, "pitch_rel":pitch_rel_trim, "pitch_rel_mask":pitch_rel_mask_trim, "dur_rel":dur_rel_trim,"dur_rel_mask": dur_rel_mask_trim, "dur_onset_cumsum":dur_cumsum_trim, "pitch_prob":[], "dur_prob":[]} 
			out_note,out_dict = generator.generate(inp_dict, target_bar_num=target_bar_num)
			write_midi_monophonic(out_note, chord_list = [], mid_name = os.path.join(save_dir, f"{str(i)}.mid"))
			with open(os.path.join(save_dir, f"{str(i)}.json"), 'w') as f:
				json.dump(out_dict, f)
		except:
			continue
	