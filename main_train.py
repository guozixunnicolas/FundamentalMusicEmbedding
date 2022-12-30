import sys
import numpy as np
from model.model import RIPO_transformer, loss_function_baseline
import torch.nn as nn
import torch
import torch.nn.functional as F

from data_processing.data_loading import MotifDataset 
from torch.utils.data import DataLoader
import json
import yaml
import time
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import os

def train(model):
	model.train()  # turn on train mode
	total_loss = 0.
	total_pitch_loss = 0.
	total_dur_loss = 0.
	log_interval = len(train_dataloader)-1 #20
	start_time = time.time()

	for batch,data in enumerate(train_dataloader):		
		inp_dict, gt_dict = {}, {}
		for key, val in data[0].items():
			inp_dict[key] = val.to(dvc)
		for key, val in data[1].items():
			gt_dict[key] = val.to(dvc)
		
		pred_dict = model(inp_dict)
		loss_dict = loss_function_baseline(pred_dict, gt_dict)
		optimizer.zero_grad()
		loss_dict['total_loss'].backward()
		torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
		optimizer.step()

		total_loss += loss_dict['total_loss'].item()
		total_pitch_loss += loss_dict['pitch_loss'].item()
		total_dur_loss += loss_dict['dur_loss'].item()

		if batch % log_interval == 0 and batch > 0:

			lr = scheduler.get_last_lr()[0]
			ms_per_batch = (time.time() - start_time) * 1000 / (log_interval+1)
			cur_loss = total_loss / (log_interval+1) 
			cur_pitch_loss = total_pitch_loss / (log_interval+1)
			cur_dur_loss = total_dur_loss / (log_interval+1)

			writer.add_scalar('training total loss',cur_loss,epoch )
			writer.add_scalar('training pitch loss',cur_pitch_loss,epoch )
			writer.add_scalar('training dur loss',cur_dur_loss,epoch )
			print(f'| epoch {epoch:3d} | {batch:5d} batches | '
				  f'lr {lr:02.2f} | ms/batch {ms_per_batch:5.2f} | '
				  f'loss {cur_loss:5.4f} |')
			total_loss = 0
			total_pitch_loss = 0
			total_dur_loss = 0
			start_time = time.time()

def evaluate(model):
	model.eval()  # turn on evaluation mode
	return_total_loss = 0.
	total_loss = 0.
	total_pitch_loss = 0.
	total_dur_loss = 0.
	log_interval = len(valid_dataloader)-1
	with torch.no_grad():
		for batch,data in enumerate(valid_dataloader):
			inp_dict, gt_dict = {}, {}
			for key, val in data[0].items():
				inp_dict[key] = val.to(dvc)
			for key, val in data[1].items():
				gt_dict[key] = val.to(dvc)

			pred_dict = model(inp_dict)
			loss_dict = loss_function_baseline(pred_dict, gt_dict)

			return_total_loss+=loss_dict['total_loss'].item()
			total_loss += loss_dict['total_loss'].item()
			total_pitch_loss +=loss_dict['pitch_loss'].item()
			total_dur_loss +=loss_dict['dur_loss'].item()

			if batch % log_interval == 0 and batch > 0:
				cur_loss = total_loss / (log_interval+1) 
				cur_pitch_loss = total_pitch_loss / (log_interval+1)
				cur_dur_loss = total_dur_loss / (log_interval+1)

				writer.add_scalar('valid total loss',cur_loss,epoch)
				writer.add_scalar('valid pitch loss',cur_pitch_loss,epoch)
				writer.add_scalar('valid dur loss',cur_dur_loss,epoch)

				total_loss = 0
				total_pitch_loss = 0
				total_dur_loss = 0
			
	return return_total_loss / len(valid_dataloader)

if __name__ =="__main__":
	torch.set_printoptions(precision = 2)
	np.set_printoptions(precision = 2)
	config_path = "config/ripo_transformer.yaml" 
	
	#config 
	with open (config_path, 'r') as f:
		cfg = yaml.safe_load(f)
	dvc = cfg['device']

	#define model
	model = RIPO_transformer(**cfg['relative_pitch_attention']).to(dvc)

	#Loss func
	criterion = nn.CrossEntropyLoss()

	#optimizer
	if cfg['optimizer']=="adam":
		optimizer = torch.optim.Adam(model.parameters(), lr=cfg['lr'])
	elif cfg['optimizer']=="sgd":
		optimizer = torch.optim.SGD(model.parameters(), lr=cfg['lr'])
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1.0, gamma=0.95)

	#tensorboard
	#checkpoint_path
	date_time = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")+"_baseline"
	checkpoint_path = os.path.join("checkpoints", date_time)
	writer = SummaryWriter(checkpoint_path)

	#save training config
	with open(os.path.join(checkpoint_path, config_path.split("/")[-1]), 'w') as f:
		documents = yaml.dump(cfg, f)

	#dataloading 
	dataset = MotifDataset(**cfg["dataset"])	
	train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [int(len(dataset)*0.9), len(dataset) - int(len(dataset)*0.9)], generator=torch.Generator().manual_seed(0))
	train_dataloader = DataLoader(train_dataset, batch_size=cfg['batch_size'], shuffle=True)
	valid_dataloader = DataLoader(valid_dataset, batch_size=cfg['batch_size'], shuffle=True)
	print(f"total len dataset:{len(dataset)}, training set:{len(train_dataset)}, validation set:{len(valid_dataset)}")
	print(f"len train loader:{len(train_dataloader)}, {len(valid_dataloader)}")
	print("trainable parameters:")
	for name, param in model.named_parameters():
		if param.requires_grad:
			print (name)

	best_val_loss = float('inf')
	best_model = None
	
	for epoch in range(cfg['epochs']):
		epoch_start_time = time.time()
		train(model)
		val_loss = evaluate(model)
		elapsed = time.time() - epoch_start_time
		print('-' * 89)
		print(f'| end of epoch {epoch:3d} | time: {elapsed:5.2f}s | '
			f'valid loss {val_loss:5.4f} |')
		print('-' * 89)
		if val_loss < best_val_loss:
			best_val_loss = val_loss
			torch.save(
				model.state_dict(),
				os.path.join(checkpoint_path, f"state_ep{epoch}_{best_val_loss:5.4f}.pth"),
			)

			torch.save(
				model.state_dict(),
				os.path.join(checkpoint_path, f"best_loss.pth"),
			)
		scheduler.step()

