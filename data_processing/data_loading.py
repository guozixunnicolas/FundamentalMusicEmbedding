import torch
import numpy as np
import glob
import random
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import json
def split_lst(g_lst, split_ratio,seed=666):
	random.seed(seed)
	random.shuffle(g_lst)
	training_data_num = int(len(g_lst)*(1-split_ratio))

	training_lst = g_lst[:training_data_num]
	val_lst = g_lst[training_data_num:] #[(note,chord), (note,chord)...]
	return training_lst, val_lst

class MotifDataset(Dataset):
	def __init__(self,data_dir, if_pad = True, seq_len_chord=None, seq_len_note=None):
		self.json_lst = glob.glob(data_dir+"/theory/**/**/*.json") +glob.glob(data_dir+"/wiki/*.json")
		self.tknz = tokenizer_plain(seq_len_note = seq_len_note, seq_len_chord = seq_len_chord, if_pad = if_pad)
		self.notes_inp, self.notes_pred, self.chords = self.process()
	def process(self):
		note_inp_lst = []
		note_pred_lst = []
		chord_lst = []
		for json_file in self.json_lst:
			try:
				note, chord, pitch_rel, pitch_rel_mask, onset_rel, onset_rel_mask, dur_onset_cumsum = self.tknz(json_file)
				inp_dict, pred_dict = {}, {}
				inp_dict['pitch'] = note[:-1, 0] #(n_nodes, )
				inp_dict['dur_p'] = note[:-1, 1]#(n_nodes, )

				inp_dict['pitch_rel'] = pitch_rel[:-1, :-1] #(n_nodes, )
				inp_dict['pitch_rel_mask'] = pitch_rel_mask[:-1, :-1] #(n_nodes, )
				inp_dict['dur_rel'] = onset_rel[:-1, :-1] #(n_nodes, )
				inp_dict['dur_rel_mask'] = onset_rel_mask[:-1, :-1] #(n_nodes, )
				inp_dict['dur_onset_cumsum'] = dur_onset_cumsum[:-1] #(n_nodes, )

				pred_dict['pitch'] = note[1:, 0] #(n_nodes, )
				pred_dict['dur_p'] = note[1:, 1]#(n_nodes, )

				pred_dict['pitch_rel'] = pitch_rel[1:, 1:] #(n_nodes, )
				pred_dict['pitch_rel_mask'] = pitch_rel_mask[1:, 1:] #(n_nodes, )
				pred_dict['dur_rel'] = onset_rel[1:, 1:] #(n_nodes, )
				pred_dict['dur_rel_mask'] = onset_rel_mask[1:, 1:] #(n_nodes, )
				pred_dict['dur_onset_cumsum'] = dur_onset_cumsum[1:] #(n_nodes, )


				note_inp_lst.append(inp_dict)
				note_pred_lst.append(pred_dict)
				chord_lst.append(chord)
			except:
				continue
		return note_inp_lst, note_pred_lst, chord_lst
	def __getitem__(self, i):
		return self.notes_inp[i], self.notes_pred[i],self.chords[i]

	def __len__(self):
		return len(self.notes_inp)







class tokenizer_plain():
	def __init__(self,seq_len_note=246, seq_len_chord=88,if_pad = True):
		self.duration_dict_inv = {0: -999, 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.25, 6: 1.5, 7: 1.75, 8: 2.0, 9: 2.25, 10: 2.5, 11: 2.75, 12: 3.0, 13: 3.25, 14: 3.5, 15: 3.75, 16: 4.0}

		self.chord_dict2 = {"pad":0, "rest":1,"sustain":2,\
			'Cmaj': 3, 'Cmin': 4, 'Cdim': 5, 'C7': 6, \
			'C#maj': 7, 'C#min': 8, 'C#dim': 9, 'C#7': 10,'Dbmaj': 7, 'Dbmin': 8, 'Dbdim': 9, 'Db7': 10, \
			'Dmaj': 11, 'Dmin': 12, 'Ddim': 13, 'D7': 14, \
			'D#maj': 15, 'D#min': 16,'D#dim': 17, 'D#7': 18, 'Ebmaj': 15, 'Ebmin': 16,'Ebdim': 17, 'Eb7': 18,\
			'Emaj': 19, 'Emin': 20, 'Edim': 21, 'E7': 22, \
			'Fmaj': 23, 'Fmin': 24, 'Fdim': 25, 'F7': 26, \
			'F#maj': 27, 'F#min': 28, 'F#dim': 29, 'F#7': 30, 'Gbmaj': 27, 'Gbmin': 28, 'Gbdim': 29, 'Gb7': 30,\
			'Gmaj': 31, 'Gmin': 32,'Gdim': 33, 'G7': 34, \
			'G#maj': 35, 'G#min': 36, 'G#dim': 37, 'G#7': 38, 'Abmaj': 35, 'Abmin': 36, 'Abdim': 37, 'Ab7': 38,\
			'Amaj': 39, 'Amin': 40, 'Adim': 41, 'A7': 42, \
			'A#maj': 43, 'A#min': 44, 'A#dim': 45, 'A#7': 46, 'Bbmaj': 43, 'Bbmin': 44, 'Bbdim': 45, 'Bb7': 46, \
			'Bmaj': 47, 'Bmin': 48,'Bdim': 49, 'B7': 50}

		self.duration_dict = {"pad":0, 0.25: 1, 0.5: 2, 0.75: 3, 1.0: 4, 1.25: 5, 1.5: 6, 1.75: 7, 2.0: 8,\
							   2.25: 9, 2.5: 10, 2.75: 11, 3.0: 12, 3.25: 13, 3.5: 14, 3.75: 15, 4.0: 16}
					   		
		self.pitch_dict2 = {'pad': 0, 'rest': 1, 'sustain': 2, 'C-1': 3, 'Db-1': 4, 'C#-1': 4, 'D-1': 5, 'Eb-1': 6, 'D#-1': 6, 'E-1': 7, 'F-1': 8, 'Gb-1': 9, 'F#-1': 9, 'G-1': 10, 'Ab-1': 11, 'G#-1': 11, 'A-1': 12, 'Bb-1': 13, 'A#-1': 13, 'B-1': 14, 'C0': 15, 'Db0': 16, 'C#0': 16, 'D0': 17, 'Eb0': 18, 'D#0': 18, 'E0': 19, 'F0': 20, 'Gb0': 21, 'F#0': 21, 'G0': 22, 'Ab0': 23, 'G#0': 23, 'A0': 24, 'Bb0': 25, 'A#0': 25, 'B0': 26, 'C1': 27, 'Db1': 28, 'C#1': 28, 'D1': 29, 'Eb1': 30, 'D#1': 30, 'E1': 31, 'F1': 32, 'Gb1': 33, 'F#1': 33, 'G1': 34, 'Ab1': 35, 'G#1': 35, 'A1': 36, 'Bb1': 37, 'A#1': 37, 'B1': 38, 'C2': 39, 'Db2': 40, 'C#2': 40, 'D2': 41, 'Eb2': 42, 'D#2': 42, 'E2': 43, 'F2': 44, 'Gb2': 45, 'F#2': 45, 'G2': 46, 'Ab2': 47, 'G#2': 47, 'A2': 48, 'Bb2': 49, 'A#2': 49, 'B2': 50, 'C3': 51, 'Db3': 52, 'C#3': 52, 'D3': 53, 'Eb3': 54, 'D#3': 54, 'E3': 55, 'F3': 56, 'Gb3': 57, 'F#3': 57, 'G3': 58, 'Ab3': 59, 'G#3': 59, 'A3': 60, 'Bb3': 61, 'A#3': 61, 'B3': 62, 'C4': 63, 'Db4': 64, 'C#4': 64, 'D4': 65, 'Eb4': 66, 'D#4': 66, 'E4': 67, 'F4': 68, 'Gb4': 69, 'F#4': 69, 'G4': 70, 'Ab4': 71, 'G#4': 71, 'A4': 72, 'Bb4': 73, 'A#4': 73, 'B4': 74, 'C5': 75, 'Db5': 76, 'C#5': 76, 'D5': 77, 'Eb5': 78, 'D#5': 78, 'E5': 79, 'F5': 80, 'Gb5': 81, 'F#5': 81, 'G5': 82, 'Ab5': 83, 'G#5': 83, 'A5': 84, 'Bb5': 85, 'A#5': 85, 'B5': 86, 'C6': 87, 'Db6': 88, 'C#6': 88, 'D6': 89, 'Eb6': 90, 'D#6': 90, 'E6': 91, 'F6': 92, 'Gb6': 93, 'F#6': 93, 'G6': 94, 'Ab6': 95, 'G#6': 95, 'A6': 96, 'Bb6': 97, 'A#6': 97, 'B6': 98, 'C7': 99, 'Db7': 100, 'C#7': 100, 'D7': 101, 'Eb7': 102, 'D#7': 102, 'E7': 103, 'F7': 104, 'Gb7': 105, 'F#7': 105, 'G7': 106, 'Ab7': 107, 'G#7': 107, 'A7': 108, 'Bb7': 109, 'A#7': 109, 'B7': 110, 'C8': 111, 'Db8': 112, 'C#8': 112, 'D8': 113, 'Eb8': 114, 'D#8': 114, 'E8': 115, 'F8': 116, 'Gb8': 117, 'F#8': 117, 'G8': 118, 'Ab8': 119, 'G#8': 119, 'A8': 120, 'Bb8': 121, 'A#8': 121, 'B8': 122, 'C9': 123, 'Db9': 124, 'C#9': 124, 'D9': 125, 'Eb9': 126, 'D#9': 126, 'E9': 127, 'F9': 128, 'Gb9': 129, 'F#9': 129, 'G9': 130}  

		self.seq_len_note = seq_len_note
		self.seq_len_chord = seq_len_chord
		self.if_pad = if_pad
		
	def __call__(self, json_path):
		self.json_path = json_path
		self.music_dict= json.load(open(json_path,"r"))
		note_lst = self.music_dict["chord_note"][0] #[[p1, d1], [p2, d2], ]
		chord_lst = self.music_dict["chord_note"][1] #[[c1, d1], [c2, d2], ]

		#get rid of over-long sequences
		if self.seq_len_note<len(note_lst) or self.seq_len_chord <len(chord_lst):
			raise ValueError(f'Skipping... sequence too long! Length of note seq should be less than {self.seq_len_note} but is:{len(note_lst)}, Length of chord seq should be less than {self.seq_len_chord} but is:{len(chord_lst)}')
			#pass		
		#pad in the end
		if self.if_pad:
			pad_len_note = self.seq_len_note-len(note_lst)
			pad_len_chord = self.seq_len_chord - len(chord_lst)
			note_lst += [["pad", "pad"] for _ in range(pad_len_note)]
			chord_lst += [["pad", "pad"]  for _ in range(pad_len_chord)]

		self.note_lst, self.chord_lst=note_lst, chord_lst
		self.note_lst_enc = self.tokenize_note_lst()
		self.chord_lst_enc = self.tokenize_chord_lst()
		self.onset_rel_mask, self.onset_rel, self.dur_onset_cumsum = self.calculate_relative(self.note_lst_enc[:, 1], input_type = "duration")
		self.pitch_rel_mask, self.pitch_rel, _ = self.calculate_relative(self.note_lst_enc[:, 0], input_type = "pitch")
		return self.note_lst_enc, self.chord_lst_enc, self.pitch_rel, self.pitch_rel_mask,  self.onset_rel, self.onset_rel_mask, self.dur_onset_cumsum
	
	def calculate_relative(self, inp, input_type = "pitch"):
		# unkown_token = torch.tensor([999 for _ in range(len(inp))])[..., None]
		if input_type=="pitch":
			unkown_token = 0
			mask0 = inp==0
			mask1 = inp==1
			mask2 = inp==2
			mask_01= torch.logical_or(mask0, mask1)
			mask_012= torch.logical_or(mask_01, mask2)

			mask_matrix = torch.logical_or(mask_012[..., None], mask_012[None, ...])
			diff = inp[..., None] - inp[None, ...] #246, 246
			rel_matrix = torch.where( mask_matrix, unkown_token, diff)
			inp_cumsum = None

		elif input_type=="duration":

			unkown_token = torch.tensor(0.)
			inp_cumsum = torch.cumsum(torch.tensor([0.]+[self.duration_dict_inv[x] for x in inp.clone().numpy()][:-1]), dim = 0) 

			mask0 = inp==0 #an additional one here!
			total_dur = torch.sum(torch.logical_not(mask0).to(torch.int32)*torch.tensor([self.duration_dict_inv[x] for x in inp.clone().numpy()]), dim=-1)
			
			mask_matrix = torch.logical_or(mask0[..., None], mask0[None, ...])
			diff = inp_cumsum[..., None] - inp_cumsum[None, ...] #246, 246
			rel_matrix = torch.where( mask_matrix, unkown_token, diff)
			inp_cumsum = torch.where( mask0, total_dur, inp_cumsum)
		return mask_matrix, rel_matrix, inp_cumsum	
	
	def tokenize_chord_lst(self, external_chord_lst = None):
		
		if external_chord_lst:
			chord_lst = external_chord_lst
		else:
			chord_lst = self.chord_lst

		out_lst = [ [self.chord_dict2[x[0]], self.duration_dict[x[1]] ] for x in chord_lst]
		return torch.tensor(out_lst)

	def tokenize_note_lst(self, external_note_lst = None):

		if external_note_lst:
			note_lst = external_note_lst
		else:
			note_lst = self.note_lst

		out_lst = [ [self.pitch_dict2[x[0]], self.duration_dict[x[1]] ] for x in note_lst]

		return torch.tensor(out_lst)


class detokenizer_plain():
	def __init__(self):
		self.decode_dur = {0: 'pad', 1: 0.25, 2: 0.5, 3: 0.75, 4: 1.0, 5: 1.25, 6: 1.5, 7: 1.75, 8: 2.0, 9: 2.25, 10: 2.5, 11: 2.75, 12: 3.0, 13: 3.25, 14: 3.5, 15: 3.75, 16: 4.0}
		self.decode_pitch = {0: 'pad', 1: 'rest', 2: 'sustain', 3: 'C-1', 4: 'C#-1', 5: 'D-1', 6: 'D#-1', 7: 'E-1', 8: 'F-1', 9: 'F#-1', 10: 'G-1', 11: 'G#-1', 12: 'A-1', 13: 'A#-1', 14: 'B-1', 15: 'C0', 16: 'C#0', 17: 'D0', 18: 'D#0', 19: 'E0', 20: 'F0', 21: 'F#0', 22: 'G0', 23: 'G#0', 24: 'A0', 25: 'A#0', 26: 'B0', 27: 'C1', 28: 'C#1', 29: 'D1', 30: 'D#1', 31: 'E1', 32: 'F1', 33: 'F#1', 34: 'G1', 35: 'G#1', 36: 'A1', 37: 'A#1', 38: 'B1', 39: 'C2', 40: 'C#2', 41: 'D2', 42: 'D#2', 43: 'E2', 44: 'F2', 45: 'F#2', 46: 'G2', 47: 'G#2', 48: 'A2', 49: 'A#2', 50: 'B2', 51: 'C3', 52: 'C#3', 53: 'D3', 54: 'D#3', 55: 'E3', 56: 'F3', 57: 'F#3', 58: 'G3', 59: 'G#3', 60: 'A3', 61: 'A#3', 62: 'B3', 63: 'C4', 64: 'C#4', 65: 'D4', 66: 'D#4', 67: 'E4', 68: 'F4', 69: 'F#4', 70: 'G4', 71: 'G#4', 72: 'A4', 73: 'A#4', 74: 'B4', 75: 'C5', 76: 'C#5', 77: 'D5', 78: 'D#5', 79: 'E5', 80: 'F5', 81: 'F#5', 82: 'G5', 83: 'G#5', 84: 'A5', 85: 'A#5', 86: 'B5', 87: 'C6', 88: 'C#6', 89: 'D6', 90: 'D#6', 91: 'E6', 92: 'F6', 93: 'F#6', 94: 'G6', 95: 'G#6', 96: 'A6', 97: 'A#6', 98: 'B6', 99: 'C7', 100: 'C#7', 101: 'D7', 102: 'D#7', 103: 'E7', 104: 'F7', 105: 'F#7', 106: 'G7', 107: 'G#7', 108: 'A7', 109: 'A#7', 110: 'B7', 111: 'C8', 112: 'C#8', 113: 'D8', 114: 'D#8', 115: 'E8', 116: 'F8', 117: 'F#8', 118: 'G8', 119: 'G#8', 120: 'A8', 121: 'A#8', 122: 'B8', 123: 'C9', 124: 'C#9', 125: 'D9', 126: 'D#9', 127: 'E9', 128: 'F9', 129: 'F#9', 130: 'G9'}
		self.decode_chord = {0: 'pad', 1: 'rest', 2: 'sustain', 3: 'Cmaj', 4: 'Cmin', 5: 'Cdim', 6: 'C7', 7: 'C#maj', 8: 'C#min', 9: 'C#dim', 10: 'C#7', 11: 'Dmaj', 12: 'Dmin', 13: 'Ddim', 14: 'D7', 15: 'D#maj', 16: 'D#min', 17: 'D#dim', 18: 'D#7', 19: 'Emaj', 20: 'Emin', 21: 'Edim', 22: 'E7', 23: 'Fmaj', 24: 'Fmin', 25: 'Fdim', 26: 'F7', 27: 'F#maj', 28: 'F#min', 29: 'F#dim', 30: 'F#7', 31: 'Gmaj', 32: 'Gmin', 33: 'Gdim', 34: 'G7', 35: 'G#maj', 36: 'G#min', 37: 'G#dim', 38: 'G#7', 39: 'Amaj', 40: 'Amin', 41: 'Adim', 42: 'A7', 43: 'A#maj', 44: 'A#min', 45: 'A#dim', 46: 'A#7', 47: 'Bmaj', 48: 'Bmin', 49: 'Bdim', 50: 'B7'}
		self.NOTE_TO_OFFSET = {
			'Ab': 8,
			'A': 9,
			'A#': 10,
			'Bb': 10,
			'B': 11,
			'Cb': 11,
			'C': 0,
			'C#': 1,
			'Db': 1,
			'D': 2,
			'D#': 3,
			'Eb': 3,
			'E': 4,
			'E#': 5,  # knife-party/power-glove/intro
			'F': 5,
			'F#': 6,
			'Gb': 6,
			'G': 7,
			'G#': 8,
		} 

	def drop_pad_new(self, lst = None, trimmed_len = None):
		"""
		input(batch, len) #batch = 1
		"""
		if trimmed_len is None:
			lst_tmp = np.trim_zeros(lst[0, :], trim='b')
			trimmed_len = len(lst_tmp)
		
		return lst[:, :trimmed_len]


	def drop_pad(self, lst, trimmed_len = None):
		"""
		input(batch, len, dim)
		"""
		if trimmed_len is None: 
			lst_tmp = np.trim_zeros(lst[:,0], trim='b')
			trimmed_len = len(lst_tmp)
		
		drop_pad_lst = lst[:trimmed_len, :]
		return drop_pad_lst
	
	def decode_chord_lst(self, external_lst):
		chord_lst = self.drop_pad(external_lst)
		# print("fuck", len(chord_lst))

		if self.mode == "parallel":
			out_chord = [[self.decode_chord[x[0]], self.decode_dur[x[1]]] for x in chord_lst]
		return out_chord
	def decode_note_lst(self, external_lst):
		note_lst = self.drop_pad(external_lst)
		if self.mode == "parallel":
			out_note = [[self.decode_pitch[x[0]], self.decode_dur[x[1]]] for x in note_lst]
		return out_note

