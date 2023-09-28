import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d import proj3d
from matplotlib.patches import FancyArrowPatch
import json
from midiutil import MIDIFile
import pretty_midi

class Sampler_tf():
	def __init__(self, decoder_choice, temperature=1.0, top_k=None, top_p=None):
		self.decoder_choice = decoder_choice
		self.temperature = temperature
		self.top_k = top_k
		if decoder_choice =="greedy":
			self.top_k = 1
		self.top_p = top_p

	def get_shape_list(self, tensor):
		"""Returns a list of the shape of tensor, preferring static dimensions.
		Args:
			tensor: A tf.Tensor object to find the shape of.
			expected_rank: (optional) int. The expected rank of `tensor`. If this is
			specified and the `tensor` has a different rank, and exception will be
			thrown.
			name: Optional name of the tensor for the error message.
		Returns:
			A list of dimensions of the shape of tensor. All static dimensions will
			be returned as python integers, and dynamic dimensions will be returned
			as tf.Tensor scalars.
		"""

		shape = tensor.shape.as_list()

		non_static_indexes = []
		for (index, dim) in enumerate(shape):
			if dim is None:
				non_static_indexes.append(index)

		if not non_static_indexes:
			return shape

		dyn_shape = tf.shape(tensor)
		for index in non_static_indexes:
			shape[index] = dyn_shape[index]
		return shape

	# def greedy(self, log_probs):
	# 	log_probs, ids = tf.math.top_k(log_probs, k=1)
	# 	return log_probs, ids
	# def greedy(self, logits):
	# 	return logits
	def sample_logits_with_temperature(self, logits):
		return logits / self.temperature

	def sample_top_k(self,logits):
		top_k_logits = tf.math.top_k(logits, k=self.top_k)
		indices_to_remove = logits < tf.expand_dims(top_k_logits[0][..., -1], -1)
		top_k_logits = self.set_tensor_by_indices_to_value(logits, indices_to_remove,
														np.NINF)
		return top_k_logits

	def sample_top_p(self, logits):
		sorted_indices = tf.argsort(logits, direction="DESCENDING")
		# Flatten logits as tf.gather on TPU needs axis to be compile time constant.
		logits_shape = self.get_shape_list(logits)
		range_for_gather = tf.expand_dims(tf.range(0, logits_shape[0]), axis=1)
		range_for_gather = tf.tile(range_for_gather * logits_shape[1],
									[1, logits_shape[1]]) + sorted_indices
		flattened_logits = tf.reshape(logits, [-1])
		flattened_sorted_indices = tf.reshape(range_for_gather, [-1])
		sorted_logits = tf.reshape(
			tf.gather(flattened_logits, flattened_sorted_indices),
			[logits_shape[0], logits_shape[1]])
		cumulative_probs = tf.cumsum(tf.nn.softmax(sorted_logits, axis=-1), axis=-1)

		# Remove tokens with cumulative probability above the threshold.
		sorted_indices_to_remove = cumulative_probs > self.top_p

		# Shift the indices to the right to keep the first token above threshold.
		sorted_indices_to_remove = tf.roll(sorted_indices_to_remove, 1, axis=-1)
		sorted_indices_to_remove = tf.concat([
			tf.zeros_like(sorted_indices_to_remove[:, :1]),
			sorted_indices_to_remove[:, 1:]
		], -1)

		# Scatter sorted indices to original indexes.
		indices_to_remove = self.scatter_values_on_batch_indices(sorted_indices_to_remove,
															sorted_indices)
		top_p_logits = self.set_tensor_by_indices_to_value(logits, indices_to_remove,
														np.NINF)
		return top_p_logits

	def scatter_values_on_batch_indices(self, values, batch_indices):
		"""Scatter `values` into a tensor using `batch_indices`.
		Args:
			values: tensor of shape [batch_size, vocab_size] containing the values to
			scatter
			batch_indices: tensor of shape [batch_size, vocab_size] containing the
			indices to insert (should be a permutation in range(0, n))
		Returns:
			Tensor of shape [batch_size, vocab_size] with values inserted at
			batch_indices
		"""
		tensor_shape = self.get_shape_list(batch_indices)
		broad_casted_batch_dims = tf.reshape(
			tf.broadcast_to(
				tf.expand_dims(tf.range(tensor_shape[0]), axis=-1), tensor_shape),
			[1, -1])
		pair_indices = tf.transpose(
			tf.concat([broad_casted_batch_dims,
						tf.reshape(batch_indices, [1, -1])], 0))
		return tf.scatter_nd(pair_indices, tf.reshape(values, [-1]), tensor_shape)

	def set_tensor_by_indices_to_value(self, input_tensor, indices, value):
		"""Where indices is True, set the value in input_tensor to value.
		Args:
			input_tensor: float (batch_size, dim)
			indices: bool (batch_size, dim)
			value: float scalar
		Returns:
			output_tensor: same shape as input_tensor.
		"""
		value_tensor = tf.zeros_like(input_tensor) + value
		output_tensor = tf.where(indices, value_tensor, input_tensor)
		return output_tensor

	def __call__(self, logits):
		# print("input logits", logits)
		logits = self.sample_logits_with_temperature(logits)
		# print("input logits tempe", logits)
		if self.decoder_choice =="greedy":
			filtered_logits = self.sample_top_k(logits)
		elif self.decoder_choice =="top_k":
			filtered_logits = self.sample_top_k(logits)
		elif self.decoder_choice =="top_p":
			filtered_logits = self.sample_top_p(logits)
		# print("after filtering", filtered_logits)
		# filtered_logits = tf.nn.softmax(filtered_logits, axis = -1)
		sampled_logits = tf.random.categorical(filtered_logits, dtype=tf.int64, num_samples=1)
		# sampled_logits = tf.cast(sampled_logits, tf.int64)
		return sampled_logits

class Sampler_torch():
	def __init__(self, decoder_choice, temperature=1.0, top_k=None, top_p=None):
		self.decoder_choice = decoder_choice
		self.temperature = temperature
		self.top_k = top_k
		if decoder_choice =="greedy":
			self.top_k = 1
		self.top_p = top_p

	def sample_logits_with_temperature(self, logits):
		return logits / self.temperature

	def sample_top_k(self,logits):
		indices_to_remove = logits < torch.topk(logits, self.top_k)[0][..., -1, None]
		logits[indices_to_remove] = -float('Inf')
		return logits

	def sample_top_p(self, logits):
		sorted_logits, sorted_indices = torch.sort(logits, descending=True)
		cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)

		# Remove tokens with cumulative probability above the threshold
		sorted_indices_to_remove = cumulative_probs > self.top_p
		# Shift the indices to the right to keep also the first token above the threshold
		sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
		sorted_indices_to_remove[..., 0] = 0

		indices_to_remove = sorted_indices[sorted_indices_to_remove]
		logits[:, indices_to_remove] = -float('Inf')

		return logits

	def __call__(self, logits):
		# print("input logits", logits)
		logits = self.sample_logits_with_temperature(logits)
		# print("input logits tempe", logits)
		if self.decoder_choice =="greedy":
			filtered_logits = self.sample_top_k(logits)
		elif self.decoder_choice =="top_k":
			filtered_logits = self.sample_top_k(logits)
		elif self.decoder_choice =="top_p":
			filtered_logits = self.sample_top_p(logits)
		probabilities = F.softmax(filtered_logits, dim=-1)

		sampled_logits = torch.multinomial(probabilities, 1)
		return sampled_logits

def write_midi_monophonic(note_list, chord_list, mid_name, if_cosiatec = False):
    MyMIDI2 = MIDIFile(numTracks=1,ticks_per_quarternote=220) #track0 is melody, track1 is chord

    cum_time = float(0)
    for pos, (note, dur) in enumerate(note_list): 
        if note=="sustain": 
            continue
        else:
            #check future pos whether sustain
            next_pos = pos + 1 
            if next_pos<=len(note_list)-1:
                note_next, dur_next = note_list[next_pos]
                while(note_next == "sustain" and next_pos<=len(note_list)-1):
                    dur += dur_next
                    next_pos+=1
                    try:
                        note_next, dur_next = note_list[next_pos]
                    except:
                        note_next = "break the loop"
                        dur_next = "break the loop"
            
            if note=="rest":
                cum_time = cum_time+dur
            else:
                MyMIDI2.addNote(track = 0, channel = 0, pitch = pretty_midi.note_name_to_number(note), time = cum_time, duration = dur, volume = 100)
                cum_time = cum_time+dur

    with open(mid_name, "wb") as output_file2:
        MyMIDI2.writeFile(output_file2)  
