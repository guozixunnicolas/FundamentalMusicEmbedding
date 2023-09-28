import torch
import pickle
from tutorial.fme.consts import (
    WE_MODEL_PICKLE, FME_MODEL_PICKLE
)

def relative_pitch_matrix(notes: torch.Tensor):
    return notes[None, ...] - notes[..., None]

def self_distance_matrix(embedding: torch.Tensor):
  e1 = torch.unsqueeze(embedding, dim = 1)  # Dim: (len, 1, dim)
  e2 = torch.unsqueeze(embedding, dim = 0)  # Dim: (1, len, dim)
  dist = e1 - e2 # Dim: (len, len, dim)
  dist_l2 = torch.sqrt(torch.sum(torch.pow(dist, 2), dim = -1)) # Dim: (len, len)
  return dist_l2

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        raise Warning('No CUDA devices found. Using CPU')

    return device

def _load_model(pickle_path, device='cuda:0'):
    with open(pickle_path, 'rb') as o:
        model = pickle.load(o)

    # Switch to eval mode
    model.eval()
    return model

def load_we_model():
    return _load_model(WE_MODEL_PICKLE)

def load_fme_model():
    return _load_model(FME_MODEL_PICKLE)
