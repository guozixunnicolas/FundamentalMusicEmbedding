import torch
import yaml
from tutorial.fme.consts import (
    WE_CONFIG_FILE, WE_MODEL_FILE, FME_CONFIG_FILE, FME_MODEL_FILE
)
from model.model import RIPO_transformer


def relative_pitch_matrix(notes: torch.Tensor):
    return notes[None, ...] - notes[..., None]

def self_distance_matrix(embedding: torch.Tensor):
  e1 = torch.unsqueeze(embedding, dim = 1)  # Dim: (len, 1, dim)
  e2 = torch.unsqueeze(embedding, dim = 0)  # Dim: (1, len, dim)
  dist = e1 - e2 # Dim: (len, len, dim)
  dist_l2 = torch.sqrt(torch.sum(torch.pow(dist, 2), dim = -1)) # Dim: (len, len)
  return dist_l2

def load_config(file, device='cuda:0'):
    with open(file, 'r') as f:
        model_cfg = yaml.safe_load(f)
        model_cfg['device'] = device
        model_cfg['relative_pitch_attention']['device'] = device
        model_cfg['relative_pitch_attention']['pitch_embedding_conf']['device'] = device
        model_cfg['relative_pitch_attention']['dur_embedding_conf']['device'] = device
        model_cfg['relative_pitch_attention']['position_encoding_conf']['device'] = device
    
    return model_cfg

def get_device():
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
        raise Warning('No CUDA devices found. Using CPU')

    return device

def load_we_model(device='cuda:0'):
    device = get_device()
    model_cfg = load_config(WE_CONFIG_FILE, device)

    model = RIPO_transformer(**model_cfg['relative_pitch_attention']).to(device)
    model.load_state_dict(torch.load(WE_MODEL_FILE, map_location=device))

    # Switch to eval mode
    model.eval()
    return model

def load_fme_model(device='cuda:0'):
    device = get_device()
    model_cfg = load_config(FME_CONFIG_FILE, device)

    model = RIPO_transformer(**model_cfg['relative_pitch_attention']).to(device)
    model.load_state_dict(torch.load(FME_MODEL_FILE, map_location=device))

    # Switch to eval mode
    model.eval()
    return model
