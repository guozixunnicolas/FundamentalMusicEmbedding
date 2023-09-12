import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from tutorial.fme.utils import load_we_model, load_fme_model, relative_pitch_matrix, self_distance_matrix
from tutorial.fme.visual import plot_distance_matrix, plot_we_pitch_embedding


the_lick_Cm = torch.tensor([60, 62, 63, 65, 62, 58, 60])
the_lick_Gm = torch.tensor([67, 69, 70, 72, 69, 65, 67])

the_lick_Cm_relative = relative_pitch_matrix(the_lick_Cm)
the_lick_Gm_relative = relative_pitch_matrix(the_lick_Gm)

_, ax = plt.subplots(1, 2)

plot_distance_matrix(the_lick_Cm_relative.numpy(), the_lick_Cm.numpy(), ax[0])
plot_distance_matrix(the_lick_Gm_relative.numpy(), the_lick_Gm.numpy(), ax[1])
print("The relative pitch matrices for \"the lick\" in different keys are identical. \n--> the reason why humans are able to identify music snippets regardless of absolute keys.")



# Embed "the lick" using One Hot Encoding
the_lick_Cm_one_hot = F.one_hot(the_lick_Cm, num_classes = 256).to(torch.float32)
the_lick_Gm_one_hot = F.one_hot(the_lick_Gm, num_classes = 256).to(torch.float32)

# Plot self distance matrices with different underlying embedding methods (WE and OH)

sdm_Cm_one_hot = self_distance_matrix(the_lick_Cm_one_hot)
sdm_Gm_one_hot = self_distance_matrix(the_lick_Gm_one_hot)

_, ax = plt.subplots(1, 2)
plot_distance_matrix(sdm_Cm_one_hot.numpy(), the_lick_Cm.numpy(), axis=ax[0])
plot_distance_matrix(sdm_Gm_one_hot.numpy(), the_lick_Gm.numpy(), axis=ax[1])

# Load pre-trained word embedding
we_model = load_we_model()
plot_we_pitch_embedding(we_model)

we_model.pitch_embedding(the_lick_Cm)

sdm_Cm_word_embedding = self_distance_matrix(we_model.pitch_embedding(the_lick_Cm))
sdm_Gm_word_embedding = self_distance_matrix(we_model.pitch_embedding(the_lick_Gm))

_, ax = plt.subplots(1, 2)
plot_distance_matrix(sdm_Cm_word_embedding.numpy(), the_lick_Cm.numpy(), axis=ax[0])
plot_distance_matrix(sdm_Gm_word_embedding.numpy(), the_lick_Gm.numpy(), axis=ax[1])

# Load pre-trained FME
fme_model = load_fme_model()

# FME equation
import numpy as np

def w_k(B, k, d):
  return np.power(B, -2*k/d)

def p_k(f, w, bsin, bcos):
  return [np.sin(w*f) + bsin, np.cos(w*f) + bcos]

B = fme_model.pitch_embedding_conf['base']
d = fme_model.pitch_embedding_conf['d_model']
bsin = np.random.standard_normal()
bcos = np.random.standard_normal()

emb = []
for f in range(128):
  fme = []
  for k in range(d//2):
    w = w_k(B, k, d)
    p = p_k(f, w, bsin, bcos)
    fme.extend(p)
  emb.append(fme)

e = torch.nn.Embedding(num_embeddings=128, embedding_dim=d)
e.weights = torch.Tensor(emb)

n60 = e.weights.numpy()[60]
n63 = e.weights.numpy()[63]
n69 = e.weights.numpy()[69]
n72 = e.weights.numpy()[72]

print(f'L2 distance of embeddings for note 72 and 63: {np.linalg.norm(n72 - n63):.4f}')
print(f'L2 distance of embeddings for note 69 and 60: {np.linalg.norm(n69 - n60):.4f}')
