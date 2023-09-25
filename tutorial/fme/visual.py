import matplotlib.pyplot as plt


def plot_distance_matrix(distance_matrix, notes, axis: plt.Axes):
    axis.matshow(distance_matrix,  cmap=plt.cm.Blues)

    axis.set_xticks(range(0, len(notes), 1))
    axis.set_yticks(range(0, len(notes), 1))
    axis.set_xticklabels(notes[::1], fontsize=12, rotation=45)
    axis.set_yticklabels(notes[::1], fontsize=12, rotation=45)

    for i in range(distance_matrix.shape[0]):
        for j in range(distance_matrix.shape[1]):
            axis.text(i, j, str(-distance_matrix[j,i])[:4], va='center', ha='center')

def plot_we_pitch_embedding(we_model):
    word_embedding = we_model.pitch_embedding.weight
    word_embedding = word_embedding.cpu().detach().numpy()
    plt.matshow(word_embedding)
    plt.title('256-dimensional Embedding of Musical Pitch')
