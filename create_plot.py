import matplotlib.pyplot as plt
import numpy as np

def create_plot(embeddings):
    # Convert embeddings to numpy for visualization
    embeddings_np = embeddings.detach().numpy()
    # Plotting embeddings
    fig, ax = plt.subplots(figsize=(15, 5))  # Increase figure size for bigger squares
    cax = ax.matshow(embeddings_np, aspect='auto', cmap='viridis')

    # Add color bar for reference
    fig.colorbar(cax)

    # Set labels
    ax.set_xticks(np.arange(embeddings_dimension))
    ax.set_yticks(np.arange(len(tokenized_input)))
    ax.set_xticklabels([f'Dim {i}' for i in range(embeddings_dimension)])
    ax.set_yticklabels([f'Token number: {i}' for i in tokenized_input])

    # Rotate the tick labels and set their alignment
    plt.xticks(rotation=90)
    plt.xlabel('Embedding Dimensions')
    plt.ylabel('Tokens')

    # Adding numerical values to the plot
    for i in range(len(tokenized_input)):
        for j in range(embeddings_dimension):
            text = ax.text(j, i, f'{embeddings_np[i, j]:.2f}', ha='center', va='center', color='white')

    plt.title('Token Embeddings Visualization')
    plt.show()