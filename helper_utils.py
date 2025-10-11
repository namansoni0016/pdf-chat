def word_wrap(text, width=87):
    """
    Wraps the given text to the specified width.
    
    Args:
        text (str): The text to be wrapped.
        width (int): The width to wrap the text to
        
    Returns:
        str: The wrapped text with lines not exceeding the specified width.
    """
    return "\n".join([text[i : i + width] for i in range(0, len(text), width)])

def project_embeddings(embeddings, umap_transform):
    """
    Projects the given embeddings using the provided UMAP transformer
    Args:
        embeddings (numpy.ndarray): The embeddings to project
        umap_transform (umap.UMAP): The trained UMAP transformer
    Returns:
        numpy.ndarray: The projected embeddings
    """
    projected_embeddings = umap_transform.transform(embeddings)
    return projected_embeddings

