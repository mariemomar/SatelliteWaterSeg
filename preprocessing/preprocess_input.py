import numpy as np


def normalize(band):
    """Normalize a single band to [0, 1]"""
    return (band - band.min()) / (band.max() - band.min() + 1e-6)  

def preprocess_input(image):
    """
    Preprocess image exactly like during training
    Per-band min-max normalization to [0, 1]
    """
    # Convert to float32 like in training
    image = image.astype(np.float32)
    
    # Apply per-band normalization (same as training)
    for c in range(image.shape[-1]):
        band = image[..., c]
        image[..., c] = (band - band.min()) / (band.max() - band.min() + 1e-6)
    
    return image