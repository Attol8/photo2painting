import torch
from commons import get_model

def get_painting_tensor(photo):
    """Forward function used in test time.

    This function wraps <forward> function in no_grad() so we don't save intermediate steps for backprop
    It also calls <compute_visuals> to produce additional visualization results
    """
    model = get_model()

    with torch.no_grad():
        return model.forward(photo)
   