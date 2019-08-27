import functools
import io

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

from cgan.models import base_model, networks

def get_model(input_nc = 3, output_nc = 3, load_path = 'models\paintings_cyclegan\latest_net_G.pth', norm_layer = functools.partial(torch.nn.InstanceNorm2d, affine=False, track_running_stats=False),  ):
	"""
	make it dynamic (accept all networks) 
	"""
	model = networks.ResnetGenerator(input_nc, output_nc, norm_layer=norm_layer, n_blocks=9)
	state_dict = torch.load(load_path, map_location='cpu')
	model.load_state_dict(state_dict, strict=True)
	model.eval()
	return model


def get_photo(image_bytes):
	"""
	This function takes an uploaded file, apply various transforms on it and returns it as a PIL Image 
	"""

	my_transforms = transforms.Compose([transforms.Resize(256),
					transforms.CenterCrop(224),
					transforms.ToTensor(),
					transforms.Normalize(mean=[0.485, 0.456, 0.406], 
										std=[0.229, 0.224, 0.225])])
	if type(image_bytes) == str:
		image_bytes = str.encode(image_bytes)
	else:
		image = Image.open(io.BytesIO(image_bytes))
	return my_transforms(image).unsqueeze(0)

def tensor_to_PIL(tensor):
        """
		Transform function.

        This function takes a pytorch tensor and returns a PIL Image
        """
        np_im = tensor2im(tensor, imtype=np.uint8) #numpy image
        PIL_img = Image.fromarray(np_im, 'RGB')
        return PIL_img

def tensor2im(input_image, imtype=np.uint8):
    """"Converts a Tensor array into a numpy image array.

    Parameters:
        input_image (tensor) --  the input image tensor array
        imtype (type)        --  the desired type of the converted numpy array
    """
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):  # get the data from a variable
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()  # convert it into a numpy array
        if image_numpy.shape[0] == 1:  # grayscale to RGB
            image_numpy = np.tile(image_numpy, (3, 1, 1))
        image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0  # post-processing: tranpose and scaling
    else:  # if it is a numpy array, do nothing
        image_numpy = input_image
    return image_numpy.astype(imtype)
