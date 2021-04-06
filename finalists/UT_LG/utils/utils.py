import torch
from torchvision import transforms


def resize_tensor(input_tensors, new_size):
    final_output = None
    for img in input_tensors:
        img_PIL = transforms.ToPILImage()(img)
        img_PIL = transforms.Resize([new_size, new_size])(img_PIL)
        img_PIL = transforms.ToTensor()(img_PIL)
        img_PIL = img_PIL.unsqueeze(0)
        if final_output is None:
            final_output = img_PIL
        else:
            final_output = torch.cat((final_output, img_PIL), 0)
    return final_output
