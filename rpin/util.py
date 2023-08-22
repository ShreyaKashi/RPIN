import torch

def to_device(input_data, device='cuda', non_blocking=False):
    '''
    Move input_data to device. If the data is a list, move everything in the list to device
    '''
    if isinstance(input_data, list) and len(input_data) > 0:
        if isinstance(input_data[0], list):
            for idx in range(len(input_data)):
                for idx2 in range(len(input_data[idx])):
                    input_data[idx][idx2] = input_data[idx][idx2].to(device, non_blocking=non_blocking)
        else:
            for idx in range(len(input_data)):
                input_data[idx] = input_data[idx].to(device, non_blocking=non_blocking)

    if isinstance(input_data, torch.Tensor):
        input_data = input_data.to(device, non_blocking=non_blocking)

    return input_data