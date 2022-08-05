# paste before all other code in def cross:    
    if device is None and tensors is not None: 
        if type(tensors) == list:
            device = tensors[0].cores[0].device
        else:
            device = tensors.cores[0].device