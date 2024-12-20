import torch
import util.kernel as kernel


def load_config(cfg_path):
    """ Parses a configuration .yml file into a dictionary.
    """
    import yaml
    with open(cfg_path, 'r') as file:
        config = yaml.safe_load(file)

    # Map device string to torch.device object
    if 'device' in config:
        config['device'] = torch.device(config['device'])
    else:  # Default to cpu
        config['device'] = torch.device('cpu')

    # Map dtype string to torch.dtype object
    if 'dtype' in config:
        if config['dtype'] == 'float':
            config['dtype'] = torch.float
        elif config['dtype'] == 'double':
            config['dtype'] = torch.double
        else:
            raise NotImplementedError
    else:  # Default to float
        config['dtype'] = torch.float

    support = torch.linspace(config['x_min'], config['x_max'], config['n_x']).reshape(-1, 1)
    config['support'] = support.to(config['dtype'])

    return config