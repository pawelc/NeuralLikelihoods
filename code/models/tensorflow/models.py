from models.tensorflow.maf import MAF
from models.tensorflow.mdn import MDN
from models.tensorflow.pumonde_pfor import PumondePFor
from models.tensorflow.rnade import Rnade


def create_model(model_name:str, kwargs):
    if model_name == "MAF":
        params = {'num_bijectors': kwargs['nb'], 'hidden_units': [kwargs['sh']]* kwargs['nh'],
         'covariate_hidden_units': [kwargs['shc']]* kwargs['nh'], 'batch_norm': kwargs['bn']}
        return MAF(**params)
    elif model_name == "MDN":
        params = {'num_mixtures': kwargs['nm'], 'arch': [kwargs['sh']]* kwargs['nh']}
        return MDN(**params)
    elif model_name == "PumondePFor":
        arch_x_transform = [kwargs['xs']]* kwargs['xn']
        arch_hxy = [kwargs['hxys']]* kwargs['hxyn']
        hxy_x_size = kwargs['hxyxs']
        arch_xy_comb = [kwargs['xycs']]* kwargs['xycn']
        params = {'arch_x_transform': arch_x_transform, 'arch_hxy': arch_hxy, 'hxy_x_size':hxy_x_size,
                  'arch_xy_comb':arch_xy_comb}
        return PumondePFor(**params)
    elif model_name.startswith("RNADE"):
        params = {'k_mix': kwargs['km'], 'hidden_units': kwargs['sh'], 'component_distribution': model_name[6:]}
        return Rnade(**params)
    else:
        raise ValueError("model not recognized: " + model_name)