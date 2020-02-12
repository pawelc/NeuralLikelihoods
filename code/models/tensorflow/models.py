from models.tensorflow.maf import MAF
from models.tensorflow.mdn import MDN
from models.tensorflow.monde import MONDE
from models.tensorflow.monde_ar_made import MondeARMADE
from models.tensorflow.pumonde_pfor import PumondePFor
from models.tensorflow.rnade import Rnade
from models.tensorflow.rnade_deep import RnadeDeep


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
    elif model_name.startswith("RNADE_deep"):
        arch = [kwargs['sh']]* kwargs['nh']
        params = {'k_mix': kwargs['km'], 'arch': arch, 'component_distribution': model_name[11:]}
        return RnadeDeep(**params)
    elif model_name.startswith("RNADE"):
        params = {'k_mix': kwargs['km'], 'hidden_units': kwargs['sh'], 'component_distribution': model_name[6:]}
        return Rnade(**params)
    elif model_name.startswith("MONDE_copula_const_cov"):
        params = {'cov_type': 'const_cov', 'arch_hxy': [kwargs['hxy_sh']]* kwargs['hxy_nh'],
                  'arch_x_transform': [kwargs['x_sh']]* kwargs['x_nh'], 'hxy_x_size': kwargs['hxy_x']}
        return MONDE(**params)
    elif model_name.startswith("MONDE_AR_MADE"):
        params = {'transform': kwargs['tr'], 'arch': [kwargs['sh']]* kwargs['nh'],
                  'x_transform_size': kwargs['xs']}
        return MondeARMADE(**params)
    else:
        raise ValueError("model not recognized: " + model_name)