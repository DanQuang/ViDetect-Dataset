from model.model_baseline import Baseline
from model.model_trans import Model_Trans
def build_model(config):
    if config['model']['type_model']=='baseline':
        return Baseline(config)
    if config['model']['type_model']=='trans':
        return Model_Trans(config)