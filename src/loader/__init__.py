import json

from loader.gtsrb import gtsrbLoader
from loader.gtsrb2TT100K import gtsrb2TT100KLoader
from loader.belga2flickr import belga2flickrLoader
from loader.belga2toplogo import belga2toplogoLoader
from loader.miniimagenet_loader2 import MiniImageNet
from loader.omniglot import OmniglotDataset
from loader.plantae import plantae_Loader

def get_loader(name):
    return {
        'gtsrb': gtsrbLoader,
        'gtsrb2TT100K': gtsrb2TT100KLoader,
        'belga2flickr': belga2flickrLoader,
        'belga2toplogo': belga2toplogoLoader,
        'miniimagenet' : MiniImageNet,
        'omniglot': OmniglotDataset,
        'plantae': plantae_Loader
    }[name]


def get_data_path(name, config_file='/home/snag005/Desktop/fs_ood/trial2/datasets/config.json'):
    data = json.load(open(config_file))
    return data[name]['data_path']
