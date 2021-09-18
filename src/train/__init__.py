from train.trainer import train as trainer
from train.trainer_ae import train as trainer_ae
from train.trainer_bifurcate import train as trainer_bifurcate
from train.trainer_no_recon import train as trainer_no_recon
from train.new_trainer_proto import train as trainer_proto
from train.trainer_no_clf import train as trainer_no_clf
from train.trainer_no_embedding import train as trainer_no_embedding
from train.trainer_bifurcate_no_embedding import train as trainer_bifurcate_no_embedding
from train.trainer_bifurcate_no_clf import train as trainer_bifurcate_no_clf
from train.trainer_bifurcate_ae import train as trainer_bifurcate_ae

from train.tester import tester 
from train.tester_ae import tester as tester_ae
from train.tester_no_recon import tester as tester_no_recon
from train.new_tester_proto import tester as tester_proto
from train.tester_bifurcated import tester as tester_bifurcated
from train.tester_no_clf import tester as tester_no_clf
from train.tester_no_embedding import tester as tester_no_embedding
from train.tester_bifurcated_no_embedding import tester as tester_bifurcated_no_embedding
from train.tester_bifurcated_no_clf import tester as tester_bifurcated_no_clf
from train.tester_bifurcated_ae import tester as tester_bifurcated_ae
from train.tester_openmax import tester as tester_openmax
def get_trainer(name):
    return {
        'normal': trainer,
    	'ae':trainer_ae,
    	'no_recon':trainer_no_recon,
        'bifurcated': trainer_bifurcate,
        'proto': trainer_proto,
        'no_clf':trainer_no_clf,
        'no_embedding':trainer_no_embedding,
        'bifurcated_no_clf':trainer_bifurcate_no_clf,
        'bifurcated_no_embedding':trainer_bifurcate_no_embedding,
        'bifurcated_ae':trainer_bifurcate_ae
    }[name]

def get_tester(name):
    return {
        'normal': tester,
    	'ae':tester_ae,
    	'no_recon':tester_no_recon,
        'bifurcated': tester_bifurcated,
        'proto':tester_proto,        
        'no_clf':tester_no_clf,
        'no_embedding':tester_no_embedding,
        'bifurcated_no_clf':tester_bifurcated_no_clf,
        'bifurcated_no_embedding':tester_bifurcated_no_embedding,
        'bifurcated_ae':tester_bifurcated_ae,
        'openmax':tester_openmax
    }[name]

