from training_functions import *


DATA = 0
LABELS = 1

TRAIN = 0
VAL = 1
TEST = 2

fma_set=False
dataset = load_dataset(0, fma_set=fma_set, normalized=False)
train_network(dataset[TRAIN][DATA], dataset[TRAIN][LABELS], dataset[VAL][DATA], dataset[VAL][LABELS], fma_set=fma_set, save_path='unnorm_split1.network')
test_network(dataset[VAL][DATA], dataset[VAL][LABELS], network_path='unnorm_split1.network')
