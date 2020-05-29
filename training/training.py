from training_functions import *


DATA = 0
LABELS = 1

TRAIN = 0
VAL = 1
TEST = 2


dataset = load_dataset(1, fma_set=True, normalized=False)
train_network(dataset[TRAIN][DATA], dataset[TRAIN][LABELS], fma_set=True, save_path='unnorm_split1.network')
test_network(dataset[VAL][DATA], dataset[VAL][LABELS], network_path='unnorm_split1.network')
