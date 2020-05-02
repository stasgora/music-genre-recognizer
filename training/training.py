from training.training_functions import *


DATA = 0
LABELS = 1

TRAIN = 0
VAL = 1
TEST = 2


dataset = load_dataset(0, normalized=True)
train_network(dataset[TRAIN][DATA], dataset[TRAIN][LABELS], save_path='norm_split1.network')
test_network(dataset[VAL][DATA], dataset[VAL][LABELS])
