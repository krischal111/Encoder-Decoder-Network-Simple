# will download the dataset later (for audio)
# This time just using the MNIST dataset
from torchvision.datasets import MNIST as M
from torchvision.transforms import ToTensor
from torch.utils.data import random_split, DataLoader, RandomSampler
from dotmap import DotMap
import torch

# Train and val split
TRAIN_SPLIT = 0.75
VAL_SPLIT = 1-TRAIN_SPLIT

# Set the path
dataset_directory = "./datasets/MNIST"

def _download_dataset():
    print("Downloading the MNIST dataset")
    # Downloading the dataset
    trainData = M(root=dataset_directory, train=True, download=True, transform=ToTensor())
    testData = M(root=dataset_directory, train=False, download=True, transform=ToTensor())
    return (trainData, testData)

def _split_dataset(dataset, train_split_size):
    print("Splitting into train and validation dataset")

    total_dataset = len(dataset)
    numTrainSamples = int(total_dataset*train_split_size)
    numValSamples = total_dataset - numTrainSamples

    (training, validation) = random_split(dataset, [numTrainSamples, numValSamples], generator=torch.Generator().manual_seed(42))

    return (training, validation)

dataset = DotMap({
    "trainLoader" : None,
    "valLoader" : None,
    "testLoader" : None,
    "batch_size" : 64,
    "trainSteps" : None,
    "valSteps" : None
})

def _create_loaders(training, validation, test):
    dataset.trainLoader = DataLoader(training, shuffle=True, batch_size = dataset.batch_size)
    dataset.valLoader = DataLoader(validation, batch_size=dataset.batch_size)
    dataset.testLoader = DataLoader(test, batch_size=dataset.batch_size)
    dataset.trainSteps = len(dataset.trainLoader) / dataset.batch_size
    dataset.valSteps = len(dataset.valLoader) / dataset.batch_size


def prepare_loaders():
    (train, test) = _download_dataset()
    (training, validation) = _split_dataset(train, TRAIN_SPLIT)
    _create_loaders(training, validation, test)
    
def sample_train():
    return dataset.trainLoader.dataset[65]
def sample_test():
    return dataset.testLoader.dataset[0]

def random_test_sample():
    import numpy
    total = len(dataset.testLoader.dataset)
    index = numpy.random.randint(0, total)
    return dataset.testLoader.dataset[index]
    
prepare_loaders()