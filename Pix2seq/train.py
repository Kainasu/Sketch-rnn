from Pix2seq import Configs
from utils import download_dataset
from tqdm import tqdm
import argparse

parser = argparse.ArgumentParser()

# Add argument for the device ID
parser.add_argument('--device', type=int,
                    help='ID of the device to use for training', default=5)

# Add argument for the list of classes
parser.add_argument('--classes', type=str, nargs='+',
                    help='List of classes to train the model on', default=['cat'])

# Add argument for the number of epochs
parser.add_argument('--epochs', type=int,
                    help='Number of epochs to train the model for', default=100)

args = parser.parse_args()

# Get the values of the arguments
device_id = args.device
classes = args.classes
epochs = args.epochs

for c in classes:
    download_dataset(c)

c = Configs(classes, device_id)

for i in tqdm(range(epochs), desc='Epochs'):
    for batch in tqdm(c.train_loader, desc='Batch training'):
        c.step(batch)
    c.save()

# Example to run the script : python script.py --device 0 --classes class1 class2 class3 --epochs 10
