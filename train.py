from model import Configs
from utils import download_dataset, latest_checkpoint
from tqdm import tqdm
import argparse
import torch
from torch.optim.lr_scheduler import StepLR

parser = argparse.ArgumentParser()

# Add argument for the device ID
parser.add_argument('--device', type=int,
                    help='ID of the device to use for training', default=5)

# Add argument for the list of classes
parser.add_argument('--classes', type=str, nargs='+',
                    help='List of classes to train the model on', default=['cat'])

# Add argument for the number of epochs
parser.add_argument('--epochs', type=int,
                    help='Number of epochs to train the model for', default=200)

# Add argument for the number of epochs
parser.add_argument('--checkpoint', type=str,
                    help='Model to load (last for latest model)', default=None)

# Add argument for the model
parser.add_argument('--model', type=str, choices=['sketchrnn', 'pix2seq'],
                    help='Model used (sketchrnn or pix2seq)', default='sketchrnn')

# Add argument for the model
parser.add_argument('--suffix', type=str,
                    help='Folder suffix', default='')

# Add argument for the number of epochs
parser.add_argument('--percentage', type=float,
                    help='Percentage of the training dataset to take', default=1.)


args = parser.parse_args()

# Get the values of the arguments
device_id = args.device
classes = args.classes
epochs = args.epochs
checkpoint = args.checkpoint
model = args.model
suffix = args.suffix
percentage = args.percentage

for c in classes:
    download_dataset(c)

c = Configs(classes, device_id, model, suffix, percentage)

if checkpoint:
    if checkpoint == "last":
        c.load(latest_checkpoint())
    else:
        c.load(checkpoint)

scheduler = StepLR(c.optimizer, step_size=epochs//4, gamma=0.5)

for i in tqdm(range(epochs), desc='Epochs'):
    for batch in tqdm(c.train_loader, desc='Batch training'):
        c.step(batch)
    if i % 20 == 0:
        c.save()
    c.writer.add_scalar("Loss/validation", 
        torch.Tensor([c.loss_batch(batch) for batch in c.valid_loader]).mean(), i)
    c.writer.add_scalar("Loss/train", torch.Tensor(c.losses).mean(), i)
    c.losses = []
    scheduler.step()

# Example to run the script : python script.py --device 0 --classes class1 class2 class3 --epochs 10
