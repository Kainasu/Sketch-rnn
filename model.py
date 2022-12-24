import math
from typing import Optional, Tuple, Any, List, Union, Literal

import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from torch import optim
from torch.utils.data import DataLoader
import os
from datetime import datetime

import einops

import torch.nn.functional as F
from StrokesDataset import StrokesDataset
from utils import merge_npz_files, load_dataset, render_svg2bitmap

from torch.utils.tensorboard import SummaryWriter


class BivariateGaussianMixture:
    """
    ## Bi-variate Gaussian mixture
    The mixture is represented by $\Pi$ and
    $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
    This class adjusts temperatures and creates the categorical and Gaussian
    distributions from the parameters.
    """

    def __init__(self, pi_logits: torch.Tensor, mu_x: torch.Tensor, mu_y: torch.Tensor,
                 sigma_x: torch.Tensor, sigma_y: torch.Tensor, rho_xy: torch.Tensor):
        self.pi_logits = pi_logits
        self.mu_x = mu_x
        self.mu_y = mu_y
        self.sigma_x = sigma_x
        self.sigma_y = sigma_y
        self.rho_xy = rho_xy

    @property
    def n_distributions(self):
        """Number of distributions in the mixture, $M$"""
        return self.pi_logits.shape[-1]

    def set_temperature(self, temperature: float):
        """
        Adjust by temperature $\tau$
        """
        # $$\hat{\Pi_k} \leftarrow \frac{\hat{\Pi_k}}{\tau}$$
        self.pi_logits /= temperature
        # $$\sigma^2_x \leftarrow \sigma^2_x \tau$$
        self.sigma_x *= math.sqrt(temperature)
        # $$\sigma^2_y \leftarrow \sigma^2_y \tau$$
        self.sigma_y *= math.sqrt(temperature)

    def get_distribution(self):
        # Clamp $\sigma_x$, $\sigma_y$ and $\rho_{xy}$ to avoid getting `NaN`s
        sigma_x = torch.clamp_min(self.sigma_x, 1e-5)
        sigma_y = torch.clamp_min(self.sigma_y, 1e-5)
        rho_xy = torch.clamp(self.rho_xy, -1 + 1e-5, 1 - 1e-5)

        # Get means
        mean = torch.stack([self.mu_x, self.mu_y], -1)
        # Get covariance matrix
        cov = torch.stack([
            sigma_x * sigma_x, rho_xy * sigma_x * sigma_y,
            rho_xy * sigma_x * sigma_y, sigma_y * sigma_y
        ], -1)
        cov = cov.view(*sigma_y.shape, 2, 2)

        # Create bi-variate normal distribution.
        #
        # 📝 It would be efficient to `scale_tril` matrix as `[[a, 0], [b, c]]`
        # where
        # $$a = \sigma_x, b = \rho_{xy} \sigma_y, c = \sigma_y \sqrt{1 - \rho^2_{xy}}$$.
        # But for simplicity we use co-variance matrix.
        # [This is a good resource](https://www2.stat.duke.edu/courses/Spring12/sta104.1/Lectures/Lec22.pdf)
        # if you want to read up more about bi-variate distributions, their co-variance matrix,
        # and probability density function.
        multi_dist = torch.distributions.MultivariateNormal(mean, covariance_matrix=cov)

        # Create categorical distribution $\Pi$ from logits
        cat_dist = torch.distributions.Categorical(logits=self.pi_logits)

        #
        return cat_dist, multi_dist


class EncoderCNN(nn.Module):
    """
    ## Encoder module
    This consists of a bidirectional LSTM
    """

    def __init__(self, d_z: int, device : str):
        super().__init__()
        self.device = device
        self.filter_hp = torch.tensor([[[[-1, -1, -1],
                                             [-1, 8, -1],
                                             [-1, -1, -1]]]]).float().to(self.device)

        self.conv1 = nn.Conv2d(in_channels = 1, out_channels= 4, kernel_size = 2, stride =2)
        self.conv2 = nn.Conv2d(in_channels = 4, out_channels= 4, kernel_size = 2, stride =1)
        self.conv3 = nn.Conv2d(in_channels = 4, out_channels= 8, kernel_size = 2, stride =2)
        self.conv4 = nn.Conv2d(in_channels = 8, out_channels= 8, kernel_size = 2, stride =1)
        self.conv5 = nn.Conv2d(in_channels = 8, out_channels= 8, kernel_size = 2, stride =2)
        self.conv6 = nn.Conv2d(in_channels = 8, out_channels= 8, kernel_size = 2, stride =1)

        # Head to get $\mu$
        self.mu_head = nn.Linear(128, d_z)
        # Head to get $\hat{\sigma}$
        self.sigma_head = nn.Linear(128, d_z)

    def high_pass_filtering(self, img_in):
        """
        high pass filtering
        :param img_in: [N, H, W, 1]
        :return: img_out: [N, H, W, 1]
        """
        img_out = F.conv2d(img_in, self.filter_hp)
        return img_out


    def forward(self, inputs: torch.Tensor, state=None):
        x = self.high_pass_filtering(inputs)
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = torch.tanh(self.conv6(x))
        
        x = x.view(inputs.shape[0], -1)

        # $\mu$
        mu = self.mu_head(x)
        # $\hat{\sigma}$
        sigma_hat = self.sigma_head(x)
        # $\sigma = \exp(\frac{\hat{\sigma}}{2})$
        sigma = torch.exp(sigma_hat / 2.)

        # Sample $z = \mu + \sigma \cdot \mathcal{N}(0, I)$
        z = mu + sigma * torch.normal(mu.new_zeros(mu.shape), mu.new_ones(mu.shape))
      
        return z, mu, sigma_hat

class EncoderRNN(nn.Module):
    """
    ## Encoder module
    This consists of a bidirectional LSTM
    """

    def __init__(self, d_z: int, enc_hidden_size: int):
        super().__init__()
        # Create a bidirectional LSTM taking a sequence of
        # $(\Delta x, \Delta y, p_1, p_2, p_3)$ as input.
        self.lstm = nn.LSTM(5, enc_hidden_size, bidirectional=True)
        # Head to get $\mu$
        self.mu_head = nn.Linear(2 * enc_hidden_size, d_z)
        # Head to get $\hat{\sigma}$
        self.sigma_head = nn.Linear(2 * enc_hidden_size, d_z)

    def forward(self, inputs: torch.Tensor, state=None):
        # The hidden state of the bidirectional LSTM is the concatenation of the
        # output of the last token in the forward direction and
        # first token in the reverse direction, which is what we want.
        # $$h_{\rightarrow} = encode_{\rightarrow}(S),
        # h_{\leftarrow} = encode←_{\leftarrow}(S_{reverse}),
        # h = [h_{\rightarrow}; h_{\leftarrow}]$$
        _, (hidden, cell) = self.lstm(inputs.float(), state)
        # The state has shape `[2, batch_size, hidden_size]`,
        # where the first dimension is the direction.
        # We rearrange it to get $h = [h_{\rightarrow}; h_{\leftarrow}]$
        hidden = einops.rearrange(hidden, 'fb b h -> b (fb h)')

        # $\mu$
        mu = self.mu_head(hidden)
        # $\hat{\sigma}$
        sigma_hat = self.sigma_head(hidden)
        # $\sigma = \exp(\frac{\hat{\sigma}}{2})$
        sigma = torch.exp(sigma_hat / 2.)

        # Sample $z = \mu + \sigma \cdot \mathcal{N}(0, I)$
        z = mu + sigma * torch.normal(mu.new_zeros(mu.shape), mu.new_ones(mu.shape))

        #
        return z, mu, sigma_hat

class DecoderRNN(nn.Module):
    """
    ## Decoder module
    This consists of a LSTM
    """

    def __init__(self, d_z: int, dec_hidden_size: int, n_distributions: int):
        super().__init__()
        # LSTM takes $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$ as input
        self.lstm = nn.LSTM(d_z + 5, dec_hidden_size)

        # Initial state of the LSTM is $[h_0; c_0] = \tanh(W_{z}z + b_z)$.
        # `init_state` is the linear transformation for this
        self.init_state = nn.Linear(d_z, 2 * dec_hidden_size)

        # This layer produces outputs for each of the `n_distributions`.
        # Each distribution needs six parameters
        # $(\hat{\Pi_i}, \mu_{x_i}, \mu_{y_i}, \hat{\sigma_{x_i}}, \hat{\sigma_{y_i}} \hat{\rho_{xy_i}})$
        self.mixtures = nn.Linear(dec_hidden_size, 6 * n_distributions)

        # This head is for the logits $(\hat{q_1}, \hat{q_2}, \hat{q_3})$
        self.q_head = nn.Linear(dec_hidden_size, 3)
        # This is to calculate $\log(q_k)$ where
        # $$q_k = \operatorname{softmax}(\hat{q})_k = \frac{\exp(\hat{q_k})}{\sum_{j = 1}^3 \exp(\hat{q_j})}$$
        self.q_log_softmax = nn.LogSoftmax(-1)

        # These parameters are stored for future reference
        self.n_distributions = n_distributions
        self.dec_hidden_size = dec_hidden_size

    def forward(self, x: torch.Tensor, z: torch.Tensor, state: Optional[Tuple[torch.Tensor, torch.Tensor]]):
        # The target/expected vectors of strokes
        #self.output_x = self.input_data[:, 1:self.hps.max_seq_len + 1, :]  # [N, max_seq_len, 5]
        # vectors of strokes to be fed to decoder (same as above, but lagged behind
        # one step to include initial dummy value of (0, 0, 1, 0, 0))
        #self.input_x = self.input_data[:, :self.hps.max_seq_len, :]  # [N, max_seq_len, 5]

        # Calculate the initial state
        if state is None:
            # $[h_0; c_0] = \tanh(W_{z}z + b_z)$
            h, c = torch.split(torch.tanh(self.init_state(z)), self.dec_hidden_size, 1)
            # `h` and `c` have shapes `[batch_size, lstm_size]`. We want to shape them
            # to `[1, batch_size, lstm_size]` because that's the shape used in LSTM.
            state = (h.unsqueeze(0).contiguous(), c.unsqueeze(0).contiguous())
            #state = torch.cat(state, stroke)

        # Run the LSTM
        outputs, state = self.lstm(x, state)

        # Get $\log(q)$
        q_logits = self.q_log_softmax(self.q_head(outputs))

        # Get $(\hat{\Pi_i}, \mu_{x,i}, \mu_{y,i}, \hat{\sigma_{x,i}},
        # \hat{\sigma_{y,i}} \hat{\rho_{xy,i}})$.
        # `torch.split` splits the output into 6 tensors of size `self.n_distribution`
        # across dimension `2`.
        pi_logits, mu_x, mu_y, sigma_x, sigma_y, rho_xy = \
            torch.split(self.mixtures(outputs), self.n_distributions, 2)

        # Create a bi-variate Gaussian mixture
        # $\Pi$ and 
        # $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
        # where
        # $$\sigma_{x,i} = \exp(\hat{\sigma_{x,i}}), \sigma_{y,i} = \exp(\hat{\sigma_{y,i}}),
        # \rho_{xy,i} = \tanh(\hat{\rho_{xy,i}})$$
        # and
        # $$\Pi_i = \operatorname{softmax}(\hat{\Pi})_i = \frac{\exp(\hat{\Pi_i})}{\sum_{j = 1}^3 \exp(\hat{\Pi_j})}$$
        #
        # $\Pi$ is the categorical probabilities of choosing the distribution out of the mixture
        # $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
        dist = BivariateGaussianMixture(pi_logits, mu_x, mu_y,
                                        torch.exp(sigma_x), torch.exp(sigma_y), torch.tanh(rho_xy))

        #
        return dist, q_logits, state


class ReconstructionLoss(nn.Module):
    """
    ## Reconstruction Loss
    """

    def forward(self, mask: torch.Tensor, target: torch.Tensor,
                dist: 'BivariateGaussianMixture', q_logits: torch.Tensor):
        # Get $\Pi$ and $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
        pi, mix = dist.get_distribution()
        # `target` has shape `[seq_len, batch_size, 5]` where the last dimension is the features
        # $(\Delta x, \Delta y, p_1, p_2, p_3)$.
        # We want to get $\Delta x, \Delta$ y and get the probabilities from each of the distributions
        # in the mixture $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$.
        #
        # `xy` will have shape `[seq_len, batch_size, n_distributions, 2]`
        xy = target[:, :, 0:2].unsqueeze(-2).expand(-1, -1, dist.n_distributions, -1)
        # Calculate the probabilities
        # $$p(\Delta x, \Delta y) =
        # \sum_{j=1}^M \Pi_j \mathcal{N} \big( \Delta x, \Delta y \vert
        # \mu_{x,j}, \mu_{y,j}, \sigma_{x,j}, \sigma_{y,j}, \rho_{xy,j}
        # \big)$$
        probs = torch.sum(pi.probs * torch.exp(mix.log_prob(xy)), 2)

        # $$L_s = - \frac{1}{N_{max}} \sum_{i=1}^{N_s} \log \big (p(\Delta x, \Delta y) \big)$$
        # Although `probs` has $N_{max}$ (`longest_seq_len`) elements, the sum is only taken
        # upto $N_s$ because the rest is masked out.
        #
        # It might feel like we should be taking the sum and dividing by $N_s$ and not $N_{max}$,
        # but this will give higher weight for individual predictions in shorter sequences.
        # We give equal weight to each prediction $p(\Delta x, \Delta y)$ when we divide by $N_{max}$
        loss_stroke = -torch.mean(mask * torch.log(1e-5 + probs))

        # $$L_p = - \frac{1}{N_{max}} \sum_{i=1}^{N_{max}} \sum_{k=1}^{3} p_{k,i} \log(q_{k,i})$$
        loss_pen = -torch.mean(target[:, :, 2:] * q_logits)

        # $$L_R = L_s + L_p$$
        return loss_stroke + loss_pen

class KLDivLoss(nn.Module):
    """
    ## KL-Divergence loss
    This calculates the KL divergence between a given normal distribution and $\mathcal{N}(0, 1)$
    """

    def forward(self, sigma_hat: torch.Tensor, mu: torch.Tensor):
        # $$L_{KL} = - \frac{1}{2 N_z} \bigg( 1 + \hat{\sigma} - \mu^2 - \exp(\hat{\sigma}) \bigg)$$
        return -0.5 * torch.mean(1 + sigma_hat - mu ** 2 - torch.exp(sigma_hat))

Model = Literal['sketchrnn', 'pix2seq']
class Sampler:
    """
    ## Sampler
    This samples a sketch from the decoder and plots it
    """

    def __init__(self, encoder: Union[EncoderCNN, EncoderRNN], decoder: DecoderRNN, model : Model):
        self.decoder = decoder
        self.encoder = encoder
        self.model = model

    def sample(self, data: torch.Tensor, temperature: float, imgs: Optional[torch.tensor] = None, filename: Optional[str] = None):
        # $N_{max}$
        longest_seq_len = len(data)
        
        # Get $z$ from the encoder
        if self.model == 'sketchrnn' :
            z, _, _ = self.encoder(data)
        else :
            img = imgs.unsqueeze(0).unsqueeze(0).float()                  
            z, _, _ = self.encoder(img)

        # Start-of-sequence stroke is $(0, 0, 1, 0, 0)$
        s = data.new_tensor([0, 0, 1, 0, 0])
        seq = [s]
        # Initial decoder is `None`.
        # The decoder will initialize it to $[h_0; c_0] = \tanh(W_{z}z + b_z)$
        state = None

        # We don't need gradients
        with torch.no_grad():
            # Sample $N_{max}$ strokes
            for i in range(longest_seq_len):
                # $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$ is the input to the decoder
                data = torch.cat([s.view(1, 1, -1), z.unsqueeze(0)], 2)
                # Get $\Pi$, $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$,
                # $q$ and the next state from the decoder
                dist, q_logits, state = self.decoder(data, z, state)
                # Sample a stroke
                s = self._sample_step(dist, q_logits, temperature)
                # Add the new stroke to the sequence of strokes
                seq.append(s)
                # Stop sampling if $p_3 = 1$. This indicates that sketching has stopped
                if s[4] == 1:
                    break

        # Create a PyTorch tensor of the sequence of strokes
        seq = torch.stack(seq)

        # Plot the sequence of strokes
        self.plot(seq, filename)

    @staticmethod
    def _sample_step(dist: 'BivariateGaussianMixture', q_logits: torch.Tensor, temperature: float):
        # Set temperature $\tau$ for sampling. This is implemented in class `BivariateGaussianMixture`.
        dist.set_temperature(temperature)
        # Get temperature adjusted $\Pi$ and $\mathcal{N}(\mu_{x}, \mu_{y}, \sigma_{x}, \sigma_{y}, \rho_{xy})$
        pi, mix = dist.get_distribution()
        # Sample from $\Pi$ the index of the distribution to use from the mixture
        idx = pi.sample()[0, 0]

        # Create categorical distribution $q$ with log-probabilities `q_logits` or $\hat{q}$
        q = torch.distributions.Categorical(logits=q_logits / temperature)
        # Sample from $q$
        q_idx = q.sample()[0, 0]

        # Sample from the normal distributions in the mixture and pick the one indexed by `idx`
        xy = mix.sample()[0, 0, idx]

        # Create an empty stroke $(\Delta x, \Delta y, q_1, q_2, q_3)$
        stroke = q_logits.new_zeros(5)
        # Set $\Delta x, \Delta y$
        stroke[:2] = xy
        # Set $q_1, q_2, q_3$
        stroke[q_idx + 2] = 1
        #
        return stroke

    @staticmethod
    def plot(seq: torch.Tensor, filename: Optional[str] = None):
        # Take the cumulative sums of $(\Delta x, \Delta y)$ to get $(x, y)$
        seq[:, 0:2] = torch.cumsum(seq[:, 0:2], dim=0)
        # Create a new numpy array of the form $(x, y, q_2)$
        seq[:, 2] = seq[:, 3]
        seq = seq[:, 0:3].detach().cpu().numpy()

        # Split the array at points where $q_2$ is $1$.
        # i.e. split the array of strokes at the points where the pen is lifted from the paper.
        # This gives a list of sequence of strokes.
        strokes = np.split(seq, np.where(seq[:, 2] > 0)[0] + 1)
        # Plot each sequence of strokes
        for s in strokes:
            plt.plot(s[:, 0], -s[:, 1])
        # Don't show axes
        plt.axis('off')
        # Show the plot
        if filename:
            plt.savefig(filename)
        else:
            plt.show()

class Configs():
    """
    ## Configurations
    These are default configurations which can later be adjusted by passing a `dict`.
    """

    # Device configurations to pick the device to run the experiment
    #device: torch.device = DeviceConfigs()
    #
    encoder: EncoderCNN
    decoder: DecoderRNN
    optimizer: optim.Adam
    sampler: Sampler

    dataset_name: str
    train_loader: DataLoader
    valid_loader: DataLoader
    train_dataset: StrokesDataset
    valid_dataset: StrokesDataset

    # Encoder and decoder sizes
    enc_hidden_size = 256
    dec_hidden_size = 512

    # Batch size
    batch_size = 256

    # Number of features in $z$
    d_z = 128
    # Number of distributions in the mixture, $M$
    n_distributions = 20

    # Weight of KL divergence loss, $w_{KL}$
    kl_div_loss_weight = 0.5
    # Gradient clipping
    grad_clip = 1.
    # Temperature $\tau$ for sampling
    temperature = 0.4

    # Filter out stroke sequences longer than $200$
    max_seq_length = 200

    kl_div_loss = KLDivLoss()
    reconstruction_loss = ReconstructionLoss()

    def __init__(self, classes: List[str], device_id: int, model: Model, suffix: str, percentage: float):
        self.last_loss = None
        self.device = f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu'
        self.model = model
        self.suffix = suffix

        # Initialize encoder & decoder
        if self.model == 'sketchrnn':
            self.encoder = EncoderRNN(self.d_z, self.enc_hidden_size).to(self.device)
        else:            
            self.encoder = EncoderCNN(self.d_z, self.device).to(self.device)
            
        self.decoder = DecoderRNN(self.d_z, self.dec_hidden_size, self.n_distributions).to(self.device)

        # Set optimizer. Things like type of optimizer and learning rate are configurable
        #optimizer = OptimizerConfigs()
        #optimizer.parameters = list(self.encoder.parameters()) + list(self.decoder.parameters())
        #self.optimizer = optimizer
        self.optimizer = optim.Adam(list(self.encoder.parameters()) + list(self.decoder.parameters()), lr=0.001)

        # Create sampler
        self.sampler = Sampler(self.encoder, self.decoder, self.model)

        # `npz` file path is `data/sketch/[DATASET NAME].npz`

        if self.model == 'sketchrnn':
            dataset = merge_npz_files([os.path.join('data', f'{c}.npz') for c in classes], percentage)
            # Create training dataset
            self.train_dataset = StrokesDataset(dataset['train'], self.max_seq_length)
            # Create validation dataset
            self.valid_dataset = StrokesDataset(dataset['valid'], self.max_seq_length, None, self.train_dataset.scale)        
        else :        
            data_sets = [f'{c}.npz' for c in classes]
            dataset, png_paths, _ = load_dataset('./', data_sets, percentage)
            if not os.path.exists(png_paths['test'][-1]): 
                #Generate all images as png file
                render_svg2bitmap('./', data_sets, percentage)        
            # Create training dataset
            self.train_dataset = StrokesDataset(dataset['train'], self.max_seq_length, png_paths['train'])
            # Create validation dataset
            self.valid_dataset = StrokesDataset(dataset['valid'], self.max_seq_length, png_paths['valid'], self.train_dataset.scale)

        # Create training data loader
        self.train_loader = DataLoader(self.train_dataset, self.batch_size, shuffle=True)
        # Create validation data loader
        self.valid_loader = DataLoader(self.valid_dataset, self.batch_size)

        self.state_modules = []

        self.sampling_images_indices = None
        self.train_images_indices = None
        self.writer = SummaryWriter(self.suffix)

        self.losses = []

    def save(self, n_images=6):
        
        # Init at first time
        if not self.sampling_images_indices:
            self.sampling_images_indices = [0, 1, 2, 125, 126, 127]#[np.random.choice(len(self.valid_dataset)) for i in range(n_images)]
        #if not self.train_images_indices:
        #    self.train_images_indices = [np.random.choice(len(self.train_dataset)) for i in range(n_images)]

        dirname = 'checkpoints' + ('_' + self.suffix) if self.suffix != '' else ''
        if not os.path.exists(dirname):
            os.makedirs(dirname)

        now = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')

        torch.save({
            'encoder': self.encoder.state_dict(),
            'decoder': self.decoder.state_dict()
        }, os.path.join(dirname, f'model-{now}.dict'))

        """ # Uncomment to save training images
        plt.close()
        plt.figure(figsize=(22,20))
        plt.tight_layout()


        plt.subplot(n_images, 6, 1)
        filename = os.path.join(dirname, f'model-{now}-train.png')
        for i, index in enumerate(self.sampling_images_indices):
            plt.subplot(n_images, 6, i*6+1)
            plt.title(f'Train dataset image {index}')
            if self.model == 'sketchrnn':
                data, *_ = self.train_dataset[index]
                img = None
            else : 
                data, _, img = self.train_dataset[index]
                img = torch.tensor(img).to(self.device)
            Sampler.plot(data.clone(), filename)
            data_ = data.unsqueeze(1).to(self.device)            
            for j, temperature in enumerate(np.linspace(0.2, 0.8, 5)):
                plt.subplot(n_images, 6, i*6+2+j)
                plt.title(f'temperature={temperature}')
                self.sampler.sample(data_, temperature, img, filename)
        """

        plt.close()
        plt.figure(figsize=(20,18))
        plt.tight_layout()


        plt.subplot(n_images, 6, 1)
        filename = os.path.join(dirname, f'model-{now}-validation.png')
        for i, index in enumerate(self.sampling_images_indices):
            plt.subplot(n_images, 6, i*6+1)
            plt.title(f'Validation dataset image {index}')
            if self.model == 'sketchrnn':
                data, *_ = self.valid_dataset[index]
                img = None
            else : 
                data, _, img = self.valid_dataset[index]
                img = torch.tensor(img).to(self.device)
            Sampler.plot(data.clone(), filename)
            data_ = data.unsqueeze(1).to(self.device)   
            for j, temperature in enumerate(np.linspace(0.2, 0.8, 5)):
                plt.subplot(n_images, 6, i*6+2+j)
                plt.title(f'temperature={temperature}')
                self.sampler.sample(data_, temperature, img, filename)

                
    def load(self, path):
        checkpoint = torch.load(path)
        self.encoder.load_state_dict(checkpoint['encoder'])
        self.decoder.load_state_dict(checkpoint['decoder'])


    def step(self, batch: Any, is_train=True):
        self.encoder.train(is_train)
        self.decoder.train(is_train)

        # Move `data` and `mask` to device and swap the sequence and batch dimensions.
        # `data` will have shape `[seq_len, batch_size, 5]` and
        # `mask` will have shape `[seq_len, batch_size]`.
        data = batch[0].to(self.device).transpose(0, 1)
        mask = batch[1].to(self.device).transpose(0, 1)

        if self.model == 'sketchrnn':
            z, mu, sigma_hat = self.encoder(data)
        else:            
            img = batch[2].to(self.device).view(len(batch[2]), 1, 48,48).float()                
            z, mu, sigma_hat = self.encoder(img)

        # Decode the mixture of distributions and $\hat{q}$

        # Concatenate $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$
        z_stack = z.unsqueeze(0).expand(data.shape[0] - 1, -1, -1)
        inputs = torch.cat([data[:-1], z_stack], 2)
        # Get mixture of distributions and $\hat{q}$
        dist, q_logits, _ = self.decoder(inputs, z, None)

        # Compute the loss
        # $L_R$
        reconstruction_loss = self.reconstruction_loss(mask, data[1:], dist, q_logits)
        
        if self.model == 'sketchrnn':
            # $L_{KL}$
            kl_loss = self.kl_div_loss(sigma_hat, mu)
            # $Loss = L_R + w_{KL} L_{KL}$
            loss = reconstruction_loss + self.kl_div_loss_weight * kl_loss
        else :
            # $Loss = L_R
            loss = reconstruction_loss

        # Only if we are in training state
        if is_train:

            # Set `grad` to zero
            self.optimizer.zero_grad()            
            # Compute gradients
            loss.backward()
            self.losses.append(loss)
            #print(loss.item(), end='\r')
            # Clip gradients
            nn.utils.clip_grad_norm_(self.encoder.parameters(), self.grad_clip)
            nn.utils.clip_grad_norm_(self.decoder.parameters(), self.grad_clip)
            # Optimize
            self.optimizer.step()
        
    def loss_batch(self, batch):
        with torch.no_grad():
            data = batch[0].to(self.device).transpose(0, 1)
            mask = batch[1].to(self.device).transpose(0, 1)

            if self.model == 'sketchrnn':
                z, mu, sigma_hat = self.encoder(data)
            else:            
                img = batch[2].to(self.device).view(len(batch[2]), 1, 48,48).float()                
                z, mu, sigma_hat = self.encoder(img)
                                    
            # Concatenate $[(\Delta x, \Delta y, p_1, p_2, p_3); z]$
            z_stack = z.unsqueeze(0).expand(data.shape[0] - 1, -1, -1)
            inputs = torch.cat([data[:-1], z_stack], 2)
            # Get mixture of distributions and $\hat{q}$
            dist, q_logits, _ = self.decoder(inputs, z, None)
            
            # Compute the loss
            # $L_R$
            reconstruction_loss = self.reconstruction_loss(mask, data[1:], dist, q_logits)

            if self.model == 'sketchrnn':
                # $L_{KL}$
                kl_loss = self.kl_div_loss(sigma_hat, mu)
                # $Loss = L_R + w_{KL} L_{KL}$
                loss = reconstruction_loss + self.kl_div_loss_weight * kl_loss
            else :
                # $Loss = L_R
                loss = reconstruction_loss

            return loss

    def sample(self, filename: Optional[str] = None, index=None):
        # Randomly pick a sample from validation dataset to encoder
        if self.model == 'sketchrnn':
            data, *_ = self.valid_dataset[np.random.choice(len(self.valid_dataset)) if not index else index]
            img = None
        else: 
            data, _, img = self.valid_dataset[np.random.choice(len(self.valid_dataset)) if not index else index]   
            img = torch.tensor(img).to(self.device)     
        # Add batch dimension and move it to device
        data = data.unsqueeze(1).to(self.device)
        # Sample
        self.sampler.sample(data, self.temperature, img, filename)