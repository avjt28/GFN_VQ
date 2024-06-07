from __future__ import print_function
import os
import datetime
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import savgol_filter
from six.moves import xrange
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torchvision.utils import make_grid
import tqdm
import pandas as pd
from torch.optim.lr_scheduler import ReduceLROnPlateau

from network import Encoder, Decoder, VectorQuantizerEMA, GFlowNet, LatentDictionary

## PARAMETERS

batch_size = 256
epochs = 10
alternate_every = 50
num_hiddens = 128
num_residual_hiddens = 32
num_residual_layers = 2

embedding_dim = 64
num_embeddings = 512

channels = 1024

commitment_cost = 0.25
greedy_prob = 1
decay = 0.99

rand_prob = 0.2
sleep_prob= 0.3
gfn_logZ_lr = 1e-3
gfn_lr = 1e-3
learning_rate = 1e-3
# load_model = None
load_model = 'base'

device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
model_dir = 'models/'


now = datetime.datetime.now()
model_name = now.strftime("%Y%m%d_%H%M%S")


def sample(gfn, img, p=1, rand_prob=0.05):
    '''Samples from the GFlowNet policy. Implemented for the autoregressive policy.
    Args:
        gfn (torch.nn.Module): GFlowNet network
        img (torch.Tensor): Image to condition GFlowNet sampling on
        p (float): If p > 0, raises the predicted action probabilites to the power of p. If p < 0 samples greedily from the GFlowNet
        rand_prob (float): Probability to perform random action
    Returns:
        state (torch.Tensor): Returns the state constructed by sampling actions from the GFlowNet
        logprobs (torch.Tensor): Sum of log-probabilities of the actions taken
    '''
    device = img.device
    batch_size, dict_size, lh, lw = img.shape[0], gfn.dictionary_size, gfn.lh, gfn.lw 
    steps = lh*lw

    # Initialize state with zeros
    state = torch.zeros((batch_size, dict_size*lh*lw)).float().to(device)

    logprobs = torch.zeros((batch_size,1)).float().to(device)
    for i in range(steps):
        # Predict logits over next states
        pred_logits = gfn(img, state.clone().float())

        # Mask out already completed states (AR policy)
        mask = torch.zeros((batch_size, lh*lw), device=device)
        mask[:,i:i+1] = 1
        mask = mask.view(batch_size,lh,lw).tile(dict_size,1,1,1).permute([1,0,2,3]).reshape(batch_size, dict_size*lh*lw)
                
        # Compute probabilites over next actions
        # This is equivalent to computing softmax over 'unmasked' positions
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_probs = (pred_probs + 1e-9) / torch.sum((pred_probs + 1e-9) * mask, dim=-1, keepdims=True)
            
        # Perform actions
        if p > 0:
            pred_step = torch.multinomial(pred_probs**p * mask, 1)
        else:
            pred_step = torch.argmax(pred_probs * mask, 1, keepdims=True)
        # Random steps
        rand_step = torch.multinomial(torch.ones_like(pred_probs) * mask, 1)
        rand_update = (torch.rand((batch_size,1), device=device) < rand_prob).long()
        pred_step = (1-rand_update)*pred_step + rand_update*rand_step
        
        # Add step to state
        state.scatter_(1, pred_step, 1)
        # Add log prob of selected step
        logprobs = logprobs + (pred_probs.gather(1, pred_step)).log()
                
    assert torch.all(state.view(batch_size, dict_size, lh, lw).sum(1) == 1), 'State incomplete'

    return state, logprobs


def sleep_step(encoder, gfn, decoder, latent_dict, batch_size=128):
    '''Performs a sleep-phase step by sampling a random (state, image) pair and computing log-probabilities
    Args:
        gfn (torch.nn.Module): GFlowNet network
        decoder (torch.nn.Module): Decoder network
        latent_dict (torch.nn.Module): Discrete latent->Embeddings dictionary
        batch_size (int): Number of sleep-phase samples to draw
    Returns:
        logprobs (torch.Tensor): Sum of log-probabilities of sampled actions
        img_in (torch.Tensor): Sampled image
    '''
    device = latent_dict.dictionary.weight.device
    dict_size, embedding_dim, lh, lw = gfn.dictionary_size, gfn.embedding_dim, gfn.lh, gfn.lw
    steps = lh*lw
    
    # Sample a random latent
    # Trajectory is fixed since we use an AR policy
    rand_t = torch.stack([torch.arange(0, lh*lw, device=device) for _ in range(batch_size)])
    random_latent = torch.randint(0, dict_size, (batch_size,lh*lw), device=device)
    random_latent_state = F.one_hot(random_latent, dict_size).permute([0,2,1]).reshape(batch_size, -1)
    
    # Sample an image from the decoder
    state_in = torch.sum(latent_dict.dictionary.weight.view(1,dict_size,embedding_dim,1,1) * random_latent_state.reshape(batch_size, dict_size, 1, lh, lw), dim=1)
    img_in = decoder(state_in)
    
    state = torch.zeros((batch_size, dict_size*lh*lw)).float().to(device)
    logprobs = torch.zeros((batch_size,1)).float().to(device)
    for i in range(steps):
        # Predict logits over next states
        z = encoder(img_in)
        pred_logits = gfn(z, state.clone().float())

        # Mask out already completed states - AR policy
        mask = torch.zeros((batch_size, lh*lw), device=device)
        mask[:,i:i+1] = 1
        mask = mask.view(batch_size,lh,lw).tile(dict_size,1,1,1).permute([1,0,2,3]).reshape(batch_size, dict_size*lh*lw)
        
        # Compute probabilites over next actions
        # This is equivalent to computing softmax over 'unmasked' positions
        pred_probs = F.softmax(pred_logits, dim=-1)
        pred_probs = (pred_probs + 1e-9) / torch.sum((pred_probs+1e-9) * mask, dim=-1, keepdims=True)

        # Perform action - Predetermined by the sampled latent
        pred_step = random_latent[:,[i]] * lh*lw + rand_t[:,[i]]
        
        # Add step to state
        state.scatter_(1, pred_step, 1)

        # Add log prob of sampled step
        logprobs = logprobs + (pred_probs.gather(1, pred_step)).log()
        
    assert torch.all(state.view(batch_size, dict_size, lh, lw).sum(1) == 1), 'State incomplete'
    
    return logprobs, img_in



## LOAD DATA


training_data = datasets.CIFAR10(root="data", train=True, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
validation_data = datasets.CIFAR10(root="data", train=False, download=True,
                                  transform=transforms.Compose([
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.5,0.5,0.5), (1.0,1.0,1.0))
                                  ]))
data_variance = np.var(training_data.data / 255.0)
training_loader = DataLoader(training_data, batch_size=batch_size, shuffle=True, pin_memory=True)
validation_loader = DataLoader(validation_data, batch_size=32, shuffle=True, pin_memory=True)



## LOAD MODEL

encoder = Encoder(3, num_hiddens, num_residual_layers, num_residual_hiddens, embedding_dim).to(device)

gfn = GFlowNet(channels=channels, dictionary_size=num_embeddings, embedding_dim= embedding_dim).to(device)
gfn.train()
vq_vae = VectorQuantizerEMA(num_embeddings, embedding_dim, commitment_cost, decay).to(device)

latent_dict = LatentDictionary(embedding_dim=embedding_dim, dictionary_size=num_embeddings).to(device)
latent_dict.train()
decoder = Decoder(embedding_dim, num_hiddens, num_residual_layers, num_residual_hiddens).to(device)


if load_model!=None:
    # gfn.load_state_dict(torch.load(model_dir + load_model + '/gfn_.pth'))
    encoder.load_state_dict(torch.load(model_dir + load_model + '/encoder_.pth'))
    vq_vae.load_state_dict(torch.load(model_dir+load_model+'/vq_vae_.pth'))
    decoder.load_state_dict(torch.load(model_dir+load_model+'/decoder_.pth'))
    latent_dict.dictionary.weight.data.copy_(vq_vae._embedding.weight.data)


gfn_opt = torch.optim.Adam([{'params': gfn.img_enc.parameters()},
                            {'params': gfn.state_enc.parameters()},
                            {'params': gfn.pred.parameters()},
                            {'params': gfn.logZ, 'lr': gfn_logZ_lr},], # Higher lr for logZ
                            lr=gfn_lr)

optimizer = optim.Adam(list(decoder.parameters())+list(latent_dict.parameters()), lr=learning_rate)#, amsgrad=False)


encoder.train()
vq_vae.train()
decoder.train()
steps = gfn.lh*gfn.lw
train_res_vq_loss = []
train_res_recon_error = []
train_res_perplexity = []


for e in range(epochs):
    print(f'Epoch [{e+1}/{epochs}]')
    batch_bar = tqdm.tqdm(training_loader)
    epoch_vq_loss = 0.0
    epoch_recons_error = 0.0
    epoch_perplexity = 0.0

    itr = 0
    for batch in batch_bar:
        itr+=1
        data, _ = batch
        data = data.to(device)
        batch_size = data.shape[0]

        z = encoder(data)
        # E-step
        if 1:#itr%(2*alternate_every)<alternate_every:
            
            state, logprobs = sample(gfn, z, p=1, rand_prob=rand_prob)

            vq_loss, quantized, perplexity, encodings = vq_vae(inputs=z, enc=state)

            data_recon = decoder(quantized)

            recon_error = F.mse_loss(data_recon, data) / data_variance
            
            dist1 = F.mse_loss(data_recon, data, reduction='none').sum((1,2,3))
            dist2 = F.mse_loss(quantized, z, reduction='none').sum((1,2,3))
            reward = (-dist1-dist2) / steps

            fw_loss = (gfn.logZ.view(1,1) + logprobs.view(batch_size,1) / steps - reward.view(batch_size,1))**2
            fw_loss = fw_loss.mean()
            gfn_opt.zero_grad()
            fw_loss.backward()
            gfn_opt.step()


            # Sleep phase exploration
            if np.random.rand() < sleep_prob:
                # Maximize log-probabilities of randomly sampled trajectory
                logprobs, _ = sleep_step(encoder, gfn, decoder, latent_dict, batch_size=batch_size)
                
                sleep_loss = -10 * logprobs.mean() / steps
                gfn_opt.zero_grad()
                sleep_loss.backward()
                gfn_opt.step()
        
        # M-step 
        else:
            # Greedy encoder - sample from GFlowNet greedily
            if np.random.rand() < greedy_prob:
                state, logprobs = sample(gfn, z, p=-1, rand_prob=0)
            else:
                state, logprobs = sample(gfn, z, p=1, rand_prob=0)

            # Reconstruct image from sampled latent
            vq_loss, quantized, perplexity, encodings = vq_vae(x=z, enc=state, gfn=False)

            data_recon = decoder(quantized)
            # Decoder loss
            recon_error = F.mse_loss(data_recon, data)/data_variance
            
            dist1 = F.mse_loss(data_recon, data, reduction='none').sum((1,2,3))
            dist2 = F.mse_loss(quantized, z, reduction='none').sum((1,2,3))
            reward = (-dist1) / steps

            fw_loss = (gfn.logZ.view(1,1) + logprobs.view(batch_size,1) / steps - reward.view(batch_size,1))**2
            fw_loss = fw_loss.mean()
            loss = recon_error+vq_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        epoch_vq_loss += vq_loss.item()* data.size(0)
        epoch_recons_error += recon_error.item()* data.size(0)
        # epoch_perplexity +=  perplexity.item()* data.size(0)

        batch_bar.set_postfix({'recon Loss': (recon_error.item()), 'gfn loss':fw_loss.item(), 'vq Loss':vq_loss.item()})
    
    train_res_vq_loss.append(epoch_vq_loss/len(training_loader.dataset))
    train_res_recon_error.append(epoch_recons_error/len(training_loader.dataset))
    # train_res_perplexity.append(epoch_perplexity/len(training_loader.dataset))
    # scheduler.step(epoch_recons_error/len(training_loader.dataset))

model_path = os.path.join(model_dir, model_name)
if not os.path.isdir(model_path):
    os.makedirs(model_path)
torch.save(encoder.state_dict(), os.path.join(model_path, f'encoder_.pth'))
torch.save(gfn.state_dict(), os.path.join(model_path, f'gfn_.pth'))
torch.save(vq_vae.state_dict(), os.path.join(model_path, f'vq_vae_.pth'))
torch.save(decoder.state_dict(), os.path.join(model_path, f'decoder_.pth'))


 
epochs_plot = range(1, len(train_res_vq_loss) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs_plot, train_res_vq_loss, label='Training VQ Loss')
plt.title('VQ Loss in Training')
plt.xlabel('Epoch')
plt.ylabel('VQ Loss')
plt.legend()
plt.grid(True)
filename = 'training_vq_loss_plot.png'
save_path = os.path.join(model_path, filename)
plt.savefig(save_path, format='png', dpi=300)
plt.clf()

epochs_plot = range(1, len(train_res_recon_error) + 1)
plt.figure(figsize=(10, 5))
plt.plot(epochs_plot, train_res_recon_error, label='Training Reconstruction Loss')
plt.title('Reconstruction Loss in Training')
plt.xlabel('Epoch')
plt.ylabel('Reconstruction Loss')
plt.legend()
plt.grid(True)
filename = 'training_Reconstruction_loss_plot.png'
save_path = os.path.join(model_path, filename)
plt.savefig(save_path, format='png', dpi=20)
plt.clf()

# epochs_plot = range(1, len(train_res_perplexity) + 1)
# plt.figure(figsize=(10, 5))
# plt.plot(epochs_plot, train_res_perplexity, label='Training Perplexity')
# plt.title('Perplexity in Training')
# plt.xlabel('Epoch')
# plt.ylabel('Perplexity')
# plt.legend()
# plt.grid(True)
# filename = 'training_Perplexity_plot.png'
# save_path = os.path.join(model_path, filename)
# plt.savefig(save_path, format='png', dpi=300)
# plt.clf()


# training_data = {
#     "epochs": epochs_plot,
#     "train_res_vq_loss": train_res_vq_loss,
#     "train_res_recon_error": train_res_recon_error,
#     "train_res_perplexity": train_res_perplexity
# }
# df = pd.DataFrame(training_data)
# filename = 'training_data.xlsx'
# df.to_excel(os.path.join(model_path, filename), index=False)




encoder.eval()
vq_vae.eval()
decoder.eval()

data, _ = next(iter(training_loader))
data = data[:16]
data = data.to(device)
z = encoder(data)

state, logprobs = sample(gfn, z, p=-1, rand_prob=0)
vq_loss, quantized, perplexity, encodings = vq_vae(x=z, enc=state, gfn=False)
data_recon = decoder(quantized)

def plot_images(tensor, model_path, name):
    plt.clf()
    num_rows = 4
    num_cols = 4
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(32, 32))
    axes = axes.flatten()
    for i, ax in enumerate(axes):
        img = (tensor[i]).cpu()
        img = img.detach().numpy()
        img = np.transpose(img, (1, 2, 0))
        img = (img - img.min()) / (img.max() - img.min())
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    filename = name + '.png'
    save_path = os.path.join(model_path, filename)
    plt.savefig(save_path, format='png', dpi=64)
    plt.clf()

# Now call the function with your tensor
plot_images(data, model_path, "original")
plot_images(data_recon, model_path, "generated")

