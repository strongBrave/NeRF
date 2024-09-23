import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import os
import imageio

class Embedding(nn.Module):
    def __init__(self, in_channels, N_freqs):
        super(Embedding, self).__init__()
        self.in_channels = in_channels
        self.N_freqs = N_freqs
        self.funcs = [torch.sin, torch.cos]
        self.len = in_channels * (len(self.funcs) * N_freqs + 1)

        self.freq_bands = torch.linspace(2 ** 0, 2 ** (self.N_freqs - 1), self.N_freqs)

    def forward(self, x):
        """
        
        Inputs:
            x: [batch_size, self.in_channels]
        Returns:
            embedded_x = [batch_size, self.len]
        """
        
        embedded_x = [x]
        for f in self.freq_bands:
            for funcs in self.funcs:
                embedded_x.append(funcs(f * x))
        embedded_x = torch.cat(embedded_x, -1)
        return embedded_x
    
class NerfModel(nn.Module):
    def __init__(self, n_embbed_coord=63, n_embbed_dir=27, n_hidden=256):
        super(NerfModel, self).__init__()

        self.n_hidden = n_hidden
        self.n_input = n_embbed_coord

        self.pre_block1 = nn.Sequential(
            nn.Linear(n_embbed_coord, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU()
        )

        self.pre_block2 = nn.Sequential(
            nn.Linear(n_hidden + n_embbed_coord, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
            nn.Linear(n_hidden, n_hidden), nn.ReLU(),
        )

        self.sigma_block = nn.Linear(n_hidden, n_hidden + 1)

        self.color_block = nn.Sequential(
            nn.Linear(n_hidden + n_embbed_dir, n_hidden // 2), nn.ReLU(),
            nn.Linear(n_hidden // 2, 3), nn.Sigmoid()
        )
        self.relu = nn.ReLU()

    def forward(self, o, d):
        """
        Inputs:
            o: [batch_size, n_embbed_coord]  embbed origin
            d: [batch_size, n_embbed_dir] embbed ray direction
        outputs
            sigma: [batch_size, ] density
            colorl: [batch_size, 3] color

        """

        inter_x1 = self.pre_block1(o) # [batch_size, n_hidden]
        inter_x2 = self.pre_block2(torch.cat([inter_x1, o], -1))  #[batch_size, n_hidden]
        inter_x3 = self.sigma_block(inter_x2) # [batch_size, n_hidden + 1]

        features, sigma = inter_x3[:, :-1], self.relu(inter_x3[:, -1])

        color = self.color_block(torch.cat([features, d], -1)) # [batch_size, 3]

        return sigma, color

def render_rays(nerf_model, embedding_model, ray_origins, ray_directions, hn=0, hf=0.5, bins=192):
    """
    Inputs:
        nerf_model: nerf neural model
        embedding_model: position embedding model
        ray_origins: [batch_size, 3]
        ray_directions: [batch_size, 3]
        hn: the distance to the near plane
        hf: the distance to the far plane
        bins: the number of sampled points along each ray
    Returns:
        px_values: [batch_size, 3] the predicted pixel values 
    """
    device = ray_origins.device


    t = torch.linspace(hn, hf, bins, device=device).expand(ray_origins.shape[0], bins) # [batch_size, bins]
    # Each batch represents a ray
    # Perturb sampleing along each ray
    mid = (t[:, 1:] + t[:, :-1]) / 2
    lower = torch.cat((t[:, :1], mid), dim=-1) # [batch_size, bins]
    upper = torch.cat((mid, t[:, -1:]), dim=-1) # [batch_size, bins]
    u = torch.rand(t.shape).to(device)
    t = lower + (upper - lower) * u # [batch_size, bins]
    delta = torch.cat((t[:, 1:] - t[:, :-1], torch.tensor([1e10], device=device).expand(t.shape[0], 1)), dim=-1) # [batch_size, bins]

    # Computer the 3d coordinates along each ray
    x = ray_origins.unsqueeze(1) + t.unsqueeze(2) * ray_directions.unsqueeze(1) # [batch_size, bins, 3]
    ray_directions = ray_directions.unsqueeze(1).expand(-1, bins, 3)
    embedding_coord_model, embedding_dir_model = embedding_model[0], embedding_model[1]
    embedd_x = embedding_coord_model(x.reshape(-1, 3)).to(device) # [batch_size, 27]
    embedd_directions = embedding_dir_model(ray_directions.reshape(-1, 3)).to(device) # [batch_size, 63]

    sigma, color = nerf_model(embedd_x, embedd_directions) # [batch_size, 1] [batch_size, 3]

    color = color.reshape(x.shape) # [batch_size, bins, 3]
    sigma = sigma.reshape(x.shape[:-1]) # [batch_size, bins]

    alpha = 1 - torch.exp(- sigma * delta)
    alpha_shifted = torch.cat((torch.ones(alpha.shape[0], 1, device=device), 1 - alpha + 1e-10), dim=-1) # [batch_size, bins]
    weights = alpha * torch.cumprod(alpha_shifted, dim=1)[:, :-1] # [vatch_size, bins]
    # Compute the pixel values as a weighted sum of colors along each ray
    c = (weights.unsqueeze(-1) * color).sum(dim=1) # [batch_size, 3]
    weight_sum = weights.sum(-1)

    return c + 1 - weight_sum.unsqueeze(-1)

    

@torch.no_grad()
def test(nerf_model, hn, hf, dataset, current_epoch, chunk_size=10, img_index=0, bins=192, H=400, W=400):
    print(f"In epoch {current_epoch + 1}, Generating the {img_index + 1}th image.")
    ray_origins = dataset[img_index * H * W: (img_index + 1) * H * W, :3]
    ray_directions = dataset[img_index * H * W: (img_index + 1) * H * W, 3:6]

    data = [] # list of predicted pixel values
    for i in range(int(np.ceil(H / chunk_size))):  # iterate over chunks
        # Get chunk of rays
        ray_origins_ = ray_origins[W * i * chunk_size: W * (i + 1) \
                                   * chunk_size, :].to(device)
        ray_directions_ = ray_directions[W * i * chunk_size: W * (i + 1) \
                                         * chunk_size, :].to(device)
        pred_px_values = render_rays(nerf_model, embedding_model, ray_origins_, ray_directions_, hn=hn, hf=hf, bins=bins) # [chunk_size, 3]
        data.append(pred_px_values)
    img = torch.cat(data, dim=0).data.cpu().numpy().reshape(H, W, 3)


    # create save path
    save_dir = f"novel_views/{current_epoch + 1}"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    plt.figure()
    plt.imshow(img)
    plt.savefig(f"novel_views/{current_epoch + 1}/img_{img_index}.png", bbox_inches='tight')
    plt.close()


def train(nerf_model, embedding_model, criterion, optimizer, scheduler, dataloader, device='cpu', hn=0, hf=1, epochs=5,
          bins=192, H=400, W=400, checkpoint_dir="checkpoints"):
    train_loss = []
    best_loss = float("inf") 

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)

    total_len = len(dataloader)
    for e in tqdm(range(epochs)):
        epoch_loss = 0.0
        for i, batch in enumerate(dataloader):
            ray_origins = batch[:, :3].to(device)
            ray_directions = batch[:, 3:6].to(device)
            real_px_values = batch[:, 6:].to(device)

            pred_px_values = render_rays(nerf_model, embedding_model, ray_origins, ray_directions, hn=hn, hf=hf, bins=bins)
            # print(pred_px_values)
            l = ((pred_px_values - real_px_values) ** 2).sum()

            optimizer.zero_grad()
            l.backward()
            optimizer.step()
            train_loss.append(l.item())
            epoch_loss += l.item()
        
            print("Epoch: [%d/%d], Batch: [%d/%d], loss: %.4f" % (e+1, epochs, i+1, total_len, l.item()))

        epoch_loss = epoch_loss / float(total_len)
        # Save model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            checkpoint_path = os.path.join(checkpoint_dir, f"nerf_best_model_epoch_{e+1}.pth")
            print(f"Saving checkpoint with loss: {best_loss:.4f} at epoch {e+1}.")
            torch.save({
                "epoch": e + 1,
                "model_state_dict": nerf_model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": best_loss
            }, checkpoint_path)

        scheduler.step()

        for img_index in range(20):
            test(nerf_model, hn, hf, test_dataset, e, img_index=img_index, bins=bins, H=H, W=W)
    return train_loss

# load model
def load_model(model, optimizer, scheduler, load_path):
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('加载 epoch {} 成功！'.format(start_epoch))
    return model, optimizer, scheduler, start_epoch

# Create gif using a series of consecutive photos generated by nerf model
def create_gif(image_folder, output_gif, duration=0.1):
    images = []
    for img_index in range(20):
        img_path = os.path.join(image_folder, f'img_{img_index}.png')
        images.append(imageio.imread(img_path))
    
    # Save as gif
    imageio.mimsave(output_gif, images, duration=duration, loop=0)


# Single test
def single_test(model, dataset):
    test(model, hn=2, hf=6, dataset=dataset, current_epoch=1, img_index=2, chunk_size=10, bins=192, H=400, W=400)

if __name__ == "__main__":
    device = 'cuda'

    L = [10, 4]
    L_after = [63, 27]

    train_dataset = torch.from_numpy(np.load("data/training_data.pkl", allow_pickle=True))
    test_dataset = torch.from_numpy(np.load("data/testing_data.pkl", allow_pickle=True))

    model = NerfModel(L_after[0], L_after[1], 256).to(device)
    embedding_model = [Embedding(3, l).to(device) for l in L]
    criterion = nn.MSELoss(reduction='sum')

    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[2, 4, 8], gamma=0.5)
    dataloader = DataLoader(train_dataset, batch_size=1024, shuffle=True)

    # load_path = "checkpoints/"
    # load_model(model, optimizer, scheduler, )

    train(model, embedding_model, criterion, optimizer, scheduler, dataloader, device, epochs=10,
          hn=2, hf=6, bins=192, H=400, W=400)

    # create_gif("./novel_views/6", "nerf_animation.gif", duration=0.1)
