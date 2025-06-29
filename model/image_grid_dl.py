import torch
import torchvision as tv
import matplotlib.pyplot as plt
import random
import einops

torch.manual_seed(47)
random.seed(47)

class Combine(torch.utils.data.Dataset):
  def __init__(self, fullset=None, train=True):
    super().__init__()
    self.tf = tv.transforms.Compose([
      tv.transforms.ToTensor(),
      tv.transforms.Normalize((0.1307,), (0.3081,))
    ])
    self.tk = {str(i): i for i in range(10)}
    if fullset is not None:
      self.ds = fullset
      self.data = fullset.data
      self.targets = fullset.targets
    self.ti = tv.transforms.ToPILImage()
    self.ln = len(self.data)

  def __len__(self):
    return self.ln

  def __getitem__(self, idx):
    idxs = random.sample(range(self.ln), 4)
    imgs = [self.data[i] for i in idxs]
    labels = [self.targets[i].item() for i in idxs]
    tnsrs = [self.tf(self.ti(img)) for img in imgs]
    stack = torch.stack(tnsrs, dim=0).squeeze()
    combo = einops.rearrange(stack, '(h w) ph pw -> (h ph) (w pw)', h=2, w=2, ph=28, pw=28)
    patch = einops.rearrange(combo, '(h ph) (w pw) -> (h w) ph pw', ph=14, pw=14)
    return combo, patch, torch.tensor(labels)
  
if __name__ == "__main__":

  ds = Combine()
  cmb, pch, lbl = ds[0]
  print('lbl', lbl) # [2, 4, 9, 7]
  print('cmb', cmb.shape)
  print('pch', pch.shape)
  plt.imshow(ds.ti(cmb)); plt.show()
  plt.imshow(ds.ti(einops.rearrange(pch, 'p h w -> h (p w)'))); plt.show()

  pass