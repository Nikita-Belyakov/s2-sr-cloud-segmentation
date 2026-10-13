import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

from .DenoisingNAFNet import ConditionalNAFNet
from .irsde_utils import IRSDE
from .base_inference import BaseInferenceModel
  
    
class EDiffSR(BaseInferenceModel):
    def __init__(self, device : str, 
                 scale: int = 4, 
                 img_channel: int = 4, 
                 model_path: str | None = None):
        self.device = device
        self.scale = scale
        self.img_channel = img_channel
        
        self.model = ConditionalNAFNet(
            img_channel = self.img_channel,
            width=64, enc_blk_nums=[14, 1, 1, 1],
            middle_blk_num=1, dec_blk_nums=[1, 1, 1, 1]
        )#.to(self.device)
        self.model = nn.DataParallel(self.model)
        
        self.sde = IRSDE(max_sigma=50, T=100, schedule='cosine', 
                         eps=0.005, device=self.device, model=self.model)
        if model_path:
            self.load_weights(model_path)
            print(f'Model weights loaded from {model_path}')
        else:
            print('Model weights NOT loaded')
        
    def load_weights(self, model_path, strict=True):
        if isinstance(self.model, nn.DataParallel):
            self.model = self.model.module
        load_net = torch.load(model_path)
        load_net_clean = OrderedDict()  # remove unnecessary 'module.'
        for k, v in load_net.items():
            if k.startswith("module."):
                load_net_clean[k[7:]] = v
            else:
                load_net_clean[k] = v

        self.model.load_state_dict(load_net_clean, strict=strict)
        self.model.eval()
        self.model.to(self.device)    
    
    def predict_batch(self, lr_batch):
#         lr_batch = F.interpolate(
#             lr_batch, scale_factor=self.scale, mode='bicubic'
#         ).to(self.device)
        lr_batch.to(self.device)
        noisy_state = self.sde.noise_state(lr_batch).to(self.device)
        self.sde.set_mu(lr_batch) # sde.mu = LQ
        with torch.no_grad():
            sr_batch = self.sde.reverse_sde(noisy_state)
            
        return sr_batch