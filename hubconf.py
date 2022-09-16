# 2022.09.16-GhostNet & SNN-MLP definition for pytorch hub
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
dependencies = ['torch']
import torch
from ghostnet_pytorch.ghostnet import ghostnet
from snnmlp_pytorch.models.snn_mlp import SNNMLP


state_dict_url = 'https://github.com/huawei-noah/ghostnet/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth'
state_dict_url_snnmlp_t = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/snnmlp/snnmlp_tiny_81.88.pt'
state_dict_url_snnmlp_s = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/snnmlp/snnmlp_small_83.30.pt'
state_dict_url_snnmlp_b = 'https://github.com/huawei-noah/Efficient-AI-Backbones/releases/download/snnmlp/snnmlp_base_83.59.pt'


def ghostnet_1x(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    GhostNet 1.0x model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = ghostnet(num_classes=1000, width=1.0, dropout=0.2)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model

def snnmlp_t(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    SNN-MLP tiny model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = SNNMLP(num_classes=1000, embed_dim=96, depths=[2, 2, 6, 2], drop_path_rate=0.2)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_t, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model

def snnmlp_s(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    SNN-MLP small model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = SNNMLP(num_classes=1000, embed_dim=96, depths=[2, 2, 18, 2], drop_path_rate=0.3)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_s, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model

def snnmlp_b(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    SNN-MLP base model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = SNNMLP(num_classes=1000, embed_dim=128, depths=[2, 2, 18, 2], drop_path_rate=0.5)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url_snnmlp_b, progress=True)
	  	  model.load_state_dict(state_dict)
	  return model
