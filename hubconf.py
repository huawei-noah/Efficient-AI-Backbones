dependencies = ['torch']
import torch
from pytorch.ghostnet import ghostnet


state_dict_url = 'https://github.com/huawei-noah/ghostnet/blob/master/pytorch/models/state_dict_93.98.pth'


def ghostnet_1x(pretrained=False, **kwargs):
	  """ # This docstring shows up in hub.help()
    GhostNet 1.0x model
    pretrained (bool): kwargs, load pretrained weights into the model
    """
	  model = ghostnet(num_classes=1000, width=1.0, dropout=0.2)
	  if pretrained:
	  	  state_dict = torch.hub.load_state_dict_from_url(state_dict_url, progress=True)
	  	  model.load_state_dict()
