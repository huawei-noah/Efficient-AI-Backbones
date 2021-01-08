# 2020.06.09-GhostNet definition for pytorch hub
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
dependencies = ['torch']
import torch
from ghostnet_pytorch.ghostnet import ghostnet


state_dict_url = 'https://github.com/huawei-noah/ghostnet/raw/master/ghostnet_pytorch/models/state_dict_73.98.pth'


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
