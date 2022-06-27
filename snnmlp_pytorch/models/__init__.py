# 2022.06.27-Changed for building SNN-MLP
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
from .snn_mlp import SNNMLP

def build_model(config):
    model = SNNMLP(img_size=config.DATA.IMG_SIZE,
                            patch_size=config.MODEL.SNNMLP.PATCH_SIZE,
                            in_chans=config.MODEL.SNNMLP.IN_CHANS,
                            num_classes=config.MODEL.NUM_CLASSES,
                            embed_dim=config.MODEL.SNNMLP.EMBED_DIM,
                            depths=config.MODEL.SNNMLP.DEPTHS,
                            mlp_ratio=config.MODEL.SNNMLP.MLP_RATIO,
                            drop_rate=config.MODEL.DROP_RATE,
                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                            patch_norm=config.MODEL.SNNMLP.PATCH_NORM,
                            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                            lif=config.LIF, lif_fix_tau=config.LIF_FIX_TAU, lif_fix_vth=config.LIF_FIX_VTH,
                            lif_init_tau=config.LIF_INIT_TAU, lif_init_vth=config.LIF_INIT_VTH)
    return model