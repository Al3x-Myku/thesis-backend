
import torch, sys
from src.core import YAMLConfig
try: from thop import profile
except: sys.exit(1)
cfg = YAMLConfig("configs/dfine/dfine_hgnetv2_l_coco.yml")
model = cfg.model.deploy().cuda()
try:
    macs, params = profile(model, inputs=(torch.randn(1, 3, 640, 640).cuda(), ), verbose=False)
    print(f"RESULTS_PARAMS:{params / 1e6:.2f}\nRESULTS_FLOPS:{(macs * 2) / 1e9:.2f}")
except: print("Failed")
