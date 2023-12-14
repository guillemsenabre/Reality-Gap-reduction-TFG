import torch
import torch.nn as nn

from arm_pkg.arm_pkg.drl.ddpg import DDPGAgent

model_path = "./models/"
saved_model_filename = "your_ddpg_model.pth"  


ddpg_model = DDPGAgent()
ddpg_model.load_state_dict(torch.load(model_path + saved_model_filename))
ddpg_model.eval()

for param in ddpg_model.layer1.parameters():
    param.requires_grad = False

for param in ddpg_model.layer2.parameters():
    param.requires_grad = False
