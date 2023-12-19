import torch
import torch.nn as nn

from arm_pkg.arm_pkg.drl.ddpg import DDPGAgent

model_path = ""
ddpg_model = DDPGAgent()

train_or_pretrained = input("Do you want to 'train' or use a 'pretrained' model?")

if train_or_pretrained == "pretrained":
    print(f"using pretrained model located in {model_path} ...")

    ddpg_model.load_state_dict(torch.load(model_path))
    ddpg_model.eval()

    for param in ddpg_model.layer1.parameters():
        param.requires_grad = False

    for param in ddpg_model.layer2.parameters():
        param.requires_grad = False

elif train_or_pretrained == "train":
    print(f"training ddpg model from scratch...")
    