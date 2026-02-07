import torch
checkpoint = torch.load("checkpoints_cvae/20260207_140633/best_model.pt", map_location='cpu')
print("Keys in state_dict:")
for k in list(checkpoint['model_state_dict'].keys())[:20]:
    print(k)
