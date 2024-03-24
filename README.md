# AristotleGPT
A simplified decoder-only Transformer model pre-trained on Aristotelian Philosophy

To load the model, use the following code:
```
PATH = "aristotlegpt14M.pth"
device = torch.device('cpu')
model = functions.AristotleGPTModel()
model.load_state_dict(torch.load(PATH, map_location=device))
```
