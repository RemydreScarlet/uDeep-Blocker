import torch
import torch.nn as nn

class TrackerCritic(nn.Module):
    def __init__(self, input_dim=1024):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    def forward(self, x): return self.net(x)

device = torch.device("cpu")
model = nn.Sequential(
    TrackerCritic(),
    nn.Sigmoid()
)

model.load_state_dict(torch.load("tracker_dual_gan_best.pth", map_location=device))
model.eval()

dummy_input = torch.randn(1, 1024)

print("Exporting model to ONNX...")
torch.onnx.export(
    model, 
    dummy_input, 
    "udeep_classifier.onnx", 
    export_params=True,        
    opset_version=14,          
    do_constant_folding=True,  
    input_names=['input'],     
    output_names=['output'],   
    dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
)

print("Successfully saved: udeep_classifier.onnx")