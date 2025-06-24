#!/usr/bin/env python3
"""
Create a simple neural network for demonstration and export to ONNX.
This neural net will take distribution integration results as input and 
predict some classification or regression output.
"""

import torch
import torch.nn as nn
import torch.onnx
import numpy as np

class SimpleNet(nn.Module):
    """
    Simple feedforward neural network that takes distribution integration 
    results and outputs a prediction.
    
    Input: 4 features (summary stats from distribution integrations)
    Output: 2 classes (binary classification)
    """
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.fc1 = nn.Linear(4, 16)
        self.fc2 = nn.Linear(16, 8) 
        self.fc3 = nn.Linear(8, 2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.softmax(self.fc3(x))
        return x

def create_model():
    """Create and return the neural network model."""
    model = SimpleNet()
    
    # Initialize with some reasonable weights
    with torch.no_grad():
        for param in model.parameters():
            if len(param.shape) > 1:
                nn.init.xavier_uniform_(param)
            else:
                nn.init.zeros_(param)
    
    return model

def export_to_onnx(model, filename="neural_net.onnx"):
    """Export the model to ONNX format."""
    # Create dummy input tensor (batch_size=1, features=4)
    dummy_input = torch.randn(1, 4)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        filename,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        }
    )
    print(f"Model exported to {filename}")

def test_model(model):
    """Test the model with sample input."""
    # Sample input: [total_integral, mean_integral, max_integral, std_integral]
    test_input = torch.tensor([[3.34286e+06, 3186.6, 5000.0, 1250.0]], dtype=torch.float32)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        predicted_class = torch.argmax(output, dim=1)
        
    print(f"Test input: {test_input.numpy()}")
    print(f"Model output: {output.numpy()}")
    print(f"Predicted class: {predicted_class.item()}")

if __name__ == "__main__":
    print("Creating simple neural network for distribution analysis...")
    
    # Create model
    model = create_model()
    print(f"Model architecture:\n{model}")
    
    # Test the model
    test_model(model)
    
    # Export to ONNX
    export_to_onnx(model)
    
    print("Neural network creation completed!")