import torch
import torch.nn as nn

class TinyVGG(nn.Module):
  """Creates the TinyVGG architecture.

  Replicates the TinyVGG architecture from the CNN explainer website in PyTorch.
  See the original architecture here: https://poloclub.github.io/cnn-explainer/

  Args:
    num_classes: An integer indicating number of output units.
  """
  def __init__(self, num_classes):
      super().__init__()
      self.conv_block_1 = nn.Sequential(
          nn.Conv2d(3, 16, kernel_size=3, stride=1),  
          nn.ReLU(),
          nn.Conv2d(16, 16, kernel_size=3, stride=1),
          nn.ReLU(),
          nn.MaxPool2d(kernel_size=2, stride=2)
      )
      self.conv_block_2 = nn.Sequential(
          nn.Conv2d(16, 16, kernel_size=3),
          nn.ReLU(),
          nn.Conv2d(16, 16, kernel_size=3),
          nn.ReLU(),
          nn.MaxPool2d(2)
      )
      self.classifier = nn.Sequential(
          nn.Flatten(),
          # Where did this in_features shape come from? 
          # It's because each layer of our network compresses and changes the shape of our inputs data.
          nn.Linear(in_features=16 * 5 * 5,
                    out_features=num_classes)
      )

  def forward(self, x: torch.Tensor):
      x = self.conv_block_1(x)
      x = self.conv_block_2(x)
      x = self.classifier(x)
      return x
      # return self.classifier(self.conv_block_2(self.conv_block_1(x))) # <- leverage the benefits of operator fusion