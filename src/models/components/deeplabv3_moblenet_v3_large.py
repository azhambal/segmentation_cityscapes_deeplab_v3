import torch
from torch import nn
from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

class DeeplabV3(nn.Module):

    def __init__(
        self,
        num_classes: int = 19,
    ) -> None:
        super().__init__()

        self.model = deeplabv3_mobilenet_v3_large(num_classes=num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform a single forward pass through the network.

        :param x: The input tensor.
        :return: A tensor of predictions.
        """
        return self.model(x)['out']


if __name__ == "__main__":
    _ = DeeplabV3()
