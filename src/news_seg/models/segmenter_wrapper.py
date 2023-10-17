"""Wrapper for Segmenter to be compatible with training process"""
import torch
from torch import nn
from torchvision.transforms.functional import normalize

from segmenter.segm.model.segmenter import Segmenter


class SegmenterWrapper(nn.Module):
    """
    CNN Encoder Class, corresponding to the first resnet50 layers.
    """

    def __init__(self, segmenter: Segmenter, in_channels: int):
        super().__init__()
        self.segmenter = segmenter
        # initialize normalization
        # pylint: disable=duplicate-code
        self.register_buffer("means", torch.tensor([0] * in_channels))
        self.register_buffer("stds", torch.tensor([1] * in_channels))
        self.normalize = normalize

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        Encoder forward
        :param inputs: input tensor
        :return: dictionary with result and scip-connections
        """
        result = self.normalize(inputs, self.means, self.stds)
        result = self.segmenter(result)

        return result
