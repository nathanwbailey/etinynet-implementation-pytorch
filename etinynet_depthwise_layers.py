import torch
import torch.nn.functional as F

class LB(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int, bias: bool = True) -> None:
        super().__init__()
        self.depthwise_conv_layer_a = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=padding, stride=stride, bias=bias)
        self.batch_normalization_a = torch.nn.BatchNorm2d(num_features=in_channels)
        self.pointwise_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        self.batch_normalization_point = torch.nn.BatchNorm2d(num_features=out_channels)
        self.depthwise_conv_layer_b = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels, padding="same", stride=1, bias=bias)
        self.batch_normalization_b = torch.nn.BatchNorm2d(num_features=out_channels)

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        depthwise_result = self.batch_normalization_a(self.depthwise_conv_layer_a(input_tensor))
        pointwise_result = F.relu(self.batch_normalization_point(self.pointwise_layer(depthwise_result)))
        return F.relu(self.batch_normalization_b(self.depthwise_conv_layer_b(pointwise_result)))


class DLB(torch.nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1, downsample: torch.nn.Module | None = None, bias: bool = True) -> None:
        super().__init__()
        self.depthwise_conv_layer_a = torch.nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, groups=in_channels, padding=padding, stride=stride, bias=bias)
        self.batch_normalization_a = torch.nn.BatchNorm2d(num_features=in_channels)
        self.pointwise_layer = torch.nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, bias=bias)
        self.batch_normalization_point = torch.nn.BatchNorm2d(num_features=out_channels)
        self.depthwise_conv_layer_b = torch.nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, groups=out_channels, padding="same", stride=1, bias=bias)
        self.batch_normalization_b = torch.nn.BatchNorm2d(num_features=out_channels)
        self.downsample = downsample

    def forward(self, input_tensor: torch.Tensor) -> torch.Tensor:
        residual = input_tensor
        depthwise_result = self.batch_normalization_a(self.depthwise_conv_layer_a(input_tensor))
        pointwise_result = self.batch_normalization_point(self.pointwise_layer(depthwise_result))
        if self.downsample:
            residual = self.downsample(input_tensor)
        pointwise_result = pointwise_result + residual
        pointwise_result = F.relu(pointwise_result)
        final_depthwise_result = self.batch_normalization_b(self.depthwise_conv_layer_b(pointwise_result))
        final_depthwise_result = F.relu(final_depthwise_result + pointwise_result + residual)
        return final_depthwise_result

# downsample = torch.nn.Sequential(
#     torch.nn.Conv2d(32,64,kernel_size=1,stride=2),
#     torch.nn.BatchNorm2d(64)
# )

# dlb = DLB(32,64,kernel_size=3, padding=1, stride=2, downsample=downsample)

# lb = LB(32,64,kernel_size=3, padding=1, stride=2)

# test_tensor = torch.empty(32,32,224,224)

# print(lb(test_tensor).size())
# print(dlb(test_tensor).size())
