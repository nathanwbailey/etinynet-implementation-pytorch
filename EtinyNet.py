import torch
from etinynet_depthwise_layers import LB
from etinynet_depthwise_layers import DLB
import pytorch_model_summary as pms

class EtinyNet(torch.nn.Module):
    def __init__(self, block_info: list[dict], *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.starting_conv_layer = torch.nn.Sequential(
            torch.nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.BatchNorm2d(32),
            torch.nn.ReLU()
        )
        self.global_pool = torch.nn.AvgPool2d(kernel_size=7)
        self.fully_connected = torch.nn.Linear(block_info[-1]["layer_values"][-1]["out_channels"], 200)

        inter_block_list = []
        for block_i in block_info:
            block_section = []
            for idx, layer_value in enumerate(block_i["layer_values"]):
                if idx != 0:
                    stride=1
                    padding="same"
                else:
                    padding=1
                    stride=2
                block = self._make_layer(in_channels=layer_value["in_channels"], out_channels=layer_value['out_channels'], kernel_size=3, padding=padding, stride=stride, layer_type=block_i['block_type'])
                block_section.append(block)
            inter_block_list.append(torch.nn.Sequential(*block_section))
        
        self.blocks = torch.nn.Sequential(*inter_block_list)

        self.layers = torch.nn.Sequential(
            self.starting_conv_layer,
            self.blocks,
            self.global_pool,
            torch.nn.Flatten(),
            torch.nn.Dropout(p=0.3),
            self.fully_connected
        )

    def _make_layer(self, in_channels: int, out_channels: int, kernel_size: int, padding: int, stride: int, layer_type: str):
        if layer_type == "lb":
            layer = LB(in_channels, out_channels, kernel_size, padding, stride)
        else:
            downsample_block = None
            if stride != 1 or in_channels != out_channels:
                downsample_block = torch.nn.Sequential(
                    torch.nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                    torch.nn.BatchNorm2d(out_channels)
                )
            layer = DLB(in_channels, out_channels, kernel_size, padding, stride, downsample=downsample_block)
        return layer
    
    def forward(self, input_tensor):
        return self.layers(input_tensor)


etinynet_block_info = [
    {
        "block_type": "lb",
        "layer_values": [{"in_channels": 32, "out_channels": 32} for _ in range(4)]
    },
    {
        "block_type": "lb",
        "layer_values": [{"in_channels": 32, "out_channels": 128}] + [{"in_channels": 128, "out_channels": 128} for _ in range(3)]
    },
    # {
    #     "block_type": "dlb",
    #     "layer_values": [{"in_channels": 128, "out_channels": 192}] + [{"in_channels": 192, "out_channels": 192} for _ in range(2)]
    # },
    {
        "block_type": "dlb",
        "layer_values": [{"in_channels": 128, "out_channels": 256}, {"in_channels": 256, "out_channels": 256}, {"in_channels": 256, "out_channels": 512}]
    }
]

# network = EtinyNet(block_info=etinynet_block_info)
# x = filter(lambda param: param.requires_grad, network.parameters())
# print(x)
# print(type(network.parameters()))

# print(network)
# pms.summary(network, torch.zeros((1, 3, 224,224)), show_input=False, print_summary=True, max_depth=5, show_parent_layers=True)