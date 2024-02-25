# class Network(torch.nn.Module):
#     def __init__(self, *args, **kwargs) -> None:
#         super().__init__(*args, **kwargs)
#         self.conv = torch.nn.Conv2d(32,64,3)
    
#     def forward(self, x):
#         print(x.size())
#         x1 = self.conv(x)
#         print(x.size())
#         print(x1.size())
#         return x1

# network = Network()

# network(torch.empty(32,32,224,224))