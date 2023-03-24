
class MaskEncoder(nn.Module):

    def __init__(
        self,
        backbone,
        backbone_out_channels,
        out_channels,
        square_pad=0,
    ):
        super(MaskEncoder, self).__init__()
        self.backbone = backbone
        self.neck = nn.Sequential([
            nn.Conv2d(
                backbone_out_channels, 
                out_channels,
                kernel_size=1,
                bias=False,
            ),
            LayerNorm(out_channels),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            LayerNorm(out_channels),
        ])
        self.square_pad = square_pad

    def preprocess(self, x):
        """Get from InteractiveSegmentEverything."""
        raise NotImplementedError

    def forward(self, x):
        x = self.preprocess(x)
        x = self.net(x)
        x = self.neck(x)
        return x

class LayerNorm(nn.Module):
    """
    D2LayerNorm for BxCxHxW
    """

    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x
