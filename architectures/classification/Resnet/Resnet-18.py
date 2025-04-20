import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# 1. Basic Residual Block
# ----------------------------
class BasicBlock(nn.Module):
    expansion = 1  # Used to compute output channels in the ResNet class

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        """
        A BasicBlock has two 3x3 convolutions and a skip connection.
        If input and output dimensions differ, a downsample layer is applied.
        """
        super(BasicBlock, self).__init__()
        
        # First convolution layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolution layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample  # To match dimensions if needed
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = x  # Save input for skip connection

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample:
            identity = self.downsample(x)  # Match input shape to output shape

        out += identity  # Add skip connection
        out = self.relu(out)
        return out

# ----------------------------
# 2. ResNet-18 Model
# ----------------------------
class ResNet18(nn.Module):
    def __init__(self, num_classes=1000):
        """
        ResNet-18 has 4 stages with [2, 2, 2, 2] BasicBlocks.
        """
        super(ResNet18, self).__init__()
        self.in_channels = 64  # Initial channel count after first conv

        # Initial Conv Layer (stem)
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # Four residual stages
        self.layer1 = self._make_layer(BasicBlock, 64,  2)
        self.layer2 = self._make_layer(BasicBlock, 128, 2, stride=2)
        self.layer3 = self._make_layer(BasicBlock, 256, 2, stride=2)
        self.layer4 = self._make_layer(BasicBlock, 512, 2, stride=2)

        # Global Average Pool and FC layer
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # Output size (1x1)
        self.fc = nn.Linear(512 * BasicBlock.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        """
        Creates a stage with multiple residual blocks.
        """
        downsample = None

        if stride != 1 or self.in_channels != out_channels * block.expansion:
            # Use 1x1 conv to match dimensions when needed
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * block.expansion)
            )

        layers = []
        # First block may have downsampling
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion

        # Remaining blocks (no downsampling)
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)  # 64
        x = self.layer2(x)  # 128
        x = self.layer3(x)  # 256
        x = self.layer4(x)  # 512

        x = self.avgpool(x)  # (B, 512, 1, 1)
        x = torch.flatten(x, 1)  # (B, 512)
        x = self.fc(x)  # Final output logits
        return x

# ----------------------------
# 3. Test the Model
# ----------------------------
if __name__ == "__main__":
    model = ResNet18(num_classes=10)
    print(model)

    x = torch.randn(2, 3, 224, 224)  # Dummy input
    out = model(x)
    print("Output shape:", out.shape)  # (2, 10)
