class MetaAconC(nn.Module):
    r""" ACON activation (activate or not).
    MetaAconC: (p1*x-p2*x) * sigmoid(beta*(p1*x-p2*x)) + p2*x
    """
    def __init__(self, width):
        super().__init__()
        self.fc1 = nn.Conv2d(width, width//16, kernel_size=1, stride=1, bias=False)
        self.fc2 = nn.Conv2d(width//16, width, kernel_size=1, stride=1, bias=False)

        self.p1 = nn.Parameter(torch.randn(1, width, 1, 1))
        self.p2 = nn.Parameter(torch.randn(1, width, 1, 1))

        self.sigmoid = nn.Sigmoid()

    def forward(self, x, **kwargs):
        beta = self.sigmoid(self.fc2(self.fc1(x.mean(dim=2, keepdims=True).mean(dim=3, keepdims=True))))
        return (self.p1 * x - self.p2 * x) * self.sigmoid( beta * (self.p1 * x - self.p2 * x)) + self.p2 * x

class PSP_Acon(nn.Module):
  def __init__(self):
    super(PSP_Acon, self).__init__()
    self.conv1 = nn.Conv2d(3, 24, kernel_size=3, stride=1, padding=0, bias=False) 
    self.bn1 = nn.BatchNorm2d(24)
    self.acon1 = MetaAconC(24)
    self.conv2 = nn.Conv2d(24, 96, kernel_size=3, stride=1, padding=0, bias=False) 
    self.bn2 = nn.BatchNorm2d(96)
    self.acon2 = MetaAconC(96)
    self.conv3 = nn.Conv2d(96, 192, kernel_size=3, stride=1, padding=0, bias=False) 
    self.bn3 = nn.BatchNorm2d(192)
    self.acon3 = MetaAconC(192)
    self.max_pool = nn.MaxPool2d(kernel_size=2) 

    self.avg_pool1_1 = nn.AvgPool2d(kernel_size=2) 
    self.avg_pool1_2 = nn.AvgPool2d(kernel_size=4)
    self.avg_pool1_3 = nn.AvgPool2d(kernel_size=8)
    self.conv_middle = nn.Conv2d(192, 64, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_middle = nn.BatchNorm2d(64)
    self.acon_middle = MetaAconC(64)

    self.conv_final1 = nn.Conv2d(384, 64, kernel_size=3, stride=1, padding=0, bias=False) 
    self.bn_final1 = nn.BatchNorm2d(64)
    self.conv_final2 = nn.Conv2d(64, 24, kernel_size=1, stride=1, padding=0, bias=True) 
    self.linear1 = nn.Linear(2400, 10) 
    self.linear2 = nn.Linear(10, 2)

  def forward(self, x):
    ori_out = F.relu(self.acon1(self.bn1(self.conv1(x))))
    ori_out = F.relu(self.acon2(self.bn2(self.conv2(ori_out))))
    ori_out = F.relu(self.acon3(self.bn3(self.conv3(ori_out))))
    ori_out = self.max_pool(ori_out)

    maxpool_feature = ori_out
    ori_out_1 = F.interpolate(F.relu(self.acon_middle(self.bn_middle(self.conv_middle(self.avg_pool1_1(ori_out))))), maxpool_feature.size()[2:])  #interpolate是上采样操作
    ori_out_2 = F.interpolate(F.relu(self.acon_middle(self.bn_middle(self.conv_middle(self.avg_pool1_2(ori_out))))), maxpool_feature.size()[2:])
    ori_out_3 = F.interpolate(F.relu(self.acon_middle(self.bn_middle(self.conv_middle(self.avg_pool1_3(ori_out))))), maxpool_feature.size()[2:])
    out = [maxpool_feature, ori_out_1, ori_out_2, ori_out_3]
    out = torch.cat(out, 1)

    out = self.conv_final1(out)
    out = self.bn_final1(out)
    out = self.conv_final2(out)
    out = out.view(out.size(0), -1)
    # print(out.shape)
    out = self.linear1(out)
    out = self.linear2(out)

    return out
