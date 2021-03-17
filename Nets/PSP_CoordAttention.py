class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        
        n,c,h,w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y) 
        
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class PSP_CoordAttention(nn.Module):
  def __init__(self):
    super(PSP_CoordAttention, self).__init__()
    self.conv1 = nn.Conv2d(3, 6, kernel_size=3, stride=1, padding=0, bias=False) 
    self.bn1 = nn.BatchNorm2d(6)
    self.coord_att_first = CoordAtt(6, 6) 
    self.conv2 = nn.Conv2d(6, 24, kernel_size=3, stride=1, padding=0, bias=False) 
    self.bn2 = nn.BatchNorm2d(24)
    self.conv3 = nn.Conv2d(24, 48, kernel_size=3, stride=1, padding=0, bias=False) 
    self.bn3 = nn.BatchNorm2d(48)
    self.max_pool = nn.MaxPool2d(kernel_size=2) 

    self.coord_att_mid = CoordAtt(48, 48)

    self.avg_pool1_1 = nn.AvgPool2d(kernel_size=2) 
    self.avg_pool1_2 = nn.AvgPool2d(kernel_size=4)
    self.avg_pool1_3 = nn.AvgPool2d(kernel_size=8)
    self.conv_middle = nn.Conv2d(48, 16, kernel_size=1, stride=1, padding=0, bias=False)
    self.bn_middle = nn.BatchNorm2d(16)

    self.coord_att_last = CoordAtt(96, 96)
    self.conv_final1 = nn.Conv2d(96, 16, kernel_size=3, stride=1, padding=0, bias=False) 
    self.bn_final1 = nn.BatchNorm2d(16)
    self.conv_final2 = nn.Conv2d(16, 6, kernel_size=1, stride=1, padding=0, bias=True) 
    self.linear1 = nn.Linear(600, 10) 
    self.linear2 = nn.Linear(10, 2)

  def forward(self, x):
    ori_out = F.relu(self.bn1(self.conv1(x)))
    ori_out = self.coord_att_first(ori_out)
    ori_out = F.relu(self.bn2(self.conv2(ori_out)))
    ori_out = F.relu(self.bn3(self.conv3(ori_out)))
    ori_out = self.max_pool(ori_out)

    ori_out = self.coord_att_mid(ori_out)

    maxpool_feature = ori_out
    ori_out_1 = F.interpolate(F.relu(self.bn_middle(self.conv_middle(self.avg_pool1_1(ori_out)))), maxpool_feature.size()[2:])  #interpolate是上采样操作
    ori_out_2 = F.interpolate(F.relu(self.bn_middle(self.conv_middle(self.avg_pool1_2(ori_out)))), maxpool_feature.size()[2:])
    ori_out_3 = F.interpolate(F.relu(self.bn_middle(self.conv_middle(self.avg_pool1_3(ori_out)))), maxpool_feature.size()[2:])
    out = [maxpool_feature, ori_out_1, ori_out_2, ori_out_3]
    out = torch.cat(out, 1)

    out = self.coord_att_last(out)
    out = self.conv_final1(out)
    out = self.bn_final1(out)
    out = self.conv_final2(out)
    out = out.view(out.size(0), -1)
    out = self.linear1(out)
    out = self.linear2(out)

    return out
