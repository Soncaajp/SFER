from torch import nn
import torch
import torch.nn.functional as F
from torch.autograd import Variable

import math
from torch.nn import Parameter


class ConvBlock(nn.Module):
    def __init__(self, inp, oup, k, s, p, dw=False, linear=False, num_group = 1):
        super(ConvBlock, self).__init__()
        self.linear = linear
        if dw:
            self.conv = nn.Conv2d(inp, oup, k, s, p, groups=num_group, bias=False)
        else:
            self.conv = nn.Conv2d(inp, oup, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(oup)
        if not linear:
            self.prelu = nn.PReLU(oup)
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.linear:
            return x
        else:
            return self.prelu(x)

class DResidual(nn.Module):
    def __init__(self, inp, oup, k = (3,3), s = (2,2), p = (1,1), dw=False, linear=False, num_group = 1):
        super(DResidual, self).__init__()

        self.conv = ConvBlock(inp, oup, k = (1,1), s = (1,1), p = (0,0))
        self.conv_dw = ConvBlock(oup, oup, k, s, p, dw = True, num_group = num_group)
        self.linear = ConvBlock(oup, oup, k = (1,1), s = (1,1), p = (0,0), dw = True, linear = True)

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_dw(x)
        return self.linear(x)

class Residual(nn.Module):
    def __init__(self, inp, oup, k=(3,3), s=(1,1), p=(1,1), num_block = 1, num_group = 1):
        super(Residual, self).__init__()
        self.num_block = num_block
        self.dres = DResidual(inp, oup, k, s, p, num_group = num_group)
    def forward(self, x):

    	identity = x
    	for i in range(self.num_block):
    		shortcut = identity
    		conv = self.dres(identity)
    		identity = conv + shortcut
    	return identity

class MobileFacenet(nn.Module):
    def __init__(self):
        super(MobileFacenet, self).__init__()

        self.conv1 = ConvBlock(3, 64, 3, 2, 1)
        self.res1 = Residual(64, 64, 3, 1, 1, 2)
        self.dres1 = DResidual(64,64,3,2,1)#, num_group = 128)
        self.res2 = Residual(64,64,3,1,1, 4)#, num_group = 128)
        self.dres2 = DResidual(64,128,3,2,1)#,num_group = 256)
        self.res3 = Residual(128,128,3,1,1,8)#, num_group = 256)
        self.dres3 = DResidual(128,128,3,2,1)#, num_group = 512)
        self.res4 =Residual(128,128,3,1,1,4)#, num_group = 256)
        self.conv2 = ConvBlock(128, 512, 1,1,0)
        self.conv3 = ConvBlock(512, 512, 1,1,0)
        self.conv4 = ConvBlock(512, 128, 1,1,0)
        self.dense1 = nn.Linear(6272,7)
        # self.dense2 = nn.Linear(512,7)
        self.smax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.res1(x)
        x = self.dres1(x)
        x = self.res2(x)
        x = self.dres2(x)
        x = self.res3(x)
        x =self.dres3(x)
        x =self.res4(x)
        x =self.conv2(x)
        x =self.conv3(x)
        x =self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.dense1(x)
        # x = self.dense2(x)
        x = self.smax(x)
        
        return x

if __name__ == "__main__":
    input = Variable(torch.FloatTensor(1, 3, 112, 112))
    net = MobileFacenet()
    torch.onnx.export(net,               # model being run
                  input,                         # model input (or a tuple for multiple inputs)
                  "pytorch_model_new.onnx",   # where to save the model (can be a file or file-like object)
                  export_params=True,        # store the trained parameter weights inside the model file
                  opset_version=10,          # the ONNX version to export the model to
                  do_constant_folding=True,  # whether to execute constant folding for optimization
                  input_names = ['input'],   # the model's input names
                  output_names = ['output'], # the model's output names
                  dynamic_axes={'input' : {0 : 'batch_size'},    # variable lenght axes
                                'output' : {0 : 'batch_size'}})
    print(net)
    x = net(input)
    print(x.shape)
    print(x)