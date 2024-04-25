from src.config import *
import numpy as np
import numpy
from scipy.signal import savgol_filter
from sklearn.metrics import mean_squared_error
import pandas as pd

skip=False
def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv1d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
        nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm1d(out_channels),
        nn.GELU(),
        nn.Conv1d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, stride=1,padding=1),
        nn.BatchNorm1d(out_channels),
        nn.GELU()
    )

class Unet1d(nn.Module):
    def __init__(self, n_feature, hidden_dim, n_output):
        super(Unet1d, self).__init__()

        self.n_feature = n_feature
        self.hidden_dim = hidden_dim
        self.n_output = n_output

        self.dconv_down1 = double_conv(n_feature,hidden_dim)
        self.dconv_down2 = double_conv(hidden_dim,hidden_dim*2)
        self.dconv_down3 = double_conv(hidden_dim*2,hidden_dim*4)
        self.dconv_down4 = double_conv(hidden_dim*4,hidden_dim*8)
        self.dconv_down5 = double_conv(hidden_dim*8,hidden_dim*16)

        self.downsample = nn.MaxPool1d(kernel_size=2, stride=2)

        self.upsample = nn.Upsample(scale_factor=2, mode='linear', align_corners=True)
        if skip:
            self.dconv_up5 = double_conv(hidden_dim*(16+8),hidden_dim*8)
            self.dconv_up4 = double_conv(hidden_dim*(8+4),hidden_dim*4)
            self.dconv_up3 = double_conv(hidden_dim*(4+2),hidden_dim*2)
            self.dconv_up2 = double_conv(hidden_dim*(2+1),hidden_dim*1)
        else:
            self.dconv_up5 = double_conv(hidden_dim*(16),hidden_dim*8)
            self.dconv_up4 = double_conv(hidden_dim*(8),hidden_dim*4)
            self.dconv_up3 = double_conv(hidden_dim*(4),hidden_dim*2)
            self.dconv_up2 = double_conv(hidden_dim*(2),hidden_dim*1)

        self.dropout = nn.Dropout(0.5)

        self.lastlayer = nn.Conv1d(in_channels=hidden_dim, out_channels=2, kernel_size=1, stride=1,padding=0)

        self.flatten = Flatten()
        

    #change this to reshape the input! reshape hidenlayer 
    def forward(self, x):
        conv1 = self.dconv_down1(x)
        
        x = self.downsample(conv1)
        conv2 = self.dconv_down2(x)

        x = self.downsample(conv2)
        conv3 = self.dconv_down3(x)

        x = self.downsample(conv3)
        conv4 = self.dconv_down4(x)

        x = self.downsample(conv4)
        x = self.dconv_down5(x)
        #x = self.dropout(x)

        x = self.upsample(x)
        if skip:
            x = torch.cat([x,conv4], dim=1)
        x = self.dconv_up5(x)

        x = self.upsample(x)
        if skip:
            x = torch.cat([x,conv3], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample(x)
        if skip:
            x = torch.cat([x,conv2], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        if skip:
            x = torch.cat([x,conv1], dim=1)
        x = self.dconv_up2(x)

        x = self.lastlayer(x)
        x = self.flatten(x)


        return x


old = False
blind = 1
if blind ==0:
    inputcsv="./raw_data/traininput.csv"
    labelcsv="./raw_data/trainoutput.csv"
elif blind ==1:
    inputcsv="./raw_data/blindtestinput.csv"
    labelcsv="./raw_data/blindtestoutput.csv"

modelpath="./results/0.1model.pt"
peaknum=0
ndiff=0
maxtest=5000

from numpy import genfromtxt

features=genfromtxt(inputcsv, delimiter=',')
labels=genfromtxt(labelcsv, delimiter=',')

#need to add minibatch
if len(features)>maxtest:
    labels=labels[1+50000:maxtest+50000]
    features=features[1+50000:maxtest+50000]

features = torch.tensor(features).type(torch.FloatTensor).cuda()
features = features.view(-1,10,400)

with torch.no_grad():
    if old:
        net = Unet1d_old(features.shape[1], opt.hidden_dim, opt.n_classes).cuda()
    else:
        net = Unet1d(features.shape[1], opt.hidden_dim, opt.n_classes).cuda()
    net.load_state_dict(torch.load(modelpath))
    net.eval()
    #net.train()
    train_predicted = net(features)

labels=labels
train_n=train_predicted.data.cpu().numpy()[:,0:800]


print(len(labels))
print(peaknum,ndiff)
crosscoef_test_n=[]
crosscoef_test_k=[]
for i in range (len(labels)):
    crosscoef_test_n.append(numpy.corrcoef(labels[i][0:400],train_n[i][0:400])[1,0])
    crosscoef_test_k.append(numpy.corrcoef(labels[i][399:-1],train_n[i][399:-1])[1,0])
#print(mean_squared_error(labels[:,0:400],train_n[:,0:400]))
train_distant_n = (labels[:,0:400]-train_n[:,0:400])**2
train_distant_n = np.average(train_distant_n, axis=1)
train_distant_n = np.sqrt(train_distant_n)
train_distant_n = np.average(train_distant_n)
print(train_distant_n)
#print(mean_squared_error(labels[:,399:-1],train_n[:,399:-1]))
train_distant_k = (labels[:,399:-1]-train_n[:,399:-1])**2
train_distant_k = np.average(train_distant_k, axis=1)
train_distant_k = np.sqrt(train_distant_k)
train_distant_k = np.average(train_distant_k)
print(train_distant_k)
print('crosscoef: test Accn=%.4f' % (sum(crosscoef_test_n) / len(crosscoef_test_n)))
print('crosscoef: test Acck=%.4f' % (sum(crosscoef_test_k) / len(crosscoef_test_k)))

plt.plot(train_n[0],'r')
plt.plot(labels[0],'g')
if blind ==1:
    numpy.savetxt("blindpredicted.csv", train_n, delimiter=",")
elif blind ==0:
    numpy.savetxt("trainpredicted.csv", train_n, delimiter=",")
plt.show()
