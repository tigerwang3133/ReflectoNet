from src.config import *
from src.model.gradcam import GradCam
from numpy import dot
from numpy.linalg import norm
import numpy
import inspect
import time

'''
#test
import numpy as np
from numpy import genfromtxt
import matplotlib.pyplot as plt
#test
'''

def kkpredict(prediction, target):
    #target= prediction.view(-1,800)
    n_index=torch.tensor([i for i in range(0,400)]).cuda()
    k_index=torch.tensor([i for i in range(400,800)]).cuda()
    predictionk=torch.index_select(prediction,1,k_index)
    n2=torch.index_select(prediction,1,n_index).data.cpu().numpy()
    '''
    #test
    predictionk=prediction[:,400:800]
    predictionn=prediction[:,400:800]
    #test
    '''

    wavenumber=np.array([i for i in range(400,800)])
    wavenumberi=np.array([i for i in range(1,1200)])
    startn2=np.tile(n2[:,0], (400, 1)).transpose()
    n2 = np.concatenate((np.tile(n2[:,0], (400,1)).transpose(), n2,np.tile(n2[:,399], (399,1)).transpose()), axis=1)
    
    n2=np.concatenate(([wavenumberi], n2), axis=0)
    n2[0,:]=2*299792458*3.1415936/(n2[0,:]/1000000000)
    n2= n2[:,n2[0].argsort()]
    k2=np.copy(n2)
    for j in range(399,len(k2[0])-400):
        w0=k2[0,j]
        intergral=0
        for k in range(1,len(k2[0])):
            if n2[0,k]-w0 !=0:
                intergral=intergral+((n2[1:,k]-1)/(n2[0,k]-w0))*(n2[0,k]-n2[0,k-1])
        k2[1:,j]=-1/3.1415926*intergral
    k2[0,:]=2*299792458*3.1415936/(k2[0,:]/1000000000)
    k2= k2[:,k2[0].argsort()]
    kkpredictionk=k2[1:,400:800]

    kkpredictionk = torch.tensor(kkpredictionk).type(torch.FloatTensor).cuda()
    return predictionk,kkpredictionk
'''
#test
prediction = genfromtxt('MoS2400_n.csv', delimiter=',')
realk = genfromtxt('MoS2400_k.csv', delimiter=',')
start=time.time()
[_,kk]=kkpredict(prediction, prediction)
print(time.time()-start)

plt.plot(realk[0,0:400])
plt.plot(kk[0,:])
plt.show()
#test
'''
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
        if opt.skip:
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
        x = self.dropout(x)

        x = self.upsample(x)
        if opt.skip:
            x = torch.cat([x,conv4], dim=1)
        x = self.dconv_up5(x)

        x = self.upsample(x)
        if opt.skip:
            x = torch.cat([x,conv3], dim=1)
        x = self.dconv_up4(x)

        x = self.upsample(x)
        if opt.skip:
            x = torch.cat([x,conv2], dim=1)
        x = self.dconv_up3(x)

        x = self.upsample(x)
        if opt.skip:
            x = torch.cat([x,conv1], dim=1)
        x = self.dconv_up2(x)

        x = self.lastlayer(x)
        x = self.flatten(x)


        return x


class Classifier:
    def __init__(self, n_feature, hidden_dim, n_output):
        # try switching to TwoInputConv1d
        self.net = Unet1d(n_feature, hidden_dim, n_output).cuda()
        #self.net = TwoInputCNN1d(n_feature, hidden_dim, n_output).cuda()
        self.net = self.net.cuda()
        if opt.exsistmodel:
            self.net.load_state_dict(torch.load(opt.exsistmodel))
            self.net.train()

    def fit(self, test_set, test_labels,train_loader,test_loader, blindtestinput, blindtestlabel, realtestinput,realtestlabel):

        
        optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr)#, betas=(0.9, 0.999))
        #decay lr
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=opt.lrdecay)
        #optimizer = torch.optim.SGD(self.net.parameters(), lr=opt.lr, momentum=0.9)#, weight_decay=1e-08)
        #scheduler = ReduceLROnPlateau(optimizer, factor=0.2, patience=3, mode='min', min_lr=1e-05)
        
        loss_func1 = nn.MSELoss()
        #loss_func1 = nn.L1Loss()
        loss_func2 = nn.CosineSimilarity(dim=1, eps=1e-6)
        #regularization

        torch.cuda.empty_cache()
        train_MSE=[]
        test_MSE=[]
        traink_MSE=[]
        testk_MSE=[]
        blindn_MSE=[]
        blindk_MSE=[]
        realn_MSE=[]
        realk_MSE=[]
        train_coeff=[]
        test_coeff=[]
        traink_coeff=[]
        testk_coeff=[]
        blindn_coeff=[]
        blindk_coeff=[]
        realn_coeff=[]
        realk_coeff=[]
        train_loss1n=[]
        train_loss2n=[]
        train_loss3=[]
        train_loss1k=[]
        train_loss2k=[]
        with torch.autograd.set_detect_anomaly(True):
            swap_index=[]
            for i in range(opt.argumentation):
                single_index=(random.sample(range(0, 5), 5))
                single_index_line=[]
                for j in single_index:
                    single_index_line=single_index_line+[i + 800*j -1 for i in list(range(1,801))]
                swap_index.append(single_index_line)
            for t in range(opt.ml_n_epochs):
                starttime=time.time()
                for swap in range(opt.argumentation):
                    for ix, (_x, _y) in enumerate(train_loader):
                        _x = _x[:,swap_index[swap]]
                        _y = torch.tensor(_y).type(torch.FloatTensor).cuda()
                        _x = torch.tensor(_x).type(torch.FloatTensor).cuda()
                        _x= _x.view(-1,10,400)
                        prediction = self.net(_x)#.unsqueeze(2))
                        if opt.kkrelation:
                            n_index=torch.tensor([i for i in range(0,400)]).cuda()
                            k_index=torch.tensor([i for i in range(400,800)]).cuda()
                            prediction_n=torch.index_select(prediction,1,n_index)
                            prediction_k=torch.index_select(prediction,1,k_index)
                            _y_n=torch.index_select(_y,1,n_index)
                            _y_k=torch.index_select(_y,1,k_index)
                            #mse
                            loss1n = loss_func1(prediction_n, _y_n)
                            loss1k = loss_func1(prediction_k, _y_k)

                            #pearson
                            '''
                            loss2n =1.-loss_func2(prediction_n - prediction_n.mean(dim=1, keepdim=True),\
                                               _y_n - _y_n.mean(dim=1, keepdim=True))
                            loss2n = loss2n.mean()
                            loss2k =1.-loss_func2(prediction_k - prediction_k.mean(dim=1, keepdim=True),\
                                               _y_k - _y_k.mean(dim=1, keepdim=True))
                            loss2k = loss2k.mean()
                            '''
                            loss2 =1-loss_func2(prediction - prediction.mean(dim=1, keepdim=True),\
                                               _y - _y.mean(dim=1, keepdim=True))
                            loss2 = loss2.mean()

                            #kk pearson
                            predictionk, targetk = kkpredict(prediction, _y)
                            loss3 = 1-loss_func2(predictionk - predictionk.mean(dim=1, keepdim=True),\
                                               targetk - targetk.mean(dim=1, keepdim=True))
                            loss3 = loss3.mean()

                            #weighted loss                        
                            loss = (0.5*loss1n+0.5*loss1k)+10*loss2+10*loss3
                            
                            train_loss1n.append(loss1n)
                            train_loss2n.append(loss2)
                            train_loss1k.append(loss1k)
                            train_loss2k.append(loss2)
                            train_loss3.append(loss3)
                            
                        else:
                            n_index=torch.tensor([i for i in range(0,400)]).cuda()
                            k_index=torch.tensor([i for i in range(400,800)]).cuda()
                            prediction_n=torch.index_select(prediction,1,n_index)
                            prediction_k=torch.index_select(prediction,1,k_index)
                            _y_n=torch.index_select(_y,1,n_index)
                            _y_k=torch.index_select(_y,1,k_index)
                            #mse
                            loss1n = loss_func1(prediction_n, _y_n)
                            loss1k = loss_func1(prediction_k, _y_k)
                            
                            #pearson
                            '''
                            loss2n =1.-loss_func2(prediction_n - prediction_n.mean(dim=1, keepdim=True),\
                                               _y_n - _y_n.mean(dim=1, keepdim=True))
                            loss2n = loss2n.mean()
                            loss2k =1.-loss_func2(prediction_k - prediction_k.mean(dim=1, keepdim=True),\
                                               _y_k - _y_k.mean(dim=1, keepdim=True))
                            loss2k = loss2k.mean()
                            '''
                            loss2 =1-loss_func2(prediction - prediction.mean(dim=1, keepdim=True),\
                                               _y - _y.mean(dim=1, keepdim=True))
                            loss2 = loss2.mean()
                            
                            #weighted loss                        
                            loss = (0.5*loss1n+0.5*loss1k)+8*loss2
                            train_loss1n.append(loss1n)
                            train_loss2n.append(loss2)
                            train_loss1k.append(loss1k)
                            train_loss2k.append(loss2)
                        self.net.zero_grad()
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        #print(t)
                        torch.cuda.empty_cache()
                print('time thie epoch',time.time()-starttime)
                #decay lr
                if t>30:
                    lr_scheduler.step()
                #save model
                if (t+1) % 5 == 0:
                    torch.save(self.net.state_dict(), str(opt.lr)+"_"+str(t)+"_"+"model.pt")
                if 1:#t % 200 == 0 or t==opt.ml_n_epochs-1:
                    print('\n--------------- Epoch {} ----------------\n'.format(t))
                    with torch.no_grad():
                        self.net.eval()
                        torch.cuda.empty_cache()
                        for ix, (_x, _y) in enumerate(train_loader):
                            _y = torch.tensor(_y).type(torch.FloatTensor).cuda()
                            _x = torch.tensor(_x).type(torch.FloatTensor).cuda()
                            _x= _x.view(-1,10,400)
                            if ix==0:
                                train_predicted = self.net(_x)
                                train_labels = _y.data.cpu().numpy()
                            #else:
                            #    out=self.net(_x)
                            #    train_predicted = torch.cat((train_predicted, out), 0)
                        for ix, (_x, _y) in enumerate(test_loader):
                            _y = torch.tensor(_y).type(torch.FloatTensor).cuda()
                            _x = torch.tensor(_x).type(torch.FloatTensor).cuda()
                            _x= _x.view(-1,10,400)
                            if ix==0:
                                test_predicted = self.net(_x)
                            else:
                                out=self.net(_x)
                                test_predicted = torch.cat((test_predicted, out), 0)
                        #blind and real test
                        
                        blindx = torch.tensor(blindtestinput).type(torch.FloatTensor).cuda()
                        blindx = blindx.view(-1,10,400)
                        realx = torch.tensor(realtestinput).type(torch.FloatTensor).cuda()
                        realx = realx.view(-1,10,400)
                        
                        blind_predicted=self.net(blindx)
                        real_predicted=self.net(realx)

                        true_n=train_labels[:,0:400]
                        true_k=train_labels[:,399:-1]
                        train_n=train_predicted.data.cpu().numpy()[:,0:400]
                        train_k=train_predicted.data.cpu().numpy()[:,399:-1]
                        test_n=test_predicted.data.cpu().numpy()[:,0:400]
                        test_k=test_predicted.data.cpu().numpy()[:,399:-1]
                        true_test_n=test_labels[:,0:400]
                        true_test_k=test_labels[:,399:-1]

                        blind_n=blind_predicted.data.cpu().numpy()[:,0:400]
                        blind_k=blind_predicted.data.cpu().numpy()[:,399:-1]
                        real_n=real_predicted.data.cpu().numpy()[:,0:400]
                        real_k=real_predicted.data.cpu().numpy()[:,399:-1]
                        
                        #calculate distance
                        train_distant_n = (true_n-train_n)**2
                        train_distant_n = np.average(train_distant_n, axis=1)
                        train_distant_n = np.sqrt(train_distant_n)
                        train_distant_n = np.average(train_distant_n)
                        
                        train_distant_k = (true_k-train_k)**2
                        train_distant_k = np.average(train_distant_k, axis=1)
                        train_distant_k = np.sqrt(train_distant_k)
                        train_distant_k = np.average(train_distant_k)
                        
                        test_distant_n = (true_test_n-test_n)**2
                        test_distant_n = np.average(test_distant_n, axis=1)
                        test_distant_n = np.sqrt(test_distant_n)
                        test_distant_n = np.average(test_distant_n)
                        
                        test_distant_k = (true_test_k-test_k)**2
                        test_distant_k = np.average(test_distant_k, axis=1)
                        test_distant_k = np.sqrt(test_distant_k)
                        test_distant_k = np.average(test_distant_k)

                        blind_distant_n=(blindtestlabel[:,0:400]-blind_n)**2
                        blind_distant_n = np.average(blind_distant_n, axis=1)
                        blind_distant_n = np.sqrt(blind_distant_n)
                        blind_distant_n = np.average(blind_distant_n)

                        blind_distant_k=(blindtestlabel[:,399:-1]-blind_k)**2
                        blind_distant_k = np.average(blind_distant_k, axis=1)
                        blind_distant_k = np.sqrt(blind_distant_k)
                        blind_distant_k = np.average(blind_distant_k)

                        real_distant_n=(realtestlabel[:,0:400]-real_n)**2
                        real_distant_n = np.average(real_distant_n, axis=1)
                        real_distant_n = np.sqrt(real_distant_n)
                        real_distant_n = np.average(real_distant_n)
                        
                        real_distant_k=(realtestlabel[:,399:-1]-real_k)**2
                        real_distant_k = np.average(real_distant_k, axis=1)
                        real_distant_k = np.sqrt(real_distant_k)
                        real_distant_k = np.average(real_distant_k)
                        
                        true_intenisty_n = np.average(true_n**2, axis=1)
                        true_intenisty_n = np.sqrt(true_intenisty_n)
                        true_intenisty_n = np.average(true_intenisty_n)
                        
                        true_intensity_k = np.average(true_k**2, axis=1)
                        true_intensity_k = np.sqrt(true_intensity_k)
                        true_intensity_k = np.average(true_intensity_k)                       
                        #caculate cosinesimilarity
                        '''
                        cossim_train_n= dot(true_n, train_n)/(norm(true_n)*norm(train_n))
                        cossim_train_n = np.average(cossim_train_n.data.cpu().numpy())
                        cossim_train_k=dot(true_k, train_k)/(norm(true_k)*norm(train_k))
                        cossim_train_k = np.average(cossim_train_k.data.cpu().numpy())
                        cossim_test_n=dot(true_test_n, test_n)/(norm(true_test_n)*norm(test_n))
                        cossim_test_n = np.average(cossim_test_n.data.cpu().numpy())
                        cossim_test_k=dot(true_test_k, test_k)/(norm(true_test_k)*norm(test_k))
                        cossim_test_k = np.average(cossim_test_k.data.cpu().numpy())
                        '''
                        #caculate cross correlation coef
                        crosscoef_train_n=[]
                        crosscoef_train_k=[]
                        crosscoef_test_n=[]
                        crosscoef_test_k=[]
                        crosscoef_blind_n=[]
                        crosscoef_blind_k=[]
                        crosscoef_real_n=[]
                        crosscoef_real_k=[]
                        for i in range (len(true_n)):
                            crosscoef_train_n.append( numpy.corrcoef(true_n[i],train_n[i])[1,0])
                            crosscoef_train_k.append( numpy.corrcoef(true_k[i],train_k[i])[1,0])
                        for i in range(len(true_test_n)):
                            crosscoef_test_n.append( numpy.corrcoef(true_test_n[i],test_n[i])[1,0])
                            crosscoef_test_k.append( numpy.corrcoef(true_test_k[i],test_k[i])[1,0])
                        for i in range(len(blindtestlabel)):
                            crosscoef_blind_n.append( numpy.corrcoef(blindtestlabel[i,0:400],blind_n[i])[1,0])
                            crosscoef_blind_k.append( numpy.corrcoef(blindtestlabel[i,399:-1],blind_k[i])[1,0])
                        for i in range(len(realtestlabel)):
                            crosscoef_real_n.append( numpy.corrcoef(realtestlabel[i,0:400],real_n[i])[1,0])
                            crosscoef_real_k.append( numpy.corrcoef(realtestlabel[i,399:-1],real_k[i])[1,0])
                            
                        print('Epoch=%d, distance: train Accn=%.4f / %.4f Acck=%.4f / %.4f\n' % (t,train_distant_n,true_intenisty_n,train_distant_k,true_intensity_k))
                        #print('          cossimilarity: train Accn=%.4f Acck=%.4f\n' % (cossim_train_n,cossim_train_k))
                        print('          crosscoef: train Accn=%.4f Acck=%.4f\n' % (sum(crosscoef_train_n) / len(crosscoef_train_n) ,sum(crosscoef_train_k) / len(crosscoef_train_k)))
                        print('Epoch=%d, distance: test Accn=%.4f / %.4f Acck=%.4f / %.4f\n' % (t, test_distant_n,true_intenisty_n,test_distant_k,true_intensity_k))
                        #print('          cossimilarity: train Accn=%.4f Acck=%.4f\n' % (cossim_test_n,cossim_test_k))
                        print('          crosscoef: test Accn=%.4f Acck=%.4f\n' % (sum(crosscoef_test_n) / len(crosscoef_test_n) ,sum(crosscoef_test_k) / len(crosscoef_test_k)))

                        print('Epoch=%d, distance: blind Accn=%.4f Acck=%.4f \n' % (t, blind_distant_n,blind_distant_k))
                        #print('          cossimilarity: train Accn=%.4f Acck=%.4f\n' % (cossim_test_n,cossim_test_k))
                        print('          crosscoef: blind Accn=%.4f Acck=%.4f\n' % (sum(crosscoef_blind_n)/len(crosscoef_blind_n) ,sum(crosscoef_blind_k)/len(crosscoef_blind_k)))
                        print('Epoch=%d, distance: real Accn=%.4f  Acck=%.4f  \n' % (t, real_distant_n,real_distant_k))
                        #print('          cossimilarity: train Accn=%.4f Acck=%.4f\n' % (cossim_test_n,cossim_test_k))
                        print('          crosscoef: real Accn=%.4f Acck=%.4f\n' % (sum(crosscoef_real_n)/len(crosscoef_real_n) ,sum(crosscoef_real_k)/len(crosscoef_real_k)))

                        
                        train_MSE.append(train_distant_n)
                        test_MSE.append(test_distant_n)
                        train_coeff.append(sum(crosscoef_train_n) / len(crosscoef_train_n))
                        test_coeff.append(sum(crosscoef_test_n) / len(crosscoef_test_n))
                        
                        traink_MSE.append(train_distant_k)
                        testk_MSE.append(test_distant_k)
                        traink_coeff.append(sum(crosscoef_train_k) / len(crosscoef_train_k))
                        testk_coeff.append(sum(crosscoef_test_k) / len(crosscoef_test_k))

                        blindn_MSE.append(blind_distant_n)
                        blindk_MSE.append(blind_distant_k)
                        blindn_coeff.append(sum(crosscoef_blind_n)/len(crosscoef_blind_n))
                        blindk_coeff.append(sum(crosscoef_blind_k)/len(crosscoef_blind_k))
                                                

                        realn_MSE.append(real_distant_n)
                        realk_MSE.append(real_distant_k)
                        realn_coeff.append(sum(crosscoef_real_n)/len(crosscoef_real_n))
                        realk_coeff.append(sum(crosscoef_real_k)/len(crosscoef_real_k))
                        
                        self.net.train()
            y_pred = test_predicted
            y_train_pred= train_predicted
            #save results
            torch.save(self.net.state_dict(), str(opt.lr)+"model.pt")
            np.savetxt(str(opt.lr)+"_ypred.csv", y_pred.cpu().detach().numpy(), delimiter=",")
            np.savetxt(str(opt.lr)+"_ypredtrue.csv",test_labels , delimiter=",")
            '''
            if opt.kkrelation:
                np.savetxt(str(opt.lr)+"_trainLOSSS.csv",np.array([train_loss1n,train_loss1k,train_loss2n,train_loss2k,train_loss3]), delimiter=",")
            else:
                np.savetxt(str(opt.lr)+"_trainLOSSS.csv",np.array([train_loss1n,train_loss1k,train_loss2n,train_loss2k]), delimiter=",")
            '''
            np.savetxt(str(opt.lr)+"_trainLOSS.csv",np.array([train_MSE,traink_MSE, train_coeff, traink_coeff ]), delimiter=",")
            np.savetxt(str(opt.lr)+"_testLOSS.csv",np.array([test_MSE, testk_MSE, test_coeff, testk_coeff, blindn_MSE, blindk_MSE, blindn_coeff, blindk_coeff,realn_MSE, realk_MSE,realn_coeff, realk_coeff]) ,delimiter=",")
            np.savetxt(str(opt.lr)+"_ytestinput.csv",test_set, delimiter=",")
        return loss

