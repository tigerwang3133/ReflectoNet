from src.model.nn import *
from src.utils.util import visualize_1d_heatmap, update_dict, avg_dict
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import random
import h5py

class PrepareData():

    def __init__(self, X, y):
        if not torch.is_tensor(X):
            self.X = torch.from_numpy(X).requires_grad_()#torch.tensor(X).type(torch.FloatTensor).cuda()
            #self.X = self.X.view(-1,10,400)
        if not torch.is_tensor(y):
            self.y = torch.from_numpy(y).requires_grad_()#torch.tensor(y).type(torch.FloatTensor).cuda()

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

    

class VirusClassifier():
    def __init__(self):
        self.traininputpath = ROOT + 'argumented_feature.npy'
        self.trainoutputpath = ROOT + 'argumented_label.npy'
        
        self.testinputpath = ROOT + 'validateinput.csv'
        self.testoutputpath = ROOT + 'validateoutput.csv'

        self.blindinputpath =ROOT + 'blindtestinput.csv'
        self.blindoutputpath =ROOT + 'blindtestoutput.csv'
        self.realinputpath =ROOT + 'blindtestinput.csv'
        self.realoutputpath =ROOT + 'blindtestoutput.csv'

    def get_features(self):
        self.traininput= np.load(self.traininputpath)
        self.trainoutput= np.load(self.trainoutputpath)
        
        self.testinput= np.genfromtxt(self.testinputpath, delimiter=',')
        self.testoutput= np.genfromtxt(self.testoutputpath, delimiter=',')
        
        self.blindinput = np.genfromtxt(self.blindinputpath, delimiter=',')
        self.blindoutput = np.genfromtxt(self.blindoutputpath, delimiter=',')
        self.realinput = np.genfromtxt(self.realinputpath, delimiter=',')
        self.realoutput = np.genfromtxt(self.realoutputpath, delimiter=',')



    def eval(self, train_set, train_labels, test_set, test_labels, blindtestinput, blindtestlabel, realtestinput,realtestlabel):
        

        trainds = PrepareData(X=train_set,y=train_labels)
        train_loader = DataLoader(dataset=trainds, batch_size=opt.batch_size, shuffle=True, num_workers=0, pin_memory=True)

        testds = PrepareData(X=test_set,y=test_labels)
        test_loader = DataLoader(dataset=testds, batch_size=opt.batch_size, shuffle=False,num_workers=0, pin_memory=True)
        
        self.cnn1d = Classifier(n_feature=opt.n_channels,
                                hidden_dim=opt.hidden_dim,
                                n_output=opt.n_classes)
        self.cnn1d.fit(test_set, test_labels,train_loader,test_loader,\
                       blindtestinput, blindtestlabel, realtestinput,realtestlabel)


        return 0 
        


    def run(self, round):
        if opt.cross_val:
            for i in range(1):#train_set_idx, test_set_idx in skf.split(self.features, y=self.labels):
                

                print('train_sets, test_sets: ',self.traininput.shape, self.testinput.shape )
                print('train_labels, test_labels: ',self.trainoutput.shape, self.testoutput.shape)

                print('\n---------------------- START ---------------------\n')
                self.eval(self.traininput, self.trainoutput, self.testinput, self.testoutput,\
                          self.blindinput,self.blindoutput,self.realinput,self.realoutput)
            return 0

        else:
            return 0

    def main(self):
        self.get_features()

        for round in range(1):
            print('\n=================== START =====================\n')
            self.run(round)



