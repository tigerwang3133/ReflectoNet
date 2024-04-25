from src.config import *
from sklearn.model_selection import train_test_split
import h5py

class make_dataset():
    def __init__(self):
        self.csv_path = ROOT + 'feature.csv'
        

    def data_formater(self):
        columnsusing=[i for i in range(401*10,401*12)]
        self.labels  = pd.read_csv(self.csv_path,header=None, index_col=False,usecols=columnsusing)
        self.labels =  np.array(self.labels)
        self.labels = np.delete(self.labels, [400, 801], axis=1)


        columnsusing=[i for i in range(0,401*10)]
        features = pd.read_csv(self.csv_path,header=None, index_col=False,usecols=columnsusing)
        #features.columns = features.columns.astype(float)
        self.features = np.array(features)
        #normalize
        self.features = np.delete(self.features, [i for i in range(400,4010,401)], axis=1)

        if opt.noise:
            random.seed(12138)
            self.features = self.features * np.random.normal(1, 0.002, self.features.shape)

        np.savetxt(ROOT + "featureinputsample.csv", self.features[0:100], delimiter=',')
        np.savetxt(ROOT + "featureoutputsample.csv", self.labels[0:100], delimiter=',')
        np.save(ROOT + "featureinput", self.features)
        np.save(ROOT + "featureoutput", self.labels)

        return

    def train_test_spliter(self):
        self.features=np.load(ROOT + "featureinput.npy")
        self.labels=np.load(ROOT + "featureoutput.npy")
        
        train_set_idx=[]
        test_set_idx=[]
        index=np.array([i for i in range(len(self.labels))])
        train_set_idx, test_set_idx = train_test_split(index, test_size=2/80, random_state=12138)
        train_set_idx.sort(axis=0)
        test_set_idx.sort(axis=0)
        self.train_set, self.test_set, self.train_labels, self.test_labels = self.features[train_set_idx], \
                                                         self.features[test_set_idx], \
                                                         self.labels[train_set_idx], \
                                                         self.labels[test_set_idx]
        
        return

    def data_argument(self):
        f_feature=open(ROOT + 'argumented_feature.npy','ab')
        f_label=open(ROOT + 'argumented_label.npy','ab')
        
        np.save(f_feature,self.train_set)
        np.save(f_label,self.train_labels)

        for i in range(0):
            train_set_argument=np.copy(self.train_set)
            random.seed(12138)
            swap_index=random.sample(range(0, 5), 2)
            train_set_argument[:,swap_index[0]*800:swap_index[0]*800+800]=self.train_set[:,swap_index[1]*800:swap_index[1]*800+800]
            train_set_argument[:,swap_index[1]*800:swap_index[1]*800+800]=self.train_set[:,swap_index[0]*800:swap_index[0]*800+800]
            np.save(f_feature,train_set_argument)
            np.save(f_label,self.train_labels)
        
        f_feature.close()
        f_label.close()

        
        np.savetxt(ROOT + 'test_feature.csv',self.test_set, delimiter=',')
        np.savetxt(ROOT + 'test_label.csv',self.test_labels, delimiter=',')
        return
