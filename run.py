from src.model.pipeline import VirusClassifier
from src.model.gradcam import *
from src.model.train import trainNet
from src.loaddata.data_loaderandargumenter import *
from src.config import *

if __name__ == '__main__':
    if opt.argumentation:
        dataset=make_dataset()
        dataset.data_formater()
        dataset.train_test_spliter()
        dataset.data_argument()
        pass
    
    virus = VirusClassifier()
    virus.main()


