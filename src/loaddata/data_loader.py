from src.config import *
import warnings

from src.loaddata.prototype_sampler import PrototypicalBatchSampler

warnings.filterwarnings("ignore")


def make_dataset(path):
    data = []
    for c in os.listdir(path):
        c_path = os.path.join(path, c)
        for fname in os.listdir(c_path):
            fpath, ID, l = os.path.join(c_path, fname), \
                           int(fname.split('_')[0].replace('sample', '')), \
                           int(fname.split('_')[1].replace('.png', ''))
            item = (fpath, ID, l)
            data.append(item)
    if len(data) == 0:
        raise Exception("No data are found in {}".format(path))
    return data


class StructureDataset(data.Dataset):

    def __init__(self, type, transform, path):

        self.root = path
        self.type = type
        if not os.path.exists(self.root):
            raise Exception("[!] {} not exists.".format(self.root))

        class_idx = {CLASSES[i]: i for i in range(len(CLASSES))}

        samples = make_dataset(path)

        self.classes = CLASSES
        self.class_idx = class_idx
        self.samples = samples
        self.transform = transform
        self.labels = [item[-1] for item in self.samples]

    def __getitem__(self, index):
        fpath, ID, label = self.samples[index]
        data = Image.open(fpath).convert('RGB')
        if self.transform:
            data = self.transform(data)
        return fpath, data, ID, label

    def __len__(self):
        return len(self.samples)


def get_loader(batch_size=64,
               num_workers=4,
               path='',
               num_samples=15):
    dataset = StructureDataset('train', transform=transforms.Compose([
        transforms.ToTensor()
    ]), path=path)

    mean = 0.0
    for _, img, _, _ in dataset:
        mean += img.mean([1, 2])
    mean = mean / len(dataset)
    print(mean)

    sumel = 0.0
    countel = 0
    for _, img, _, _ in dataset:
        img = (img - mean.unsqueeze(1).unsqueeze(1)) ** 2
        sumel += img.sum([1, 2])
        countel += torch.numel(img[0])
    std = torch.sqrt(sumel / countel)
    print(std)

    dataset = StructureDataset('train', transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=mean,
            std=std
        )
    ]), path=path)

    sampler = PrototypicalBatchSampler(labels=dataset.labels,
                                       classes_per_it=opt.n_classes,
                                       num_samples=num_samples,
                                       iterations=50)


    data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                              batch_size=batch_size,
                                              # batch_sampler=sampler,
                                              shuffle=True,
                                              num_workers=num_workers,
                                              drop_last=False)
    return data_loader
