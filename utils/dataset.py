import numpy as np
import cv2
import torch
import torchvision.transforms as transform
from pathlib import Path
from torch.utils.data import DataLoader, Dataset

class PDBLOCTDataSet(Dataset) :
    def __init__(self, data_path, n_classes = 4, train_model = False):
        super(PDBLOCTDataSet, self).__init__()
        self.data_path = data_path
        self.normalize = True
        self.n_classes = n_classes
        self.train_model = train_model
        self.color_jitter = transform.ColorJitter(64.0/255, 0.75, 0.25, 0.04)
        self.preprocess()

    def preprocess(self):
        classes_paths = [name.__str__() for name in Path(self.data_path).iterdir() if name.is_dir()]
        classes_paths.sort()
        classes_to_idxs = {classes_paths[i].split('\\')[-1] : i for i in range(len(classes_paths))}
        self.items = []
        for idx, target in enumerate(classes_paths) :
            if not Path(target).is_dir() :
                continue

            for image_path in Path(target).iterdir() :
                fname = image_path.__str__().split('.')[-1]
                if fname == 'tif' or fname == 'jpeg' or fname == 'png' or fname == 'jpg' :
                    self.items.append((image_path.__str__(), classes_to_idxs[target.split('\\')[-1]]))

        self.num_items = len(self.items)

    def __getitem__(self, item):
        path, label = self.items[item]
        label = np.array(label, dtype = float)
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        img1 = cv2.resize(img, (224, 224))
        img2 = cv2.resize(img, (160, 160))
        img3 = cv2.resize(img, (112, 112))

        img1 = img1.astype(np.float32).transpose(2, 0, 1)
        if self.normalize :
            img1 = (img1 - 128.0) / 128.0

        img2 = img2.astype(np.float32).transpose(2, 0, 1)
        if self.normalize:
            img2 = (img2 - 128.0) / 128.0

        img3 = img3.astype(np.float32).transpose(2, 0, 1)
        if self.normalize:
            img3 = (img3 - 128.0) / 128.0

        label_onehot = np.zeros((self.n_classes))
        label_onehot[int(label)] = 1
        if self.train_model :
            return torch.FloatTensor(img1), torch.LongTensor(label)

        return img1, img2, img3, label_onehot

    def __len__(self):
        return len(self.items)

if __name__ == '__main__':
    data = PDBLOCTDataSet('../OCT2017/train')