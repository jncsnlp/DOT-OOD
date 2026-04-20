from PIL import Image
from torch.utils.data import Dataset
import os

class ImageTextDataset(Dataset):
    def __init__(self, root, datalist, preprocess):
        self.root = root
        # self.datalist = datalist
        self.imlist = self.__default_flist_reader__(datalist)
        self.preprocess = preprocess

    def __default_loader__(self, path):
        return self.preprocess(Image.open(path))
        # return Image.open(path).convert("JPEG")

    def __default_flist_reader__(self, flist):
        imlist = []
        with open(flist, "r") as rf:
            for line in rf.readlines():
                data = line.strip()
                data = data.split(" ")
                if len(data)==2:
                    impath = data[0]
                    imlabel = int(data[1])
                else:
                    impath, imlabel = data[0],-1
                imlist.append((impath, imlabel))
                # print(imlist[0])
        return imlist

    def __getitem__(self, idx):
        impath, text = self.imlist[idx]
        img = self.__default_loader__(os.path.join(self.root, impath))
        return img, text

    def __len__(self):
        return len(self.imlist)
