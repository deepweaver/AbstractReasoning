import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt 
import datetime,time
import os 
import torch 
if not os.path.isdir('./images'):
    os.mkdir('./images')
def getImages(array):
    assert array.shape == (160,160,9) 
    array = array.reshape((9,160,160)).astype(np.uint8)
    newarray = np.zeros((9,80,80)) 
    for j in range(9):
        img = Image.fromarray(array[j]).resize((80,80)) 
        # dataset_1b[i,j,:,:] = np.asarray(img, dtype=np.float64)  
        newarray[j,:,:] = np.asarray(img, dtype=np.uint8) 
    

def visualize(data):
    if data['image'].shape == (160,160,9):
        array = data['image'].reshape((9,160,160)).astype(np.uint8) 
    # plt.title("lskdjsl")
    fig=plt.figure(figsize=(6,5))
    fig.suptitle(str(data['relation_structure'][0]), fontsize=14, y=0.97, fontweight='semibold')
    columns = 4
    rows = 3
    x = 0
    for i in range(1, columns*rows+1):
        if i == 4 or i == 8 or i == 7:
            continue 

        img = array[x]
        x += 1
        tmp = fig.add_subplot(rows, columns, i, ) 
        if i == data['target']+9:
            # text = 'target'
            tmp.title.set_text('target')
        # tmp.axis('off')
        plt.setp(tmp.get_xticklabels(), visible=False)
        plt.setp(tmp.get_yticklabels(), visible=False)
        # https://stackoverflow.com/questions/29988241/python-hide-ticks-but-show-tick-labels/29988431
        tmp.tick_params(axis='both', which='both', length=0)
        plt.imshow(img, cmap='gray', ) 
    st = datetime.datetime.fromtimestamp(time.time()).strftime('%Y_%m_%d_%H_%M_%S')
    plt.savefig('./images/{}.png'.format(st),) 


class NormalDataset(torch.utils.data.Dataset):

    def __init__(self, root, train, transform=False):
        # read / download data
        self.datapath = root 
        self.train = train 
        self.transform = transform 
        
    def __getitem__(self, index):
        # return one item on the index
        if self.train:
            data = np.load(self.datapath+"analogy_novel.domain.transfer_train_normal_{}.npz".format(index))
            panels = data['image'].reshape((9,160,160)).astype(np.uint8)
            newpanels = np.zeros((9,80,80), dtype=np.float64)
            for j in range(9):
                # print(panels[j].shape)
                img = Image.fromarray(panels[j]).resize((80,80)) 
                newpanels[j,:,:] = np.asarray(img, dtype=np.float64)
            return torch.from_numpy(newpanels).float(), torch.from_numpy(np.asarray(data['target'], dtype=np.int64))
        else:
            data = np.load(self.datapath+"analogy_novel.domain.transfer_test_normal_{}.npz".format(index))
            panels = data['image'].reshape((9,160,160)).astype(np.uint8)
            newpanels = np.zeros((9,80,80), dtype=np.float64)
            for j in range(9):
                # print(panels[j].shape)
                img = Image.fromarray(panels[j]).resize((80,80)) 
                newpanels[j,:,:] = np.asarray(img, dtype=np.float64)
            return torch.from_numpy(newpanels).float(), torch.from_numpy(np.asarray(data['target'], dtype=np.int64))         
    def __len__(self,):
        # return the datalength
        length = 600000 if self.train else 100000
        return length










if __name__ == "__main__":
    data = np.load('/Users/tiany/Downloads/novel.domain.transfer/analogy_novel.domain.transfer_train_normal_{}.npz'.format(60000-2))
    array = data['image']
    title = str(data['relation_structure'])
    visualize(data)