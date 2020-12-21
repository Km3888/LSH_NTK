
import torch
from torch.utils.data import Dataset

class MSDDataset(Dataset):
    # A pytorch dataset class for holding data.
    def __init__(self,train=True,size=float('inf')):
        '''
        Takes as input the name of a file containing sentences with a classification label (comma separated) in each line.
        Stores the text data in a member variable X and labels in y
        '''

        # Opening the file and storing its contents in a list
        with open('/data/kelly/YearPredictionMSD.txt') as f:
            lines = f.read().split('\n')

        # Splitting the data and labels from each other
        X, y = [], []
        for i,line in enumerate(lines):
            if (not train and i<463715) or (train and i>=463715) or (not len(line)):
                continue
            x_vals=line.split(',')[1:]
            if len(x_vals)<90:
                continue
            X.append([float(x) for x in x_vals])
            y.append(float(line.split(',')[0]))

            if len(y)==size:
                break

        # Store them in member variables.
        self.X = torch.tensor(X)
        self.y = torch.tensor(y)

        self.X,self.x_avg,self.x_std=self.normalize(self.X)
        self.y, self.y_avg, self.y_std = self.normalize(self.y)

    def normalize(self,data):
        avg=data.mean(dim=0)
        data-=avg
        std=(data**2).mean(dim=0)
        std=std**0.5
        data/=std
        return data,avg,std


    def preprocess(self, data):
        return data

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        return self.preprocess(self.X[index]), self.y[index]

if __name__=='__main__':

    dataset1=MSDDataset(train=True)
    train_kwargs = {'batch_size': 16}
    dataset2=MSDDataset(train=False)
    loader= torch.utils.data.DataLoader(dataset1,**train_kwargs)
    X,y=next(iter(loader))