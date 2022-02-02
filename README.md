# Package

```python
import os
import glob 
import pandas as pd
import string
import collections
from tqdm import tqdm
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
import torch.optim as optim
```

# Data preprocess

data 에 png 파일들을 불러옵니다.
DEVICE 는 gpu 사용이 가능하다면 ‘cuda’ 아니라면 cpu 사용 해줍니다.

```python
data = glob.glob(os.path.join('./samples/samples', '*.png'))
path = './samples/samples'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

# Token 생성

토큰을 생성합니다. all_letters 에 숫자와 알파벳을 포함한 모든 글자를 넣은 후, dictionary 를 생성하여 각 알파벳과 숫자를 mapping 해줍니다. 

```python
all_letters = string.ascii_lowercase + string.digits

mapping = {}
mapping_inv = {}
i = 1
for x in all_letters:
    mapping[x] = i
    mapping_inv[i] = x
    i += 1
```

# Data preview

상위 데이터 5개를 출력합니다. 이때, 이미지의 파일명이 라벨명이기 때문에 이미지의 파일명이 각 숫자로 맵핑이 잘 되었는지 확인합니다.

```python
images = []
labels = []
datas = collections.defaultdict(list)
for d in data:
    x = d.split('/')[-1]
    datas['image'].append(x)
    datas['label'].append([mapping[i] for i in x.split('.')[0]])
df = pd.DataFrame(datas)
df.head()
```

# Train_test_split

train 데이터와 test 데이터를 분리해줍니다.

```python
df_train, df_test = train_test_split(df, test_size=0.2, shuffle=True)
```

# Dataset 생성 + Transform + Dataloader

custom Dataset을 생성해줍니다.

이때 PIL package 를 사용하여 이미지 처리를 해줍시다.

```python
class CaptchaDataset:
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        data = self.df.iloc[idx]
        image = Image.open(os.path.join(path, data['image'])).convert('L')
        label = torch.tensor(data['label'], dtype=torch.int32)
        
        if self.transform is not None:
            image = self.transform(image)
        
        return image, label
        
        
transform = T.Compose([
    T.ToTensor()
])
    
train_data = CaptchaDataset(df_train, transform)
test_data = CaptchaDataset(df_test, transform)

train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
test_loader = DataLoader(test_data, batch_size=8)
```

# Bidirectional class

CNN 을 거쳐 나온 모델에 적용시킬 RNN 모델 입니다.

parameter 에 따라서 LSTM or GRU 중 선택한 뒤 양방향 학습을 True로 설정하였기 때문에 Linear 모델의 input에 출력 채널 * 2를 인자로 넣어줍니다.

```python
class Bidirectional(nn.Module):
    def __init__(self, inp, hidden, out, lstm=True):
        super(Bidirectional, self).__init__()
        if lstm:
            self.rnn = nn.LSTM(inp, hidden, bidirectional=True)
        else:
            self.rnn = nn.GRU(inp, hidden, bidirectional=True)
        
        #bidirectional = True 이기때문에 hidden * 2
        self.embedding = nn.Linear(hidden*2, out)
    def forward(self, X):
        recurrent, _ = self.rnn(X)
        out = self.embedding(recurrent)     
        return out
```

# CRNN class

CRNN 을 적용시킬 전체 모델의 class 입니다.

CNN → Linear → Bidirectional → CTCLoss  순으로 진행됩니다.

```python
class CRNN(nn.Module):
    def __init__(self, in_channels, output):
        super(CRNN, self).__init__()

        self.cnn = nn.Sequential(
                # in chaneel = 1
                nn.Conv2d(in_channels, 256, 9, stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
                nn.MaxPool2d(3, 3),
                nn.Conv2d(256, 256, (4, 3), stride=1, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256))
        
        self.linear = nn.Linear(3328, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.rnn = Bidirectional(256, 1024, output+1)

    def forward(self, X, y=None, criterion = None):
        out = self.cnn(X)
        N, C, w, h = out.size()
        out = out.view(N, -1, h)
        out = out.permute(0, 2, 1)
        out = self.linear(out)

        out = out.permute(1, 0, 2)
        out = self.rnn(out)
            
        if y is not None:
            T = out.size(0)
            N = out.size(1)
        
            input_lengths = torch.full(size=(N,), fill_value=T, dtype=torch.int32)
            target_lengths = torch.full(size=(N,), fill_value=5, dtype=torch.int32)
        
            loss = criterion(out, y, input_lengths, target_lengths)
            
            return out, loss
        
        return out, None
    
    def _ConvLayer(self, inp, out, kernel, stride, padding, bn=False):
        if bn:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU(),
                nn.BatchNorm2d(out)
            ]
        else:
            conv = [
                nn.Conv2d(inp, out, kernel, stride=stride, padding=padding),
                nn.ReLU()
            ]
        return nn.Sequential(*conv)
```

# Engine class + train

모델을 학습시킬 차례입니다. 

**init** 을 통하여 모델에 적용시킬 하이퍼 파라미터, device 를 입력받습니다.

fit 함수를 통하여 모델을 훈련시키고, eval 함수를 통하여 모델을 검증합니다.

predict 함수는 모델의 예측값과 실제 값을 보여주는 함수입니다.

```python
class Engine:
    def __init__(self, model, optimizer, criterion, epochs=50, early_stop=False, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.epochs = epochs
        self.early_stop = early_stop
        self.device = device
        
		#model fit
    def fit(self, dataloader):
        hist_loss = []
        for epoch in range(self.epochs):
            self.model.train()
            tk = tqdm(dataloader, total=len(dataloader))
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                self.optimizer.zero_grad()

                out, loss = self.model(data, target, criterion=self.criterion)

                loss.backward()

                self.optimizer.step()

                tk.set_postfix({'Epoch':epoch+1, 'Loss' : loss.item()})
	  
		#model evalutate
    def evaluate(self, dataloader):
        self.model.eval()
        loss = 0
        hist_loss = []
        outs = collections.defaultdict(list)
        tk = tqdm(dataloader, total=len(dataloader))
        with torch.no_grad():
            for data, target in tk:
                data = data.to(device=self.device)
                target = target.to(device=self.device)

                out, loss = self.model(data, target, criterion=self.criterion)
                
                outs['pred'].append(out)
                outs['target'].append(target)
                

                hist_loss.append(loss)

                tk.set_postfix({'Loss':loss.item()})
                
        return outs, hist_loss
    
    def predict(self, image):
        image = Image.open(image).convert('L')
        image_tensor = T.ToTensor()(image)
        image_tensor = image_tensor.unsqueeze(0)        
        out, _ = self.model(image_tensor.to(device=self.device))
        out = out.permute(1, 0, 2)
        out = out.log_softmax(2)
        out = out.argmax(2)
        out = out.cpu().detach().numpy()
        
        return out
        
            
            
            
model = CRNN(in_channels=1, output=num_class).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-4)
criterion = nn.CTCLoss()

engine = Engine(model, optimizer, criterion, device=DEVICE)
engine.fit(train_loader)
outs, loss = engine.evaluate(test_loader)
```