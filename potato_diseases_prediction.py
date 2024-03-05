import numpy as np
import pandas as pd

from PIL import Image
import os
import matplotlib.pyplot as plt
import seaborn as sns
from copy import deepcopy

import torch
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def create_df(base_dir):
    dd = {"images": [], "labels": []}
    for i in os.listdir(base_dir):
        folder = os.path.join(base_dir, i)
        for j in os.listdir(folder):
            img = os.path.join(folder, j)
            dd["images"] += [img]
            dd["labels"] += [i]

    return pd.DataFrame(dd)

base_dir = "potato"
df = create_df(base_dir)

le = LabelEncoder()
df["labels"] = le.fit_transform(df["labels"].values)

EPOCHS = 35
SIZE = 224
LR = 0.1
STEP = 20
GAMMA = 0.1
BATCH = 16
NUM_CLASSES = 7


class CloudDS(Dataset):
    def __init__(self, data, transform):
        super(CloudDS, self).__init__()
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, x):
        img, label = self.data.iloc[x, 0], self.data.iloc[x, 1]
        img = Image.open(img).convert("RGB")
        img = np.array(img)
        img = self.transform(img)

        return img, label

transform_train = transforms.Compose([transforms.ToPILImage(),
                               transforms.ToTensor(),
                               transforms.Resize((SIZE, SIZE)),
                               transforms.RandomHorizontalFlip(p=0.5),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])
transform_val = transforms.Compose([transforms.ToPILImage(),
                               transforms.ToTensor(),
                               transforms.Resize((SIZE, SIZE)),
                               transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

train, mark = train_test_split(df, random_state=42, test_size=0.2)
val, test = train_test_split(mark, random_state=42, test_size=0.5)

train_ds = CloudDS(train, transform_train)
val_ds = CloudDS(val, transform_val)

train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
val_dl = DataLoader(val_ds, batch_size=BATCH, shuffle=False)


resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
num_ftrs = resnet.fc.in_features
resnet.fc = nn.Linear(num_ftrs, NUM_CLASSES)


class Classifier(nn.Module):
    def __init__(self, model):
        super(Classifier, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model(x)
        return nn.functional.softmax(x, dim=1)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(device)


model = Classifier(resnet)
model = model.to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP, gamma=GAMMA)
criterion = nn.CrossEntropyLoss()

best_model = deepcopy(model)
best_acc = 0.0
train_best = 0.0

loss_train = []
acc_train = []

loss_val = []
acc_val = []

for i in range(1, EPOCHS + 1):
    model.train()
    train_loss = 0.0
    train_acc = 0.0
    train_total = 0
    for img, target in train_dl:
        if torch.cuda.is_available():
            img, target = img.cuda(), target.cuda()
        optimizer.zero_grad()
        out = model(img)
        loss = criterion(out, target.long())
        train_loss += loss.item()
        train_acc += (out.argmax(1) == target).sum().item()
        train_total += out.size(0)
        loss.backward()
        optimizer.step()

    train_loss /= train_total
    train_acc /= train_total

    val_loss = 0.0
    val_acc = 0.0
    val_total = 0
    model.eval()
    with torch.no_grad():
        for img, target in val_dl:
            if torch.cuda.is_available():
                img, target = img.cuda(), target.cuda()

            out = model(img)
            loss = criterion(out, target.long())
            val_loss += loss.item()
            val_acc += (out.argmax(1) == target).sum().item()
            val_total += out.size(0)

    val_loss /= val_total
    val_acc /= val_total

    if val_acc >= best_acc and train_acc >= train_best:
        best_acc = val_acc
        train_best = train_acc
        best_model = deepcopy(model)

    loss_train += [train_loss]
    acc_train += [train_acc]
    loss_val += [val_loss]
    acc_val += [val_acc]

    print("Epoch {} train loss {} acc {} val loss {} acc {}".format(i, train_loss, train_acc,
                                                                    val_loss, val_acc))
    scheduler.step()

epochs = list(range(1, EPOCHS+1))
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10 ,5))
axes[0].plot(epochs, loss_train)
axes[0].plot(epochs, loss_val)
axes[0].set_title("Training and validation losses")
axes[0].legend(["Training", "Validation"])

axes[1].plot(epochs, acc_train)
axes[1].plot(epochs, acc_val)
axes[1].set_title("Training and validation accuracies")
axes[1].legend(["Training", "Validation"])
plt.tight_layout()
plt.show(block=True)


def predict(img_dir):
    img = np.array(Image.open(img_dir))
    img = transform_val(img)
    img = img.view([1, 3, SIZE, SIZE])
    if torch.cuda.is_available():
        img = img.cuda()
    best_model.eval()
    with torch.no_grad():
        out = model(img)

    return out.argmax(1).item()

predictions = []
truth = []
for i in range(len(test)):
    predictions += [predict(test.iloc[i, 0])]
    truth += [test.iloc[i, -1]]

score = accuracy_score(truth, predictions)
cm = confusion_matrix(truth, predictions)
report = classification_report(truth, predictions)
print("Score is: {}%".format(round(score*100, 2)))
print(report)
sns.heatmap(cm, annot=True)
plt.show(block=True)

# Score is: 82.61%
#               precision    recall  f1-score   support
#            0       0.60      1.00      0.75         6
#            1       1.00      0.60      0.75        10
#            2       0.75      0.75      0.75         4
#            3       0.67      1.00      0.80         2
#            4       1.00      1.00      1.00        10
#            5       0.71      0.71      0.71         7
#            6       1.00      0.86      0.92         7
#     accuracy                           0.83        46
#    macro avg       0.82      0.85      0.81        46
# weighted avg       0.87      0.83      0.83        46

fig, axes = plt.subplots(nrows=5, ncols=5, figsize=(15, 10))
index = 0
for i in range(5):
    for j in range(5):
        im = Image.open(test.iloc[index, 0]).convert("RGB")
        im = np.array(im)
        axes[i][j].imshow(im)
        axes[i][j].set_title("Predicted value: {}\nReal value: {}".format(le.inverse_transform([predictions[index]])[0],
                                                 le.inverse_transform([truth[index]])[0]))
        index += 1
plt.tight_layout()
plt.show(block=True)




