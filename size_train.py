# %%
from functools import reduce
import os
import json

# %%
HEIGHT2STRIDE = {
    16: [2] * 4 + [1],
    32: [2] * 5,
    64: [4] + [2] * 4,
    128: [4] * 2 + [2] * 3,
    256: [4] * 3 + [2] * 2,
    512: [4] * 4 + [2] * 1,
}

for k, v in HEIGHT2STRIDE.items():
    assert reduce(lambda x, y: x * y, v) == k

# %%
vocab = open("vocab.txt").read().splitlines()
vocab_size = len(vocab)
vocab_size

SIZE = 64

stride_list = HEIGHT2STRIDE[SIZE]
images_path = f"images_size/size={SIZE}/"
with open("size2labels.json", "rb") as f:
    size2labels = json.load(f)
labels = size2labels[str(SIZE)]
val_labels = {k: v for k, v in labels.items() if int(k) < 100}
train_labels = {k: v for k, v in labels.items() if int(k) >= 100}

# %%
len(val_labels), len(train_labels)

# %%


# %% [markdown]
# # Dataset

# %%
from torch.utils.data import Dataset, DataLoader
import torchvision as tv
from PIL import Image


# %%
class SimpleDataset(Dataset):
    def __init__(self, images_path, labels):
        self.images_path = images_path
        self.labels = labels
        self.keys = sorted(list(labels.keys()))

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, idx):
        idx = self.keys[idx]
        image_path = os.path.join(self.images_path, f"{idx}.png")
        image = Image.open(image_path)
        image = image.convert("RGB")

        image = tv.transforms.ToTensor()(image)

        label = self.labels[str(idx)]
        return image, label


# %%
train_dataset = SimpleDataset(images_path, train_labels)
val_dataset = SimpleDataset(images_path, val_labels)

# %%
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# %% [markdown]
# # Model

# %%
import torch

from model import CNN
from ctc import GreedyCTCDecoder
from metrics import compute_f1, compute_exact

# %%
decoder = GreedyCTCDecoder()

# %%
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# %%
def prepare_batch(batch):
    images, texts = batch
    images = images.to(device)

    y = [[vocab.index(t) for t in txt] for txt in texts]
    y = torch.tensor(y, dtype=torch.long).to(device)

    return images, texts, y


# %%
def get_ctc_loss(logits, y):
    logits = logits.permute(1, 0, 2).log_softmax(2)
    input_lengths = torch.full(
        size=(logits.size(1),),
        fill_value=logits.size(0),
        dtype=torch.int32,
    )
    target_lengths = torch.full(
        size=(y.size(0),),
        fill_value=y.size(1),
        dtype=torch.int32,
    )
    criterion = torch.nn.CTCLoss(zero_infinity=True)
    loss = criterion(logits, y, input_lengths, target_lengths)
    return loss


# %%
def get_predictions(logits):
    yp = logits.argmax(-1)
    pt = []
    for row in yp:
        predictions = decoder(row, None)
        pt.append("".join(vocab[p] for p in predictions))
    return pt


# %%
model = CNN(stride_list, vocab_size, kernel_size=5)
_ = model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# %%
gradient_steps = 4
train_losses = []
val_epoch = {
    "loss": [],
    "f1": [],
    "em": [],
}
for epoch in range(100):
    model.train()
    for idx, batch in enumerate(train_loader):
        images, texts, y = prepare_batch(batch)
        logits = model(images)
        loss = get_ctc_loss(logits, y)
        loss.backward()
        if idx % gradient_steps == 0 or idx == len(train_loader) - 1:
            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(loss.item())

    model.eval()
    with torch.no_grad():
        val_losses = []
        val_f1 = []
        val_em = []
        for batch in val_loader:
            images, texts, y = prepare_batch(batch)
            logits = model(images)
            pt = get_predictions(logits)
            loss = get_ctc_loss(logits, y)

            val_losses.append(loss.item())
            for t, p in zip(texts, pt):
                val_f1.append(compute_f1(t, p))
                val_em.append(compute_exact(t, p))
    val_epoch["loss"].append(sum(val_losses) / len(val_losses))
    val_epoch["f1"].append(sum(val_f1) / len(val_f1))
    val_epoch["em"].append(sum(val_em) / len(val_em))
    print(
        f"epoch: {epoch}, train loss: {sum(train_losses) / len(train_losses)}, val loss: {sum(val_losses) / len(val_losses)}, val f1: {sum(val_f1) / len(val_f1)}, val em: {sum(val_em) / len(val_em)}"
    )

torch.save(model.state_dict(), f"cnn_size={SIZE}.pth")
history = val_epoch
history.update({"train_losses": train_losses})
with open(f"cnn_size={SIZE}.json", "w") as f:
    json.dump(val_epoch, f)

# %%
