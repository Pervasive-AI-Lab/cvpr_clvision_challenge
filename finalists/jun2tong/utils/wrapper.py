
import torch
import torch.utils.data as data


class CustomTensorDataset(data.Dataset):
    def __init__(self, x_tensor, y_tensor, transform=None):
        self.tensors = (x_tensor, y_tensor)
        self.transform = transform
        self.length = x_tensor.shape[0]

    def __getitem__(self, index):
        x = self.tensors[0][index]

        if self.transform:
            x = self.transform(x)

        y = self.tensors[1][index]

        return x, y

    def __len__(self):
        return self.length


class LogitAppend(data.Dataset):
    """
    Use this only on training data
    """
    def __init__(self, dataset, y_tensor, name, remap=True):
        super(LogitAppend, self).__init__()
        self.dataset = dataset
        self.name = name
        self.number_classes = len(torch.unique(y_tensor))
        class_list = torch.unique(y_tensor).numpy()
        self.labels = torch.LongTensor(y_tensor)  # It will be used as tensor in loader generators.

        self.remap = remap
        if remap:
            self.class_mapping = {c: i for i, c in enumerate(class_list)}

        self.logits = [[] for _ in range(y_tensor.size(0))]

    def __len__(self):
        return len(self.logits)

    def __getitem__(self, index):
        img, _ = self.dataset[index]
        label = self.labels[index]
        if self.remap:
            raw_target = label.item() if isinstance(label, torch.Tensor) else label
            label = torch.tensor(self.class_mapping[raw_target])
        logits = self.logits[index]
        return img, label, logits

    def update_logits(self, nn_model, device):
        nn_model.eval()
        for idx in range(len(self.dataset)):
            img, _ = self.dataset[idx]
            img = img.unsqueeze(0).to(device)
            logits = nn_model(img)
            self.logits[idx] = torch.cat(logits, dim=1).to("cpu").squeeze(0).detach()
