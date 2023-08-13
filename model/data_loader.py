import random
import os
import numpy as np
import glob
import torch

from torch.utils.data.sampler import BatchSampler
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms

# borrowed from http://pytorch.org/tutorials/advanced/neural_style_tutorial.html
# and http://pytorch.org/tutorials/beginner/data_loading_tutorial.html
# define a training image loader that specifies transforms on images. See documentation for more details.
train_transformer = transforms.Compose([
    # transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.RandomHorizontalFlip(),  # randomly flip image horizontally
    transforms.RandomVerticalFlip(),
    transforms.ToTensor()])  # transform it into a torch tensor

# loader for evaluation, no horizontal flip
eval_transformer = transforms.Compose([
    # transforms.Resize(64),  # resize the image to 64x64 (remove if images are already 64x64)
    transforms.ToTensor()])  # transform it into a torch tensor


class GenericDataset(Dataset):
    """
    A standard PyTorch definition of Dataset which defines the functions __len__ and __getitem__.
    """
    def __init__(self, data_dir, transform):
        """
        Store the filenames of the jpgs to use. Specifies transforms to apply on images.

        Args:
            data_dir: (string) directory containing the dataset
            transform: (torchvision.transforms) transformation to apply on image
        """
        self.filenames = os.listdir(data_dir)
        self.filenames = [os.path.join(data_dir, f) for f in self.filenames if f.endswith('.jpg') or f.endswith('.tiff') or f.endswith('.bmp')]

        self.labels = [os.path.split(filename)[-1].split('_')[0]+'_'+os.path.split(filename)[-1].split('_')[1] for filename in self.filenames]
        self.labels_set = list(set(self.labels))
        self.str_to_int_dict = {label: self.labels_set.index(label)+1
                                 for label in self.labels_set}
        self.labels = [self.str_to_int_dict[label] for label in self.labels]
        self.transform = transform


    def __len__(self):
        # return size of dataset
        return len(self.filenames)

    def __getitem__(self, idx):
        """
        Fetch index idx image and labels from dataset. Perform transforms on image.

        Args:
            idx: (int) index in [0, 1, ..., size_of_dataset-1]

        Returns:
            image: (Tensor) transformed image
            label: (int) corresponding label of image
        """
        image = Image.open(self.filenames[idx])  # PIL image
        image = self.transform(image)
        return image, self.labels[idx]


def fetch_dataloader(types, data_dir, dataset, params):
    """
    Normal data loader. Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}

    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, f'{split}_{dataset}')

            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                dl = DataLoader(GenericDataset(path, train_transformer), batch_size=params.batch_size, shuffle=True,
                                num_workers=params.num_worker,
                                pin_memory=params.pin_memory)
            else:
                dl = DataLoader(GenericDataset(path, eval_transformer), batch_size=params.batch_size, shuffle=False,
                                num_workers=params.num_worker,
                                pin_memory=params.pin_memory)

            dataloaders[split] = dl

    return dataloaders
    

class BalancedBatchSampler(BatchSampler):
    """
    BatchSampler - from a dataset, samples n_classes and within these classes samples m_samples.
    Returns batches of size n_classes * m_samples
    """

    def __init__(self, labels, n_classes, m_samples):
        self.labels = np.array(labels)
        self.labels_set = list(set(self.labels))
        self.label_to_indices = {label: np.where(self.labels == label)[0]
                                 for label in self.labels_set}
        np.random.seed(seed=230)
        for l in self.labels_set:
            self.label_to_indices[l].sort()
            np.random.shuffle(self.label_to_indices[l])
        self.used_label_indices_count = {label: 0 for label in self.labels_set}
        self.count = 0
        self.n_classes = n_classes
        self.m_samples = m_samples
        self.n_dataset = len(self.labels)
        self.batch_size = self.m_samples * self.n_classes

    def __iter__(self):
        self.count = 0
        while self.count + self.batch_size < self.n_dataset:
            classes = np.random.choice(self.labels_set, self.n_classes, replace=False)
            indices = []
            for class_ in classes:
                indices.extend(self.label_to_indices[class_][
                               self.used_label_indices_count[class_]:self.used_label_indices_count[
                                                                         class_] + self.m_samples])
                self.used_label_indices_count[class_] += self.m_samples
                if self.used_label_indices_count[class_] + self.m_samples > len(self.label_to_indices[class_]):
                    np.random.shuffle(self.label_to_indices[class_])
                    self.used_label_indices_count[class_] = 0
            yield indices
            self.count += self.n_classes * self.m_samples

    def __len__(self):
        return self.n_dataset // self.batch_size

    
def fetch_online_dataloader(types, data_dir, dataset, params):
    """
    Normal data loader. Fetches the DataLoader object for each type in types from data_dir.

    Args:
        types: (list) has one or more of 'train', 'val', 'test' depending on which data is required
        data_dir: (string) directory containing the dataset
        dataset: (string) choosing dataset (From cli args.dataset)
        params: (Params) hyperparameters

    Returns:
        data: (dict) contains the DataLoader object for each type in types
    """
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        if split in types:
            path = os.path.join(data_dir, f'{split}_{dataset}')
            # use the train_transformer if training data, else use eval_transformer without random flip
            if split == 'train':
                custom_batch_sampler = BalancedBatchSampler(GenericDataset(path, train_transformer).labels, n_classes=params.train_n_classes, m_samples=params.train_m_samples)
                dl = DataLoader(GenericDataset(path, eval_transformer), batch_sampler=custom_batch_sampler,
                                num_workers=params.num_workers,
                                pin_memory=params.pin_memory)

            else:
                custom_batch_sampler = BalancedBatchSampler(GenericDataset(path, train_transformer).labels, n_classes=params.eval_n_classes, m_samples=params.eval_m_samples)
                dl = DataLoader(GenericDataset(path, eval_transformer), batch_sampler=custom_batch_sampler,
                                num_workers=params.num_workers,
                                pin_memory=params.pin_memory)


            dataloaders[split] = dl

    return dataloaders


if __name__ == "__main__":
    _ = GenericDataset('./data/IIT Delhi V1/train_IIT Delhi V1 Segmented', train_transformer)
    print(len(_.labels))
    print(len(set(_.labels)))
    _ = GenericDataset('./data/Tongji/train_Tongji Segmented', train_transformer)
    print(len(_.labels))
    print(len(set(_.labels)))
    _ = GenericDataset('./data/REST/train_REST', train_transformer)
    print(len(_.labels))
    print(len(set(_.labels)))
  