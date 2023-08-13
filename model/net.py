"""Defines the neural network, losss function and metrics"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.backbones import *
from itertools import combinations


Net = convnextv2_atto


class SiameseNet(nn.Module):
    def __init__(self, embedding_net):
        super(SiameseNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)


def AverageNonzeroTripletsMetric(eval_embedding, eval_labels):
    '''
    Counts average number of nonzero triplets found in minibatches
    '''
    _ = loss_fn(eval_embedding, eval_labels)
    return _[1]


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


def pdist(vectors):
    """
    Calculate distance of each vector to each other vectors using Euclidean distance or L2 Norm
    Args:
        vector: (torch.tensor) first dim is equal to number of users, second dim is equal to number of feature embedding dims.
    """
    distance_matrix = -2 * vectors.mm(torch.t(vectors)) + vectors.pow(2).sum(dim=1).view(1, -1) + vectors.pow(2).sum(
        dim=1).view(-1, 1)
    return distance_matrix


class TripletSelector:
    """
    Implementation should return indices of anchors, positive and negative samples
    return np array of shape [N_triplets x 3]
    """

    def __init__(self):
        pass

    def get_triplets(self, embeddings, labels):
        raise NotImplementedError


class AllTripletSelector(TripletSelector):
    """
    Returns all possible triplets
    May be impractical in most cases
    """

    def __init__(self):
        super(AllTripletSelector, self).__init__()

    def get_triplets(self, embeddings, labels):
        labels = labels.cpu().data.numpy()
        triplets = []
        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs

            # Add all negatives for all positive pairs
            temp_triplets = [[anchor_positive[0], anchor_positive[1], neg_ind] for anchor_positive in anchor_positives
                             for neg_ind in negative_indices]
            triplets += temp_triplets

        return torch.LongTensor(np.array(triplets))


def hardest_negative(loss_values):
    hard_negative = np.argmax(loss_values)
    return hard_negative if loss_values[hard_negative] > 0 else None


def random_hard_negative(loss_values):
    hard_negatives = np.where(loss_values > 0)[0]
    return np.random.choice(hard_negatives) if len(hard_negatives) > 0 else None


def semihard_negative(loss_values, margin):
    semihard_negatives = np.where(np.logical_and(loss_values < margin, loss_values > 0))[0]
    return np.random.choice(semihard_negatives) if len(semihard_negatives) > 0 else None


class FunctionNegativeTripletSelector(TripletSelector):
    """
    For each positive pair, takes the hardest negative sample (with the greatest triplet loss value) to create a triplet
    Margin should match the margin used in triplet loss.
    negative_selection_fn should take array of loss_values for a given anchor-positive pair and all negative samples
    and return a negative index for that pair
    """

    def __init__(self, margin, negative_selection_fn, cpu=True):
        super(FunctionNegativeTripletSelector, self).__init__()
        self.cpu = cpu
        self.margin = margin
        self.negative_selection_fn = negative_selection_fn

    def get_triplets(self, embeddings, labels):
        if self.cpu:
            embeddings = embeddings.cpu()
        distance_matrix = pdist(embeddings)
        distance_matrix = distance_matrix.cpu()

        labels = labels.cpu().data.numpy()
        triplets = []

        for label in set(labels):
            label_mask = (labels == label)
            label_indices = np.where(label_mask)[0]
            if len(label_indices) < 2:
                continue
            negative_indices = np.where(np.logical_not(label_mask))[0]
            anchor_positives = list(combinations(label_indices, 2))  # All anchor-positive pairs
            anchor_positives = np.array(anchor_positives)

            ap_distances = distance_matrix[anchor_positives[:, 0], anchor_positives[:, 1]]
            for anchor_positive, ap_distance in zip(anchor_positives, ap_distances):
                loss_values = ap_distance - distance_matrix[torch.LongTensor(np.array([anchor_positive[0]])), torch.LongTensor(negative_indices)] + self.margin
                loss_values = loss_values.data.cpu().numpy()
                hard_negative = self.negative_selection_fn(loss_values)
                if hard_negative is not None:
                    hard_negative = negative_indices[hard_negative]
                    triplets.append([anchor_positive[0], anchor_positive[1], hard_negative])

        if len(triplets) == 0:
            triplets.append([anchor_positive[0], anchor_positive[1], negative_indices[0]])

        triplets = np.array(triplets)

        return torch.LongTensor(triplets)


def HardestNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                 negative_selection_fn=hardest_negative,
                                                                                 cpu=cpu)


def RandomNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                negative_selection_fn=random_hard_negative,
                                                                                cpu=cpu)


def SemihardNegativeTripletSelector(margin, cpu=False): return FunctionNegativeTripletSelector(margin=margin,
                                                                                  negative_selection_fn=lambda x: semihard_negative(x, margin),
                                                                                  cpu=cpu)


def accuracy(outputs, labels):
    """
    Compute the accuracy, given the outputs and labels for all images.

    Args:
        outputs: (np.ndarray) dimension batch_size x 6 - log softmax output of the model
        labels: (np.ndarray) dimension batch_size, where each element is a value in [0, 1, 2, 3, 4, 5]

    Returns: (float) accuracy in [0,1]
    """
    outputs = np.argmax(outputs, axis=1)
    return np.sum(outputs==labels)/float(labels.size)


def pdist_diff_shape(vectors1, vectors2):
    '''
    Calculate distance matrix using euclidean distance with vectorization implementation.
    
    Args:
        vectors1: (torch.tensor) dimension m_samples x n dimension of feature vectors
        vectors2: (torch.tensor) dimension l_samples x n dimension of feature vectors

    Returns: (torch.tensor) dimension m_samples x n_samples
    '''
    distance_matrix = -2 * vectors1.mm(torch.t(vectors2)) + vectors2.pow(2).sum(dim=1).view(1, -1) + vectors1.pow(2).sum(dim=1).view(-1, 1)
    return distance_matrix


def eer(enrolled_embedding, enrolled_labels, eval_embedding, eval_labels, device):
    '''
    Calculate equal error rate
    
    Args:
        enrolled_embedding : (torch.tensor) dimension m_samples x n dimension of feature vectors
        enrolled_labels: (torch.tensor) dimension m_samples labels
        eval_embedding: (torch.tensor) dimension l_samples x n dimension of feature vectors
        eval_labels: (torch.tensor) dimension l_samples labels
        device: (str) cpu or cuda:i

    Returns: eer and corresponding threshold
    '''
    same_mask = torch.zeros((eval_labels.size()[0], enrolled_labels.size()[0]), dtype=bool).to(device)
    diff_mask = torch.zeros((eval_labels.size()[0], enrolled_labels.size()[0]), dtype=bool).to(device)
    for i in range(len(eval_labels)):
        indices = torch.where(enrolled_labels == eval_labels[i])[0].tolist()
        same_mask[i, indices] = True
        indices = torch.where(enrolled_labels != eval_labels[i])[0].tolist()
        diff_mask[i, indices] = True 
    distances = pdist_diff_shape(eval_embedding,enrolled_embedding).unsqueeze(0).to(device)
    threshold = torch.arange(0,2.01,0.01).view(-1, 1, 1).to(device)
    distances = distances-threshold
    # get list of FRR
    masked_distances = distances.detach().clone().to(device)
    masked_distances[masked_distances>=0]=1
    masked_distances = torch.mul(masked_distances, same_mask.unsqueeze(0))
    masked_distances[masked_distances!=1]=0
    count_FR = torch.count_nonzero(masked_distances, dim=-1)
    count_FR = torch.count_nonzero(count_FR, dim=-1)
    FRRs = count_FR/(len(eval_labels))
    # get list of FAR
    _masked_distances = distances.detach().clone().to(device)
    _masked_distances[_masked_distances<=0]=1
    _masked_distances = torch.mul(_masked_distances, diff_mask.unsqueeze(0))
    _masked_distances[_masked_distances!=1]=0
    count_FA = torch.sum(_masked_distances.sum(dim=-1), dim=(-1))
    FARs = count_FA/(diff_mask.count_nonzero().unsqueeze(0))
    er_diff = torch.abs(FARs-FRRs).to(device)
    idx_threshold = torch.argmin(er_diff)
    EER = (FARs[idx_threshold]+FRRs[idx_threshold])/2
    min_threshold = idx_threshold*0.01
    eer = EER.detach().clone().to("cpu").item()*100
    min_thresh = min_threshold.detach().clone().to("cpu").item()
    for item in [min_threshold, EER, idx_threshold, er_diff, FARs, count_FA, _masked_distances, 
                 FRRs, count_FR, masked_distances, distances, threshold, indices, 
                 same_mask, diff_mask]:
        del item 
    if not(device=="cpu"):
        torch.cuda.empty_cache()
    return round(eer, 2), round(min_thresh,4)


margin = 1
loss_fn = OnlineTripletLoss(margin, RandomNegativeTripletSelector(margin))


# maintain all metrics required in this dictionary- these are used in the training and evaluation loops
metrics = {
    'EER': eer, 'Average Non Zero': AverageNonzeroTripletsMetric
    # could add more metrics such as accuracy for each token type
}
