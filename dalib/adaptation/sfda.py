import torch
import torch.nn.functional as F


class DatasetIndex(torch.utils.data.Dataset):
    """ pytorch Dataset that return image index too"""
    def __init__(self, dt):
        self.dt = dt

    def __getitem__(self, index):
        data, target = self.dt[index]
        return data, target, index

    def __len__(self):
        return len(self.dt)


class SubsetImagesPlabels(torch.utils.data.Dataset):
    def __init__(self, dt, indices, plabels):
        self.dt = dt
        self.indices = indices
        self.plabels = plabels
        self.num_classes = dt.num_classes

    def __getitem__(self, index):
        img, _ = self.dt[self.indices[index]]
        plabel = self.plabels[index]
        return img, plabel

    def update_plabels(self, plabels):
        self.plabels = plabels

    def __len__(self):
        return len(self.indices)


class PseudoPartialLabelDisambiguation():
    """
    src hard label, trg soft label
    """

    def __init__(self, n_sample, feat_dim, n_class, device, coeff_cent=0.9, coeff_prob=0.9):
        self.n_class = n_class
        self.device = device
        self.coeff_cent = coeff_cent
        self.coeff_prob = coeff_prob

        self.centroids = torch.zeros((n_class, feat_dim)).to(self.device)     # to be updated
        self.probs = torch.zeros((n_sample, n_class)).to(self.device)


    def update_parameters(self, src_features, src_labels, trg_features, trg_plabels, data_idx):
        with torch.no_grad():

            #------ update centroids -----#
            # source
            # src_features_ = torch.cat((src_features, torch.ones((src_features.size(0), 1)).to(self.device)), dim=1)
            src_features_norm = F.normalize(src_features, dim=1)
            pred_onehot = torch.eye(self.n_class)[src_labels].to(self.device)
            centroids_add1 = (src_features_norm.T @ pred_onehot).T

            # target
            # trg_features_ = torch.cat((trg_features, torch.ones((trg_features.size(0), 1)).to(self.device)), dim=1)
            trg_features_norm = F.normalize(trg_features, dim=1)
            pred_onehot = torch.eye(self.n_class)[trg_plabels].to(self.device)
            centroids_add2 = (trg_features_norm.T @ pred_onehot).T
            # centroids = features_norm.T @ pred_onehot / (pred_onehot.sum(dim=0) + 1e-5)
            centroids = self.coeff_cent * self.centroids + (1 - self.coeff_cent) * (centroids_add1 + centroids_add2)
            self.centroids = F.normalize(centroids, dim=1)

            #------ update probs -----#
            outputs = trg_features_norm @ self.centroids.T
            predicts = outputs.argmax(dim=1)
            pred_onehot = torch.eye(self.n_class)[predicts].to(self.device)
            probs = self.coeff_prob * self.probs[data_idx] + (1 - self.coeff_prob) * pred_onehot
            self.probs[data_idx] = probs

            return probs
