import numpy as np
import torch


class ProtoBatchSampler(object):
    '''    
    ProtoBatchSampler: yield a batch of indexes at each iteration.    
    '''

    def __init__(self, labels, num_support, num_query, iterations, classes_in=5, classes_out=5):
        '''
        Initialize the ProtoBatchSampler object
        Args:
        - labels: an iterable containing all the labels for the current dataset
        - classes_per_it: number of random classes for each iteration
        - 'num_support': number of support samples for in-distribution classes
        - 'num_query': number of query samples for in(out)-distribution classes
        - iterations: number of iterations (episodes) per epoch
        '''
        super(ProtoBatchSampler, self).__init__()
        self.labels = labels
        self.classes_in = classes_in
        self.classes_out = classes_out
        self.sample_support = num_support
        self.sample_query = num_query
        self.iterations = iterations

        self.classes, self.counts = np.unique(self.labels, return_counts=True)
        self.classes = torch.LongTensor(self.classes)

        # create a matrix, indices, of dim: classes X max(elements per class)
        # fill it with nans
        # for every class c, fill the relative row with the indices samples belonging to c
        # in numel_per_class we store the number of samples for each class/row
        self.idxs = range(len(self.labels))
        self.indices = np.empty((len(self.classes), max(self.counts)), dtype=int) * np.nan
        self.indices = torch.Tensor(self.indices)
        self.numel_per_class = torch.zeros_like(self.classes)
        for idx, label in enumerate(self.labels):
            label_idx = np.argwhere(self.classes == label).item()
            self.indices[label_idx, np.where(np.isnan(self.indices[label_idx]))[0][0]] = idx
            self.numel_per_class[label_idx] += 1
        # print(self.numel_per_class)

    def __iter__(self):
        '''
        yield a batch of indexes
        '''
        c_i = self.classes_in
        c_o = self.classes_out
        s = self.sample_support
        q = self.sample_query

        for it in range(self.iterations):
            batch_size = (s + 2*q) * c_i 
            batch = torch.LongTensor(batch_size)
            c_idxs = torch.randperm(len(self.classes))[:c_i+c_o]
            c_idxs_i = c_idxs[:c_i]
            c_idxs_o = c_idxs[c_i:]
            for i, c in enumerate(self.classes[c_idxs_i]):
                label_idx = torch.arange(len(self.classes)).long()[self.classes == c].item()
                sl_i = slice(i * (s+2*q), i* (s+2*q) + (s+q))
                sample_idxs_in = torch.randperm(self.numel_per_class[label_idx])[:s+q]
                batch[sl_i] = self.indices[label_idx][sample_idxs_in]
                sample_cls_out = torch.tensor(np.random.choice(self.classes[c_idxs_o], q)).long()
                k = i* (s+2*q) + (s+q)
                for j in range(sample_cls_out.shape[0]):
                    c_ = sample_cls_out[j]
                    label_idx_o = torch.arange(len(self.classes)).long()[self.classes == c_].item()
                    idx_out = torch.randperm(self.numel_per_class[label_idx_o])[:1]
                    batch[k] = self.indices[label_idx_o][idx_out]
                    k = k + 1

            batch = self._reshuffle(batch)

            yield batch    

    def __len__(self):
        '''
        returns the number of iterations (episodes) per epoch
        '''
        return self.iterations

    def _reshuffle(self, batch):
        '''
        returns batch as [s_1,...,s_n,q_1+q_o,...,q_n+q_o]
        '''
        c_i = self.classes_in
        s = self.sample_support
        q = self.sample_query
        batch_size = (s + 2*q) * c_i 
        new_batch = torch.LongTensor(batch_size)
        i = j = 0
        for k in range(c_i):
            new_batch[i:i+s] = batch[j:j+s]
            i += s
            j += (s + 2*q)
        j = s
        for k in range(c_i):
            new_batch[i:i+2*q] = batch[j:j+2*q]
            i += 2*q
            j += (2*q + s)

        return new_batch


def softCrossEntropy(logits, target):
    logprobs = torch.nn.functional.log_softmax (logits, dim = 1)
    return  -(target * logprobs).sum() / input.shape[0]
