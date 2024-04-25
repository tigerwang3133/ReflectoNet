import torch
from torch.nn import functional as F
from torch.nn.modules import Module


class PrototypicalLoss(Module):

    def __init__(self, n_support):
        super(PrototypicalLoss, self).__init__()
        self.n_support = n_support

    def forward(self, input, target):
        return prototypical_loss(input, target, self.n_support)


def euclidean_dist(x, y):

    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    if d != y.size(1):
        raise Exception

    x = x.unsqueeze(1).expand(n, m, d)
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)


def prototypical_loss(input, target, n_support):
    target_cuda = target.to('cuda')
    input_cuda = input.to('cuda')

    def supp_idxs(c):
        return target_cuda.eq(c).nonzero()[:n_support].squeeze(1)

    classes = torch.unique(target_cuda)
    n_classes = len(classes)

    n_query = target_cuda.eq(classes[0].item()).sum().item() - n_support

    support_idxs = list(map(supp_idxs, classes))

    prototypes = torch.stack([input_cuda[idx_list].mean(0) for idx_list in support_idxs])  # 3x3
    # print(prototypes.size())

    # print('target cuda: ', target_cuda, list(map(lambda c: target_cuda.eq(c).nonzero()[n_support:], classes)))
    query_idxs = torch.stack(list(map(lambda c: target_cuda.eq(c).nonzero()[n_support:], classes))).view(-1)

    query_samples = input.to('cuda')[query_idxs]
    dists = euclidean_dist(query_samples, prototypes)

    # print(query_samples, query_idxs)
    # print(dists.size())
    # print(n_classes, n_query)
    log_p_y = F.log_softmax(-dists, dim=1).view(n_classes, n_query, -1)

    target_inds = torch.arange(0, n_classes).cuda()
    target_inds = target_inds.view(n_classes, 1, 1)
    target_inds = target_inds.expand(n_classes, n_query, 1).long()

    loss_val = -log_p_y.gather(2, target_inds).squeeze().view(-1).mean()
    _, y_hat = log_p_y.max(2)
    acc_val = y_hat.eq(target_inds.squeeze()).float().mean()
    # print(y_hat.size(), y_hat)
    # print(target_inds.size(), target_inds.squeeze())
    return loss_val, acc_val
