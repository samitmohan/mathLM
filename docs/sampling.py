# top k
import torch
import torch.nn.functional as F


def topK(logits, k=5):
    topK_val, _ = torch.topk(logits, k)
    min_val = topK_val[-1]
    logits[logits < min_val] = float("-inf")
    probs = F.softmax(logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def topP(logits, p=0.9):
    probs = F.softmax(logits, dim=-1)
    sortedProb, sortedIdx = torch.sort(probs, descending=True)

    cumulativeSum = torch.cumsum(sortedProb, dim=-1)

    remove = cumulativeSum - sortedProb > p
    sortedProb[remove] = 0.0

    sample = torch.multinomial(sortedProb, num_samples=1)
    return sortedIdx[sample]


def main():
    logits = torch.tensor([2.1, 0.5, 1.8, 0.1, 3.0])
    k = 3
    idx = topK(logits, k)
    print(logits[idx])
    idx_topP = topP(logits, p=0.9)
    print(logits[idx_topP])


main()
