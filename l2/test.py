import torch
import time

def st_time(func):
    """
        st decorator to calculate the total time of a func
    """

    def st_func(*args, **keyArgs):
        t1 = time.time()
        r = func(*args, **keyArgs)
        t2 = time.time()
        print("Function=%s, Time=%s" % (func.__name__, t2 - t1))
        return r

    return st_func

@st_time
def all_pairs_euclid_torch(A, B):
    AB = torch.mm(A, B.t())
    # print("AB", AB.shape, AB)

    # AA = torch.pow(A, 2)
    # print("AA", AA.shape, AA)
    AA = torch.sum(torch.pow(A, 2), 1, keepdim=True)
    # print("AA", AA.shape, AA)
    AA = AA.expand(A.shape[0], B.shape[0])
    # print("AA", AA.shape, AA)

    BB = torch.sum(torch.pow(B, 2), 1, keepdim=True)
    # print("BB", BB.shape, BB)
    BB = BB.expand(B.shape[0], A.shape[0]).t()
    # print("BB", BB.shape, BB)

    return AA - 2*AB + BB

@st_time
def all_pairs_euclid_torch2(A, B):
    # https://github.com/pytorch/pytorch/blob/ce9ba071fd29013e72100dd97728d01c860720d9/aten/src/ATen/native/Distance.cpp#L66
    AA = torch.sum(torch.pow(A, 2), 1, keepdim=True)
    BB = torch.sum(torch.pow(B, 2), 1, keepdim=True)

    A_pad = torch.ones_like(AA)
    B_pad = torch.ones_like(BB)

    Ax = torch.cat((-2*A, AA, A_pad), -1)
    Bx = torch.cat((B, B_pad, BB), -1).t()

    #print(A.shape, AA.shape, A_pad.shape)
    #print("Ax", Ax.shape, Ax)

    return torch.mm(Ax, Bx)

dim = 1024
n_centers = 12
n_vectors = 8

centers = torch.rand(n_centers, dim)
vectors = torch.rand(n_vectors, dim)

distances = all_pairs_euclid_torch(centers, vectors)
print(distances)

distances = all_pairs_euclid_torch2(centers, vectors)
print(distances)
