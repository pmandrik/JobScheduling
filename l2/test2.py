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

def calc_l2_cros(A, B):
    AA = torch.sum(torch.pow(A, 2), 1, keepdim=True)
    BB = torch.sum(torch.pow(B, 2), 1, keepdim=True)

    A_pad = torch.ones_like(AA)
    B_pad = torch.ones_like(BB)

    Ax = torch.cat((-2*A, AA, A_pad), -1)
    Bx = torch.cat((B, B_pad, BB), -1).t()
    return torch.mm(Ax, Bx)

def calc_l2(A, B):
  # print(A)
  # print(B)
  D = A - B
  D2 = torch.pow(D, 2)
  # print("D2 = ", D2, D2.shape)
  dist = torch.sum(D2, (0,1))
  # print("dist = ", dist)
  return dist

dim = 2
n_centers = 3
n_vectors = 600

centers = torch.rand(n_centers, dim)
counts  = torch.zeros(n_centers).view(n_centers, 1)
vectors = torch.rand(0, dim)

for i in range(n_centers):
  nv = n_vectors//n_centers
  vpart = torch.rand(nv, dim)
  vpart += 5*i*torch.ones_like(vpart)
  vectors = torch.cat((vectors, vpart), 0)

def kmean_iter(centers, vectors, counts):
  l2 = calc_l2_cros(centers, vectors)

  # print("centers = ", centers)
  # print("vectors = ", vectors)
  print("counts = ", counts, counts.shape)
  print("l2 = ", l2)
  lfactor = 0.02*3/n_centers

  fact = 0.0001*torch.rand_like(l2)
  l2c = l2 + lfactor * counts + fact
  #print(fact)
  print("l2c = ", l2c)

  x = (l2c == torch.min(l2c, 0, True)[0]).float()

  print(x, x.shape)
  Ax = x.view(n_centers, n_vectors, 1)
  # print(Ax, Ax.shape)

  vAx = Ax * vectors
  #print(vAx, vAx.shape)

  sum_vAx = torch.sum(vAx, dim=1)
  # print("sum_vAx = ", sum_vAx, sum_vAx.shape)

  counts = torch.count_nonzero(vAx[:,:,:1], dim=1)
  #print("counts = ", counts, counts.shape)
  counts[counts==0] = 1

  centroids = sum_vAx / counts
  # print("centroids = ", centroids, centroids.shape)

  dist = calc_l2(centers, centroids)

  return centroids, counts, dist

for i in range(20):
  print("========================", i)
  centers, counts, dist = kmean_iter(centers, vectors, counts)
  #print("centers = ", centers)
  print("counts = ", counts)
  print("dist = ", dist)
  if dist < 0.1:
    centers = torch.cat((centers, 0.95*centers), 0)
    counts = 0.5*counts
    counts = torch.cat((counts, counts), 0)
    n_centers = n_centers*2

#exit()
#print(vectors)
print(centers)
import matplotlib.pyplot as plt
plt.plot(vectors[:,0], vectors[:,1], 'o')
plt.plot(centers[:,0], centers[:,1], 'x')
plt.show()








