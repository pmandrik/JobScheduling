
import random
def shuffle_slice(a, start, stop):
  i = start
  while (i < stop-1):
    idx = random.randrange(i, stop)
    a[i], a[idx] = a[idx], a[i]
    i += 1

class Cluster:
  def __init__(self, start, end):
    self.start = start
    self.end = end

    self.centers = []

  def n_vecs(self):
    return self.end - self.start

  def __repr__(self):
    return f"({self.start},{self.end})"

def try_batch_clustering(items, indexes, vectors):
  clusters = []

  # setup centers
  min_size = 10
  for item in items:
    shuffle_slice(indexes, item.start, item.end)
    n_centers = min(len(item.n_vecs)//min_size, 32)
    for i in range(n_centers):
      index = indexes[item.start + i]
      item.centers.append(vectors[index])

  # cluster
  a = []

  return clusters

def try_clustering_cpu(item, indexes, vectors):
  return try_batch_clustering([item], indexes, vectors)

def do_clustering(queue, indexes):
  answer = []
  queue.sort(key=lambda x: x.n_vecs())
  # queue.sort(key=lambda x: x.end)

  batch = []
  max_b_size = 10000
  b_size = 0

  while len(queue):
    item = queue[0]

    n_vecs = item.n_vecs()
    if n_vecs + b_size > max_b_size:
      break

    queue.pop(0)
    batch.append(item)
    b_size += n_vecs

  if len(batch) == 0:
    item = queue.pop(0)
    return try_clustering_cpu(item, indexes, vectors)

  return try_batch_clustering(batch, indexes, vectors)


def kmean_iterative(vectors):
  indexes = [i for i in range(len(vectors))]

  queue = [Cluster(0,len(indexes))]

  answer = []
  min_size = 10
  while len(queue):
    clusters = do_clustering(queue, indexes, vectors)

    for cluster in clusters:
      if cluster.n_vecs() < min_size:
        answer.append(cluster)
      else:
        queue.append(cluster)

  answer.sort(key=lambda x: x.end)
  return answer

# given N vectors
# cluster iterativly using kmean


vectors = [i for i in range(10000)]
clusters = kmean_iterative(vectors)
print(clusters)
