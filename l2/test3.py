import mindspore
from mindspore import nn
from mindspore import ops
from mindspore import SummaryRecord

def get_model():
  ### define input
  n_centers = 3200
  n_vertex = 1
  dim = 2
  tree_degree = 3
  centers_dtype = mindspore.int32
  vec_dtype = mindspore.uint8
  op_size = 10
  tp_size = 102

  centers = mindspore.ops.randint(0, 255, (n_centers,dim), dtype=mindspore.int32)
  for i in range(dim):
    centers[0][i] = 999
    centers[-1][i] = 999

  centers = mindspore.ops.cast(centers, centers_dtype)
  print("centers = ", centers)

  vector = mindspore.ops.randint(0, 255, (n_vertex,dim), dtype=mindspore.int32)
  vector = mindspore.ops.cast(vector, vec_dtype)
  print("vector = ", vector)

  def build_tree():
    tree_data = [[0]*tree_degree]

    start_index = 0
    while start_index < len(centers):
      next_max = min(start_index + tree_degree, len(centers))
      item = [2+i for i in range(start_index, next_max)]
      item += [0 for i in range(next_max, start_index + tree_degree)]
      start_index = next_max
      tree_data.append(item)

    tree_data.append([0]*tree_degree)
    return mindspore.Tensor(tree_data, dtype=mindspore.int32)

  tree = build_tree()
  print("tree = ", tree)

  open_list = mindspore.Tensor([1] + [0]*(op_size-1), dtype=mindspore.int32)
  print("open_list = ", open_list)

  ### build model
  class TSModel(nn.Cell):
    def __init__(self):
      super().__init__()
      self.tp_size = tp_size
      self.dim = dim
      self.op_size = op_size
      self.tree_degree = tree_degree

    def calc_l2(self, a, b):
      # print("calc_l2 === === === === === ===")
      d = a - b
      d2 = ops.square(d)
      # print("a = ", a, "b = ", b, "d = ", d, "d2 = ", d2)
      ls = ops.sum(d2, dim=1, keepdim=False)
      # print("dist = ", ls)
      return ls

    def process_open_list(self, centers, tree, vector, open_list):
      # print("process_open_list")
      # print(open_list)
      # print(tree)
      child_indexes = ops.gather(tree, open_list, 0)
      # print(child_indexes, centers)
      selected_centers = ops.gather(centers, child_indexes, 0)
      selected_centers = ops.reshape(selected_centers, (self.op_size*self.tree_degree,self.dim))
      # print(selected_centers)

      ls = self.calc_l2(selected_centers, vector)
      tops, indexes = ops.topk(ls, self.op_size, largest=False, sorted=True)
      child_indexes_flat = ops.reshape(child_indexes, (self.op_size*self.tree_degree,))

      # print(child_indexes_flat)
      # print(indexes)
      indexes = ops.gather(child_indexes_flat, indexes, 0)
      return indexes, tops

    def update_top_list(self, top_list, top_dist, cand_top_list, cand_top_dist):
      # print(top_dist.dtype, cand_top_dist.dtype)
      cand_top_dist = ops.cast(cand_top_dist, dtype=mindspore.int32)
      tot_dist  = mindspore.ops.cat((top_dist, cand_top_dist))
      tot_index = mindspore.ops.cat((top_list, cand_top_list))

      new_tops, indexes = ops.topk(tot_dist, self.tp_size, largest=False, sorted=True)
      new_top_list = ops.gather(tot_index, indexes, 0)
      return new_top_list, new_tops

    def update_open_list(self, tree, cand_top_list):
      open_list = mindspore.ops.clip(cand_top_list, 0, tree.shape[0]-1)
      return open_list

    def construct(self, centers, tree, vector_uint8, open_list):
      top_list = mindspore.Tensor([0]*self.tp_size, dtype=mindspore.int32)
      top_dist = mindspore.Tensor([999999]*self.tp_size, dtype=mindspore.int32)

      vector = ops.cast(vector_uint8, dtype=mindspore.int32)

      # cand_top_list, cand_top_dist = self.process_open_list(centers, tree, vector, open_list)
      # top_list, top_dist = self.update_top_list(top_list, top_dist, cand_top_list, cand_top_dist)
      # open_list = self.update_open_list(tree, cand_top_list)

      max_loops = 10
      main_loop = ops.WhileLoop()

      #@mindspore.jit
      def main_loop_fn(args):
        print("main_loop_fn ====")
        #print(args)
        open_list, top_list, top_dist = args
        cand_top_list, cand_top_dist = self.process_open_list(centers, tree, vector, open_list)
        top_list, top_dist = self.update_top_list(top_list, top_dist, cand_top_list, cand_top_dist)
        open_list = self.update_open_list(tree, cand_top_list)

        print(open_list, top_list, top_dist)
        return open_list, top_list, top_dist

      #@mindspore.jit
      def cond_func(args):
        open_list, top_list, top_dist = args
        # print("LOOP:", max_loops)
        return max(open_list) != 0

      # top_list, top_dist, open_list, max_loops = main_loop(cond_func, main_loop_fn, main_loop_init_state)
      main_loop_init_state = (open_list, top_list, top_dist)
      open_list, top_list, top_dist = main_loop_fn(main_loop_init_state)

      if cond_func((open_list, top_list, top_dist)):
        main_loop_init_state = (open_list, top_list, top_dist)
        open_list, top_list, top_dist = main_loop_fn(main_loop_init_state)

      return top_list, top_dist, open_list


  model = TSModel()
  inputs = (centers, tree, vector, open_list)
  output = model(*inputs)
  print("output = ", output)

  if True:
    mindspore.set_context(mode=mindspore.GRAPH_MODE, save_graphs=True)
    with SummaryRecord('./summary_dir/summary_04', network=model) as summary_record:
      for i in range(10):
        top_list, top_dist, open_list = model(*inputs)
        summary_record.add_value('scalar', 'loss', sum(open_list))
        summary_record.record(i)
      exit()

  ### save model
  mindspore.export(model, *inputs, file_name="ntree_search", file_format="MINDIR")

  mindspore.export(model, *inputs, file_name="ntree_search", file_format="ONNX")

def test_cpu():
  mindspore.set_device(device_target='CPU')
  mindspore.run_check()

get_model()
