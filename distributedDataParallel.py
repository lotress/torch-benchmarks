import torch
import torch.nn as nn
import torch.multiprocessing as mp
import torch.nn.functional as F
import torch.optim as optim
from time import perf_counter
from torch.distributed import init_process_group
from torch.nn.parallel import DistributedDataParallel as DDP

types = (torch.float, torch.half)
devs = (0, 1, 2, 3)
nprocs = len(devs)
devices = [torch.device('cuda:{}'.format(i)) for i in devs]
N = 10
lr = 0.0004
sharedfile = 'file:///home/lotress/sharedfile'
identity = lambda x, *_: x

def run(cases, *args, times=1):
  timing = [0 for _ in cases]
  results = [0 for _ in cases]
  for _ in range(times):
    for j, (f0, f) in enumerate(cases):
      x = f0(*args)
      start = perf_counter()
      res = f(x, *args)
      results[j] = res
      timing[j] += perf_counter() - start
      torch.cuda.empty_cache()
  return timing, results

mish = lambda x: x * F.softplus(x).tanh()
class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()
    self.f0 = nn.Conv2d(4, 64, 3, 1, padding=1, bias=False)
    self.fs = nn.ModuleList([nn.Conv2d(64, 64, 3, 1, padding=1, bias=False) for _ in range(8)])
    self.f1 = nn.Conv2d(64, 4, 3, 1, padding=1)
    self.act = mish

  def forward(self, x):
    x = self.act(self.f0(x))
    for f in self.fs:
      x = self.act(f(x)) + x
    return self.f1(x)

def f(x, *_):
  for _ in range(N):
    out = model(x)
    loss = F.mse_loss(out, x)
    loss.backward()
    optimizer.step()
  print(float(loss.cpu()))

g1 = lambda bsz: lambda i, dtype: torch.randn((bsz, 4, 256, 256), dtype=dtype, device=devices[i])
case = [(g1(4), f)]

def init(i, t):
  global model, optimizer
  model = DDP(Model().cuda().to(t), device_ids=[devs[i]])
  optimizer = optim.SGD(model.parameters(), lr=lr)
  return model, optimizer

def main(i):
  init_process_group('nccl', world_size=nprocs, rank=i, init_method=sharedfile)
  torch.cuda.set_device(devs[i])
  init(i, types[0])
  run(case, i, types[0])
  for t in types:
    init(i, t)
    time, _ = run(case, i, t, times=10)
    print(i, t, time[0])

def single():
  global model, optimizer
  torch.cuda.set_device(devs[-1])
  case = [(g1(16), f)]
  t = types[1]
  model = Model().cuda().to(t)
  optimizer = optim.SGD(model.parameters(), lr=lr / nprocs)
  time, _ = run(case, 3, t, times=10)
  print(3, t, time[0])

if __name__ == '__main__':
  mp.spawn(main, nprocs=nprocs, join=True)
  single()