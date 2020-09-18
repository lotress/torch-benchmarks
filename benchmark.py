import torch
import torch.nn as nn
from time import perf_counter

types = (torch.float, torch.half)
inTensor = torch.randn((16, 1088, 1920))
devCount = torch.cuda.device_count()
devices = [torch.device('cuda:{}'.format(i)) for i in range(devCount)]
devInd = [i for i in range(devCount)]
times = 10
N = 1000
identity = lambda x, *_: x
dtype = None

def run(cases, *args, times=1):
  timing = [0 for _ in cases]
  results = [0 for _ in cases]
  for _ in range(times):
    for j, (f0, f) in enumerate(cases):
      x = f0(inTensor, *args)
      start = perf_counter()
      res = f(x, *args)
      results[j] = res
      timing[j] += perf_counter() - start
      torch.cuda.empty_cache()
  return timing, results

class Model(nn.Module):
  def __init__(self):
    super(Model, self).__init__()

  def forward(self, x):
    return x + fx(x).to(x.dtype)

fx = lambda x: torch.bmm(x, x.transpose(1, 2)).mean(dtype=torch.float)
scatter = lambda x: torch.cuda.comm.scatter(x, devInd)
gather = lambda t:torch.cuda.comm.gather(t, destination=-1) # destination=CPU

net = torch.nn.DataParallel(Model().to(devices[0]), device_ids=list(range(devCount)))
ms = [Model().to(dev) for dev in devices]

def f0(t):
  for _ in range(N):
    xs = [ms[i](t[i]) for i in range(devCount)]
  xs = [x.mean().to(torch.float).unsqueeze(0) for x in xs]
  return gather(xs).mean()

def f1(t):
  xs = []
  for i in range(devCount):
    x = t[i]
    for _ in range(N):
      x = ms[i](x)
    xs.append(x.mean().to(torch.float).unsqueeze(0))
  return gather(xs).mean()

def f2(t):
  x = t
  for _ in range(N):
    x = net(x)
  return x.mean(dtype=torch.float).cpu()

def f3(t):
  x = t.to(devices[0])
  for _ in range(N):
    x = ms[0](x)
  return x.mean(dtype=torch.float).cpu()

g1 = lambda x: x.to(dtype)
g0 = lambda x: x.to(dtype=dtype, device=devices[0])
g2 = lambda x: scatter(g1(x))

#cases = [(g2, f0), (g2, f1), (g1, f2)] if devCount > 1 else []
cases = [(g2, f0)] if devCount > 1 else []
#cases.append((g0, f3))

print([torch.cuda.get_device_properties(i) for i in range(devCount)])

run(cases)

for t in types:
  dtype = t
  print(dtype)
  time, _ = run(cases, times=times)
  print(time)