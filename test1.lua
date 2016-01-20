--require 'cutorch'
require 'warp_ctc'
acts = torch.Tensor({{0,0,0,0,0}}):float()
grads = torch.Tensor():float()
labels = {{1}}
sizes ={1}
cpu_ctc(acts, grads, labels, sizes)

acts = torch.Tensor({{1,2,3,4,5},{6,7,8,9,10},{11,12,13,14,15}}):float()
labels = {{3,3}}
sizes = {3}
cpu_ctc(acts, grads, labels, sizes)


acts = torch.Tensor({{-5,-4,-3,-2,-1},{-10,-9,-8,-7,-6},{-15,-14,-13,-12,-11}}):float()
labels = {{2,3}}
sizes = {3}
cpu_ctc(acts, grads, labels, sizes)

acts = torch.Tensor({{0,0,0,0,0},{1,2,3,4,5},{-5,-4,-3,-2,-1},
                        {0,0,0,0,0},{6,7,8,9,10},{-10,-9,-8,-7,-6},
                        {0,0,0,0,0},{11,12,13,14,15},{-15,-14,-13,-12,-11}}):float()
labels = {{1}, {3,3}, {2,3}}
sizes = {1,3,3}
cpu_ctc(acts, grads, labels, sizes)
