# 2022.06.27-Changed for building SNN-MLP
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
from torch.autograd import Function
import torch
from torch.nn.modules.utils import _pair
import torch.nn.functional as F
import torch.nn as nn

from collections import namedtuple
import cupy
from string import Template
import math

Stream = namedtuple('Stream', ['ptr'])


def Dtype(t):
    if isinstance(t, torch.cuda.FloatTensor):
        return 'float'
    elif isinstance(t, torch.cuda.DoubleTensor):
        return 'double'


@cupy._util.memoize(for_each_device=True)
def load_kernel(kernel_name, code, **kwargs):
    code = Template(code).substitute(**kwargs)
    kernel_code = cupy.cuda.compile_with_cache(code)
    return kernel_code.get_function(kernel_name)


CUDA_NUM_THREADS = 1024

kernel_loop = '''
#define CUDA_KERNEL_LOOP(i, n)                        \
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; \
      i < (n);                                       \
      i += blockDim.x * gridDim.x)
'''


def GET_BLOCKS(N):
    return (N + CUDA_NUM_THREADS - 1) // CUDA_NUM_THREADS


_lif_kernel_h = kernel_loop + '''
extern "C"
__global__ void lif_forward_kernel_h(
const ${Dtype}* bottom_data, const ${Dtype}* tau, const ${Dtype}* vth, ${Dtype}* top_data, bool* flag, ${Dtype}* o) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int s = ${lif};
    const int lifh = (${height} + s - 1) / s;
    const int lifhh = ${height} / s;
    const int n = index / ${channels} / lifh / ${width};
    const int c = (index / lifh / ${width}) % ${channels};
    const int h = (index / ${width}) % lifh;
    const int w = index % ${width};

    ${Dtype} u = 0;

    const int offset = ((n * ${channels} + c) * ${height} + h * s) * ${width} + w;
    for(int j = 0; j < s; j++) {
      const int toffset = offset + j * ${width};
      if(toffset < ${numel} && h * s + j < lifhh * s) {
        u = tau[0] * o[toffset] + bottom_data[toffset];
        flag[toffset] = u > vth[0];
        if(j < s - 1)
          o[toffset + ${width}] = flag[toffset] ? 0 : u;
        top_data[toffset] = flag[toffset] ? u : vth[0];
      } else if(toffset < ${numel} && h * s + j < ${height}) {
        flag[toffset] = 1;
        top_data[toffset] = bottom_data[toffset];
      }
    }
  }
}
'''

_lif_kernel_w = kernel_loop + '''
extern "C"
__global__ void lif_forward_kernel_w(
const ${Dtype}* bottom_data, const ${Dtype}* tau, const ${Dtype}* vth, ${Dtype}* top_data, bool* flag, ${Dtype}* o) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int s = ${lif};
    const int lifw = (${width} + s - 1) / s;
    const int lifww = ${width} / s;
    const int n = index / ${channels} / ${height} / lifw;
    const int c = (index / ${height} / lifw) % ${channels};
    const int h = (index / lifw) % ${height};
    const int w = index % lifw;

    ${Dtype} u = 0;

    const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w * s;
    for(int j = 0; j < s; j++) {
      const int toffset = offset + j;
      if(toffset < ${numel} && w * s + j < lifww * s) {
        u = tau[0] * o[toffset] + bottom_data[toffset];
        flag[toffset] = u > vth[0];
        if(j < s - 1)
          o[toffset + 1] = flag[toffset] ? 0 : u;
        top_data[toffset] = flag[toffset] ? u : vth[0];
      } else if(toffset < ${numel} && w * s + j < ${width}) {
        flag[toffset] = 1;
        top_data[toffset] = bottom_data[toffset];
      }
    }
  }
}
'''

_lif_kernel_backward_grad_input_h = kernel_loop + '''
extern "C"
__global__ void lif_backward_grad_input_kernel_h(
    const ${Dtype}* const top_diff, const ${Dtype}* const tau, const ${Dtype}* const flag, const ${Dtype}* const tmpo, ${Dtype}* const bottom_diff, ${Dtype}* const tau_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int s = ${lif};
    const int lifh = (${height} + s - 1) / s;
    const int lifhh = ${height} / s;
    const int n = index / ${channels} / lifh / ${width};
    const int c = (index / lifh / ${width}) % ${channels};
    const int h = (index / ${width}) % lifh;
    const int w = index % ${width};

    ${Dtype} tmp_bottom[${lif}];
    ${Dtype} tmp_tau[${lif}];
    
    const int offset = ((n * ${channels} + c) * ${height} + h * s) * ${width} + w;
    for(int j = 0; j < s; j++) {
      const int toffset = offset + j * ${width};
      if(toffset < ${numel} && h * s + j < lifhh * s) {
        tmp_bottom[j] = flag[toffset] * top_diff[toffset];
        if(j == 0) {
          tmp_tau[j] = 0;
        } else {
          tmp_tau[j] = tmp_tau[j - 1] * tau[0] * (1 - flag[toffset - ${width}]) + tmpo[toffset];
        }
        tau_diff[toffset] = top_diff[toffset] * flag[toffset] * tmp_tau[j];
      }
    }
    if(offset + (s - 1) * ${width} < ${numel} && h * s + s - 1 < lifhh * s)
      bottom_diff[offset + (s - 1) * ${width}] = tmp_bottom[s - 1];
    for(int j = s - 2; j >= 0; j--) {
      const int toffset = offset + j * ${width};
      if(toffset + ${width} < ${numel} && h * s + j + 1 < lifhh * s) {
        tmp_bottom[j] += tmp_bottom[j + 1] * (1 - flag[toffset]) * tau[0];
        bottom_diff[toffset] = tmp_bottom[j];
      } if(toffset < ${numel} && h * s + j < lifhh * s) {
        bottom_diff[toffset] = tmp_bottom[j];
      } else if(toffset < ${numel} && h * s + j < ${height}) {
        bottom_diff[toffset] = top_diff[toffset];
      }
    }
  }
}
'''

_lif_kernel_backward_grad_input_w = kernel_loop + '''
extern "C"
__global__ void lif_backward_grad_input_kernel_w(
    const ${Dtype}* const top_diff, const ${Dtype}* const tau, const ${Dtype}* const flag, const ${Dtype}* const tmpo, ${Dtype}* const bottom_diff, ${Dtype}* const tau_diff) {
  CUDA_KERNEL_LOOP(index, ${nthreads}) {
    const int s = ${lif};
    const int lifw = (${width} + s - 1) / s;
    const int lifww = ${width} / s;
    const int n = index / ${channels} / ${height} / lifw;
    const int c = (index / ${height} / lifw) % ${channels};
    const int h = (index / lifw) % ${height};
    const int w = index % lifw;

    ${Dtype} tmp_bottom[${lif}];
    ${Dtype} tmp_tau[${lif}];

    const int offset = ((n * ${channels} + c) * ${height} + h) * ${width} + w * s;
    for(int j = 0; j < s; j++) {
      const int toffset = offset + j;
      if(toffset < ${numel} && w * s + j < lifww * s) {
        tmp_bottom[j] = flag[toffset] * top_diff[toffset];
        if(j == 0) {
          tmp_tau[j] = 0;
        } else {
          tmp_tau[j] = tmp_tau[j - 1] * tau[0] * (1 - flag[toffset - 1]) + tmpo[toffset];
        }
        tau_diff[toffset] = top_diff[toffset] * flag[toffset] * tmp_tau[j];
      }
    }
    if(offset + s - 1 < ${numel} && w * s + s - 1 < lifww * s)
        bottom_diff[offset + s - 1] = tmp_bottom[s - 1];
    for(int j = s - 2; j >= 0; j--) {
      const int toffset = offset + j;
      if(toffset + 1 < ${numel} && w * s + j + 1 < lifww * s) {
        tmp_bottom[j] += tmp_bottom[j + 1] * (1 - flag[toffset]) * tau[0];
        bottom_diff[toffset] = tmp_bottom[j];
      } else if(toffset < ${numel} && w * s + j < lifww * s) {
        bottom_diff[toffset] = tmp_bottom[j];
      } else if(toffset < ${numel} && w * s + j < ${width}) {
        bottom_diff[toffset] = top_diff[toffset];
      }
    }
  }
}
'''

class _lif_h(Function):
    @staticmethod
    def forward(ctx, input, tau, vth, lif, dim):
        assert input.dim() == 4 and input.is_cuda
        batch_size, channels, height, width = input.size()

        output = torch.zeros_like(input)#input.new(batch_size, channels, height, width)
        flag = torch.zeros_like(input).type(torch.bool)#input.new(batch_size, channels, height, width).type(torch.bool)
        tmpo = torch.zeros_like(input)#input.new(batch_size, channels, height, width)

        n = batch_size * channels * int(math.ceil(height / lif)) * width

        with torch.cuda.device_of(input):
            f = load_kernel('lif_forward_kernel_h', _lif_kernel_h, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, 
                            height=height, width=width,
                            lif=lif, numel=output.numel()
                            )
            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), tau.data_ptr(), vth.data_ptr(), output.data_ptr(), flag.data_ptr(), tmpo.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, tau, vth, flag, tmpo)
        ctx.lif, ctx.dim = lif, dim
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input, tau, vth, flag, tmpo = ctx.saved_tensors
        flag = flag.type(input.dtype)
        lif, dim = ctx.lif, ctx.dim
        batch_size, channels, height, width = input.size()

        grad_input = None
        grad_tau = None
        grad_vth = None
        n = batch_size * channels * int(math.ceil(height / lif)) * width

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels,
                   height=height, width=width, nthreads=n, numel=grad_output.numel(),
                   lif=lif
              )
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_vth = ((1. - flag) * grad_output).sum().unsqueeze(0).contiguous()
                grad_input = torch.zeros_like(input)#input.new(input.size())
                grad_tau = torch.zeros_like(input)#input.new(input.size())

                f = load_kernel('lif_backward_grad_input_kernel_h',
                                _lif_kernel_backward_grad_input_h, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), tau.data_ptr(), flag.data_ptr(), tmpo.data_ptr(), grad_input.data_ptr(), grad_tau.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                grad_tau = grad_tau.sum().unsqueeze(0).contiguous()
        return grad_input, grad_tau, grad_vth, None, None
    

class _lif_w(Function):
    @staticmethod
    def forward(ctx, input, tau, vth, lif, dim):
        assert input.dim() == 4 and input.is_cuda
        batch_size, channels, height, width = input.size()

        output = torch.zeros_like(input)
        flag = torch.zeros_like(input).type(torch.bool)
        tmpo = torch.zeros_like(input)

        n = batch_size * channels * int(math.ceil(width / lif)) * height

        with torch.cuda.device_of(input):
            f = load_kernel('lif_forward_kernel_w', _lif_kernel_w, Dtype=Dtype(input), nthreads=n,
                            num=batch_size, channels=channels, 
                            height=height, width=width,
                            lif=lif, numel=output.numel()
                            )

            f(block=(CUDA_NUM_THREADS,1,1),
              grid=(GET_BLOCKS(n),1,1),
              args=[input.data_ptr(), tau.data_ptr(), vth.data_ptr(), output.data_ptr(), flag.data_ptr(), tmpo.data_ptr()],
              stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))

        ctx.save_for_backward(input, tau, vth, flag, tmpo)
        ctx.lif, ctx.dim = lif, dim
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        assert grad_output.is_cuda
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        input, tau, vth, flag, tmpo = ctx.saved_tensors
        flag = flag.type(input.dtype)
        lif, dim = ctx.lif, ctx.dim
        batch_size, channels, height, width = input.size()

        grad_input = None
        grad_tau = None
        grad_vth = None
        n = batch_size * channels * int(math.ceil(width / lif)) * height

        opt = dict(Dtype=Dtype(grad_output),
                   num=batch_size, channels=channels,
                   height=height, width=width, nthreads=n, numel=grad_output.numel(),
                   lif=lif
              )
        with torch.cuda.device_of(input):
            if ctx.needs_input_grad[0]:
                grad_vth = ((1. - flag) * grad_output).sum().unsqueeze(0).contiguous()
                grad_input = torch.zeros_like(input)#input.new(input.size())
                grad_tau = torch.zeros_like(input)#input.new(input.size())
                f = load_kernel('lif_backward_grad_input_kernel_w',
                                _lif_kernel_backward_grad_input_w, **opt)
                f(block=(CUDA_NUM_THREADS,1,1),
                  grid=(GET_BLOCKS(n),1,1),
                  args=[grad_output.data_ptr(), tau.data_ptr(), flag.data_ptr(), tmpo.data_ptr(), grad_input.data_ptr(), grad_tau.data_ptr()],
                  stream=Stream(ptr=torch.cuda.current_stream().cuda_stream))
                grad_tau = grad_tau.sum().unsqueeze(0).contiguous()
        return grad_input, grad_tau, grad_vth, None, None


def _lif_cuda(input, tau, vth, lif, dim):
    """ involution kernel
    """
    assert dim == 2 or dim == 3

    if input.is_cuda:
        if dim == 2:
            out = _lif_h.apply(input, tau, vth, lif, dim)
        elif dim == 3:
            out = _lif_w.apply(input, tau, vth, lif, dim)
    else:
        raise NotImplementedError
    return out


class LIFSpike(nn.Module):
    def __init__(self,
                 lif, fix_tau=False, fix_vth=False, init_tau=0.25, init_vth=-1., dim=2):
        super(LIFSpike, self).__init__()
        self.lif = lif
        self.dim = dim
        if fix_tau:
            self.tau = init_tau
        else:
            self.tau = torch.nn.Parameter(torch.Tensor([init_tau]))
        if fix_vth:
            self.Vth = init_vth
        else:
            self.Vth = torch.nn.Parameter(torch.Tensor([init_vth]))
        assert dim == 2 or dim == 3

    def forward(self, x):
        if self.lif == -1:
            return x
        out = _lif_cuda(x, self.tau, self.Vth, self.lif, self.dim)
        return out



