---
layout: post
title: 'Building Support for Deep Learning Frameworks in TVM via DLPack'
author: Eddie Yan
date: 2018-07-24
---

DLPack is an intermediate in-memory representation standard for tensor data
structures. With DLPack as a common representation, we can leverage TVM in
scripts written for frameworks that traditionally could only rely on
vendor-provided libraries. TVM packed functions can operate on DLPack tensors,
which provide wrappers bridging tensor data structures from framworks such as
PyTorch and MxNet _without any data copying_.


As an example, we declare and compile a matrix multiplication operator in TVM,
and build a wrapper that uses the DLPack representation to allow this operator
to support PyTorch tensors. We also repeat this demonstration with MxNet. This
extension allows machine learning developers to quickly port research code to
relatively unsupported hardware platforms without sacrificing performance.


Illustration of how DLPack provides an intermediate wrapper that is shared
between frameworks and TVM:
#TODO update figure
{:center: style="text-align: center"}
![image](/images/pytorch-dlpack/flow.png){: width="65%"}<br />
Figure 1
{:center}

First, we compute a reference output in PyTorch:
```
    import torch
    x = torch.rand(56,56)
    y = torch.rand(56,56)
    z = x.mm(y)
```

We then define and build a TVM matrix multiplication operator, using the default
schedule:
```
    n = tvm.convert(56)
    X = tvm.placeholder((n,n), name='X')
    Y = tvm.placeholder((n,n), name='Y')

    k = tvm.reduce_axis((0, n), name='k')
    Z = tvm.compute((n,n), lambda i,j : tvm.sum(X[i,k]*Y[k,j], axis=k))
    s = tvm.create_schedule(Z.op)
    fadd = tvm.build(s, [X, Y, Z], target_host='llvm', name='fadd')
```
For brevity, we do not cover TVM's large collection of scheduling primitives
that we can use to optimize matrix multiplication. If you wish to make a custom
GEMM operator run _fast_ on your hardware device, a detailed tutorial can be
found [here](https://docs.tvm.ai/tutorials/optimize/opt_gemm.html).

We then convert the TVM function into one that supports PyTorch tensors:
```
    fadd_pytorch = to_pytorch(fadd)
    z2 = torch.empty(56,56)
    fadd_pytorch(x, y, z2)
    np.testing.assert_allclose(z.numpy(), z2.numpy())
```
and verify that the results match.

We can repeat the same example, but using MxNet instead:
```
    import mxnet
    from tvm.contrib.mxnet import to_mxnet_func
    ctx = mxnet.cpu(0)
    x = mxnet.nd.uniform(shape=(56,56), ctx=ctx)
    y = mxnet.nd.uniform(shape=(56,56), ctx=ctx)
    z = mxnet.nd.empty(shape=(56,56), ctx=ctx)
    f = tvm.build(s, [X, Y, Z], target_host='llvm', name='f')
    f_mxnet = to_mxnet_func(f)
    f_mxnet(x, y, z)
    np.testing.assert_allclose(z.asnumpy(), x.asnumpy().dot(y.asnumpy()))
```


Under the hood of the PyTorch Example
-------------------------------------

All that is required in this scenario is the extraction of the relevant tensor
description (type information) and some syntactic sugar via Python decorators.
To extract the relevant PyTorch information, we use Josh Fromm's "FireTensor"
TVM extension which uses the DLPack PyTorch bridge.

```
@tvm.register_extension
class FireTensor(object):

    _tvm_tcode = tvm.TypeCode.ARRAY_HANDLE

    def __init__(self, tensor):
        self.handle = torch._C._to_dlpack(tensor)
        self.name = self.get_name()

    def get_name(self):
        ctypes.pythonapi.PyCapsule_GetName.restype = ctypes.c_char_p
        ctypes.pythonapi.PyCapsule_GetName.argtypes = [ctypes.py_object]
        return ctypes.pythonapi.PyCapsule_GetName(self.handle)

    def to_torch(self):
        return torch._C._from_dlpack(self.handle)

    @property
    def _tvm_handle(self):
        ctypes.pythonapi.PyCapsule_GetPointer.restype = ctypes.c_void_p
        ctypes.pythonapi.PyCapsule_GetPointer.argtypes = [ctypes.py_object, ctypes.c_char_p]
        return ctypes.pythonapi.PyCapsule_GetPointer(self.handle, self.name)
```

As this extension defines the necessary transformation into a TVM NDArray, we
can just wrap this using decorators to implement our `to_pytorch` function
above:
```
def to_pytorch(module):
    #import pytorch, check for pytorch tensor
    import torch

    def converter(func, *args):
        new_args = tuple([FireTensor(arg) if isinstance(arg, torch.Tensor) else arg for arg in args])
        return func(*new_args)

    def wrapper(*args):
        module(*args)

    return decorate(wrapper, converter)
```
