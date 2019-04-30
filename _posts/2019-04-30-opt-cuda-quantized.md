---
layout: post
title:  "Automating Optimization of Quantized Deep Learning Models on CUDA"
date:   2019-04-30 00:00:00 +0800
author: Wuwei Lin
---
Deep learning has been successfully applied to a variety of tasks.
On real-time scenarios such as inference on autonomous vehicles, the inference speed of the model is critical.
Network quantization is an effective approach to accelerating deep learning models.
In quantized models, both data and model parameters are represented with low precision data types such as `int8` and `float16`.
The lowered data bandwidth reduces the inference time and memory/storage requirements, as well as the power consumption.
Meanwhile, under proper quantization schemes, we can minimize the accuracy drops of the quantized models.
Therefore, quantized operators are of particular interests of researchers and developers as it makes large models suitable to deploy on diverse devices, such as GPU, CPU and mobile devices.

Previously, quantized operators are usually optimized with handcrafted microkernels for different workloads, or relies on blackbox proprietary solutions such as cuDNN and TensorRT.
Writing a high-performance microkernel in assembly can be very challenging and usually requires heavy engineering effort.
Besides, it is difficult to adapt these ad-hoc microkernels to emerging workloads and new devices.

{:center: style="text-align: center"}
![image](/images/cuda-quantized/benchmark.svg){: width="100%"}
{:center}
<center> Figure 1. Inference time of different models on TVM, TensorRT, and MXNet </center> <p></p>

TVM solves this challenge with a full stack compiler and a machine-learning-based optimizer to automatically generate computing kernels.
TVM can generate effcient kernels via automatic search in a human-designed search space.
In standard workloads such as VGG and ResNet, TVM achieves competitive performance compared with other state-of-the-art frameworks. 
In emerging models such as ResNeXt and Deformable ConvNets, the automatic optimization makes it easy for TVM to adapt to these new workloads and achieve a significant performance boost.

In this post, we show how to use TVM to automatically optimize of quantized deep learning models on CUDA.

# Expressing Quantized CUDA Kernels in TVM
## Leveraging Tensor Intrinsics via Tensorization
Many platforms provide architecture-specific instructions for special computation patterns, for example, the SIMD instructions on x86, and the `dp4a` and `hfma` instructions on CUDA.
These intrinsic instructions are highly optimized for specific devices.
By leveraging hardware intrinsics, we can achieve a significant performance boost for quantized operators.

Currently, [dp4a](https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/) has been extensively used in TVM int8 operators on CUDA.
`dp4a` is a CUDA intrinsic on Compute Capability 6.1 devices.
It is a mixed-precision instruction that provides the efficient computation of the dot product between two 4-element 8-bit integer vectors and accumulates the result in 32-bit format.
Using `dp4a`, we can implement a dot product between 8-bit integer vectors with number of elements evenly divisible by four.
With an efficient dot product operator, we can implement high-level operators such as 2d convolution dense layers as these operators are commonly backed by dot products.

To illustrate, in 2d convolution we accumulate along the channel, the width, and the height axis of the kernel.
This is a typical use case of `dp4a`.
TVM uses tensorization to support calling external intrinsics.
We do not need to modify the original computation declaration; we use the schedule primitive `tensorize` to replace the accumulation with `dp4a` tensor intrinsic.
More details of tensorization can be found in the [tutorial](https://docs.tvm.ai/tutorials/language/tensorize.html).

## Data Layout Rearrangement
One of the challenges in tensorization is that we may need to design special computation logic to adapt to the requirement of tensor intrinsics.
Although it is natural to accumulate along the inner axis of the tensor in the dense operator, `conv2d` can be more challenging.
In `conv2d` we expect to take a slice in the channel dimension as the input of `dp4a` because the number of channels is typically multiple of 4 (otherwise we fall back to original `conv2d` in NCHW layout).
Meanwhile, to achieve memory locality, we would like to reduce along the innermost axis first.
Taking these factors into account, we use a custom data layout to address this challenge.

In CUDA int8 2d convolution, we empirically choose `NCHW4c` as data layout and `OIHW4o4i` as weight layout.
The templates can also be easily generalized to `NCHW[x]c` and `OIHW[x]o[x]i`, where x is an arbitrary positive integer divisible by four.
In the data layout we choose, slices of channels are in the packed innermost dimension.
Likewise, we pack slices in both the input and output channel dimensions of the weight so that the output has a consistent data layout with the input, which prevents redundant layout transformations between layers.

We show the computation of one element of the output of the 2d convolution in Figure 2.
The element in each position of the super dimension (the outer dimension of the blocked layout which contains packed elements) NCHW and OIHW is the packed input and kernel, respectively.
Each column of the packed kernel comes from a different filter.
We calculate the dot product between the packed input and each row in the packed kernel using `dp4a`, and accumulate the result to the output tensor.

{:center: style="text-align: center"}
![image](/images/cuda-quantized/conv2d.png){: width="60%"}
{:center}
<div>
Figure 2. 2D convolution with data layout in NCHW4c and weight layout in OIHW4o4i.
<b>Left</b>: The input tensor in NCHW4c layout. One moving filter of the kernel is colored in blue. One element of the input and kernel is colored in grey. 
<b>Mid</b>: The packed input and kernel in the grey block.
<b>Right</b>: The output in NCHW4c layout. Inside the one element depicted, there are four packed elements in channel sub-dimension.
</div><p></p>

After we have specified the layout of convolution layers, other operators such as `add` and activations can automatically adapt to the chosen layout during the [AlterOpLayout](https://github.com/dmlc/tvm/blob/master/src/relay/pass/alter_op_layout.cc) pass in Relay.
The layout transformation of the weight can be precomputed offline. Therefore, we can run the whole model in the same layout without extra overhead.

## Designing Search Space for Automatic Optimization
The key to achieving good performance in our quantized operators is to integrate with machine-learning-based automatic optimization. One question is how to design an effective schedule search space.
An effective schedule template means that we can obtain good performance in a reasonable number of iterations in automatic tuning.
Generally speaking, we strive to define a flexible template to cover different configurations in the search space.
On the other hand, we also take advantage of the prior knowledge in performance optimization.
For example, as caching data in the shared memory is a common practice in CUDA programming, we utilize shared memory, but we use machine learning to choose the best tile size.
We also do some manual tiling such as splitting axes by 4 or 16 to facilitate vectorized memory access.

In quantized 2d convolution, we design a search space that includes a set of tunable options, such as the tile size, the axes to fuse, configurations of loop unrolling and double buffering.
The templates of quantized `conv2d` and `dense` on CUDA are registered under template key `int8`.
During automatic tuning, we can create tuning tasks for these quantized operators by setting the `template_key` argument.
Details of how to launch automatic optimization can be found in the [AutoTVM tutorial](https://docs.tvm.ai/tutorials/autotvm/tune_relay_cuda.html).

# General Workflow

{:center: style="text-align: center"}
![image](/images/cuda-quantized/workflow.png){: width="60%"}
{:center}
<center> Figure 3. Workflow of running quantized models </center><p></p>

TVM provides an easy workflow to quantize trained models from other frameworks, automatically optimize operators (with AutoTVM), and deploy to different devices.

First, we use the Relay frontend to import existing models. Here we use an MXNet model with `(1, 3, 224, 224)` input shape as an example.
```python
sym, arg_params, aux_params = mxnet.model.load_checkpoint(model_path, epoch)
net, params = relay.from_mxnet(sym, shape={'data': (1, 3, 224, 224)}, arg_params=arg_params, aux_params=aux_params)
```

Next, we use the relay quantization API to convert it to a quantized model.
```python
net = relay.quantize.quantize(sym, params=params)
```

Then, we use AutoTVM to extract tuning tasks for the operators in the model and perform automatic optimization. The [AutoTVM tutorial](https://docs.tvm.ai/tutorials/autotvm/tune_relay_cuda.html) provides an example for this.

Finally, we build the model and run inference in the quantized mode.
```python
with relay.build_config(opt_level=3):
    graph, lib, params = relay.build(net, target)
```
The result of `relay.build` is a deployable library.
We can either run inference [on the GPU](https://docs.tvm.ai/tutorials/frontend/from_mxnet.html#execute-the-portable-graph-on-tvm) directly or deploy [on the remote devices](https://docs.tvm.ai/tutorials/frontend/deploy_model_on_rasp.html#deploy-the-model-remotely-by-rpc) via RPC.

# Benchmark
To verify the performance of the quantized operators in TVM, we benchmark the performance of several popular network models including VGG-19, ResNet-50 and Inception V3.
We also benchmark on DRN-C-26, ResNeXt-50, and DCN-ResNet-101 from [Deformable ConvNets](https://github.com/msracver/Deformable-ConvNets) to show the performance of emerging models, which contains less conventional operators such as dilated convolutions, group convolutions and deformable convolutions.
We choose NVIDIA TensorRT as our baseline.
The result of MXNet 1.4 + cuDNN 7.3 in float32 mode is reported to show the speed up of quantization.
The experiments are conducted on NVIDIA GTX 1080.
We report the inference time per image when running in batch size = 1 and 16.

As shown in the Figure 1, TVM achieves up to 8x speedup using quantization.
In standard CNN models such as VGG and ResNet, TVM achieves parity with the state-of-the-art results from TensorRT.

When benchmarking emerging models, TVM achieves impressive results.
We obtain significant performance gains on ResNeXt and DCN-ResNet-101.
Results of DCN-ResNet-101 of TensorRT are not available because there is no official implementation of the deformable convolution.
We show that automatic optimization in TVM makes it easy and flexible to support and optimize emerging workloads.


# Show Me the Code
* [Benchmark](https://github.com/vinx13/tvm-cuda-int8-benchmark)
* [CUDA int8 conv2d](https://github.com/dmlc/tvm/blob/master/topi/python/topi/cuda/conv2d_int8.py)
* [CUDA int8 group conv2d](https://github.com/dmlc/tvm/blob/master/topi/python/topi/cuda/group_conv2d_nchw.py)
* [CUDA int8 dense](https://github.com/dmlc/tvm/blob/master/topi/python/topi/cuda/dense.py)
* [Tensor intrinsics declaration](https://github.com/dmlc/tvm/blob/master/topi/python/topi/cuda/tensor_intrin.py) 

# Bio & Acknowledgement
[Wuwei Lin](https://wuwei.io/) is an undergraduate student at SJTU. He is currently an intern at TuSimple. The author has many thanks to [Tianqi Chen](https://homes.cs.washington.edu/~tqchen/) and [Eddie Yan](https://homes.cs.washington.edu/~eqy/) for their reviews.
