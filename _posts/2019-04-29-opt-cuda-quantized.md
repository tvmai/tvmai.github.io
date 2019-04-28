---
layout: post
title:  "Optimizing Quantized Operators on CUDA"
date:   2019-04-29 00:00:00 +0800
author: Wuwei Lin
---
Deep learning has been successfully applied to a variety of tasks.
On real-time scenarios such as inference on autonomous vehicles, the inference speed of the model is critical. Network quantization is a practical approach to accelerating deep learning models.
In quantized models, both data and model parameters are represented with low precision data types such as `int8` and `float16`.
The lowered data bandwidth reduces the inference time and the requirement of memory, as well as the power consumption.
Meanwhile, under proper quantization schemes, we can minimize the accuracy drops of the quantized models.
Therefore, quantized operators are of particular interests of researchers and developers as it makes large models suitable to deploy on a diversity of devices, such as GPU, CPU and mobile devices.

{:center: style="text-align: center"}
![image](/images/cuda-quantized/benchmark.svg){: width="100%"}
{:center}
<center> Figure 1. Inference time of different models on TVM, TensorRT, and MXNet </center> <p></p>

Previously, quantized operators are usually optimized with handcrafted microkernels for different workloads.
Writing a high-performance microkernel in assembly usually requires heavy engineering efforts.
Besides, it is difficult to adopt these ad-hoc microkernels to new workloads and new devices.
TVM solves this challenge with the tensor IR and machine-learning-based automatic tuning.
In this post, we introduce the optimization for the quantized CUDA operators in TVM.

# Tensor Intrinsics
Many platforms provide architecture-specific instructions for special computation patterns, for example, the SIMD instructions on x86, and the `dp4a` and `hfma` instructions on CUDA.
These intrinsic instructions are highly-optimized for specific devices.
By leveraging hardware intrinsics, we can achieve a significant performance boost for quantized operators.

Currently, `dp4a` has been extensively used in TVM int8 operators on CUDA.
`dp4a` is a CUDA intrinsic on Compute Capability 6.1 devices.
It is a mixed-precision instruction that provides the efficient computation of the dot product between two 4-elements 8-bit integer vectors and accumulates the result in 32-bit format.
Using `dp4a`, we can implement dot product between 8-bit integer vectors that have the number of elements divisible by four.
With an efficient dot product operator, we can implement high-level operators such as 2d convolution and dense since these operators usually contain the computation of dot products.

{:center: style="text-align: center"}
![image](/images/cuda-quantized/dp4a.png){: width="20%"}
{:center}
<center> Figure 2. The dp4a instruction. Source: https://devblogs.nvidia.com/mixed-precision-programming-cuda-8/ </center> <p></p>

To illustrate, in 2d convolution we accumulate along the channel, the width and the height axis of the kernel.
This is a typical use case of `dp4a`.
TVM uses tensorization to support calling external intrinsics.
We declare the computation in the same way as the original one and then use the schedule primitive `tensorize` to replace the accumulation with `dp4a` tensor intrinsic.
More details of tensorization can be found in the [tutorial](https://docs.tvm.ai/tutorials/language/tensorize.html).

# Data Layout Rearrangement
One of the challenges in tensorization is that we may need to design special computation logic to adapt to the requirement of tensor intrinsics.
Although it is natural to accumulate along the inner axis of the tensor in the dense operator, `conv2d` can be more challenging.
In `conv2d` we expect to take a slice in the channel dimension as the input of `dp4a` because the number of channels is typically multiple of 4 (otherwise we fall back to original `conv2d` in NCHW layout).
Meanwhile, to achieve memory locality, we would like to reduce along with the innermost axis first.
Taking these factors into account, we use a custom data layout to address this challenge.

In CUDA int8 2d convolution, we empirically choose `NCHW4c` as data layout and `OIHW4o4i` as weight layout.
The templates can also be easily generalized to `NCHW[x]c` and `OIHW[x]o[x]i`, where x is an arbitrary positive integer divisible by four.
In the data layout we choose, slices of channels is packed innermost.
Likewise, we pack slices in both the input and output channel dimensions of the weight so that the output has a consistent data layout with the input, which prevents layout transformations between layers.

We show the computation of one element of the output of the 2d convolution in Figure 3.
The element in each position of the super dimension NCHW and OIHW is the packed input and kernel, respectively.
Each column of the packed kernel comes from a different filter.
We calculate the dot product between the packed input and each row in the packed kernel using `dp4a`, and accumulate the result to the output tensor.

{:center: style="text-align: center"}
![image](/images/cuda-quantized/conv2d.png){: width="60%"}
{:center}
<div>
Figure 3. 2D convolution with data layout in NCHW4c and weight layout in OIHW4o4i.
<b>Left</b>: The input tensor in NCHW4c layout. One moving filter of the kernel is colored in blue. One element of the input and kernel is colored in grey. 
<b>Mid</b>: The packed input and kernel in the grey block.
<b>Right</b>: The output in NCHW4c layout. Inside the one element depicted there are four packed elements in channel sub-dimension.
</div><p></p>

After we have specified the layout of convolution layers, other operators such as `add` and activations can automatically adapt to the chosen layout during the [AlterOpLayout](https://github.com/dmlc/tvm/blob/master/src/relay/pass/alter_op_layout.cc) pass in Relay.
The layout transformation of the weight can be precomputed offline. Therefore, we can run the whole model in the same layout without extra overhead.

# Automatic Tuning
The key to achieving good performance in our quantized operators is to integrate with automatic tuning. One question is how to write an effective schedule template.
An effective schedule template means that we can obtain good performance in a reasonable number of iterations in automatic tuning.
Generally speaking, we strive to define a flexible template to cover different configurations in the search space.
On the other hand, we also take advantage of the prior knowledge in performance optimization.
For example, as caching data in the shared memory is a common practice in CUDA programming, we utilize the shared memory, but we let the algorithm to decide the best tile size.
We also do some manual tiling such as splitting axes by 4 or 16 to facilitate vectorized memory access.

In quantized 2d convolution, we use a schedule template that includes a set of tunable options, such as the tile size, the axes to fuse, configurations of loop unrolling and double buffering.
The templates of quantized `conv2d` and `dense` on CUDA are registered under template key `int8`.
During automatic tuning, we can create tuning tasks for these quantized operators by setting the `template_key` argument.

# Running Quantized Models

{:center: style="text-align: center"}
![image](/images/cuda-quantized/workflow.png){: width="60%"}
{:center}
<center> Figure 4. Workflow of running quantized models </center><p></p>

TVM provides an easy workflow of tuning and running quantized model from the original trained model.
First, we use the Relay frontend to import existing models.
Next, we use the relay quantization API to convert it to a quantized model.
Then, we use AutoTVM to extract tuning tasks for the operators in the model and perform automatic tuning.
Finally, we build the model and run inference in the quantized mode.


# Benchmark
To verify the performance of the quantized operators in TVM, we benchmark the performance of several popular network models including VGG-19, ResNet-50 and Inception V3.
We also benchmark on DRN-C-26, ResNeXt-50 to show the performance of the dilated convolution, the group convolution.
DCN-ResNet-101 from [Deformable ConvNets](https://github.com/msracver/Deformable-ConvNets) is provided as an example of less conventional models.
We choose TensorRT as our baseline.
The result of MXNet 1.4 + cuDNN 7.3 in float32 mode is reported to show the speed up of quantization.
The experiments are conducted on NVIDIA GTX 1080.
We report the inference time per image when running in batch size = 1 and 16.

As shown in the Figure 1, TVM achieves up to 8x speedup using quantization.
Compared with TensorRT, TVM achieves parity with TensorRT performance in most models.
TVM currently lags behind in DRN-C-26, the dilated convolution model.
We can continue to improve it.

From the result of DCN-ResNet-101, the model with deformable convolutions, we can see that TVM can also bring speedup to less conventional models.
Results of TensorRT is not available because there is no official implementation of the deformable convolution.


# Show Me the Code
* [Benchmark](https://github.com/vinx13/tvm-cuda-benchmark)
* [CUDA int8 conv2d](https://github.com/dmlc/tvm/blob/master/topi/python/topi/cuda/conv2d_int8.py)
* [CUDA int8 group conv2d](https://github.com/dmlc/tvm/blob/master/topi/python/topi/cuda/group_conv2d_nchw.py)
* [CUDA int8 dense](https://github.com/dmlc/tvm/blob/master/topi/python/topi/cuda/dense.py)
* [Tensor intrinsics declaration](https://github.com/dmlc/tvm/blob/master/topi/python/topi/cuda/tensor_intrin.py) 

