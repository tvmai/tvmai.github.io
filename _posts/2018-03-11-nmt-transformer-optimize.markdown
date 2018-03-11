---
layout: post
title:  "Bringing TVM into TensorFlow for Optimizing Neural Machine Translation inside Alibaba"
date:   2018-03-11
---

# Background

Neural Machine Translation (NMT) is an end-to-end approach for automating translation, with the potential to overcome many of the weaknesses of conventional phrase-based translation systems. Recently, Alibaba Group is working on deploying neural machine translation service for global e-commerce.  

Currently we are exploiting Transformer Model as the major backbone for our NMT system since it is more friendly for offline training with the on-par(even higher) precison against classical RNN/LSTM based models. Although Transformer model is friendly for offline training phase due to that it breaks the dependencies across time steps, it is not quite friendly for online inference. In our production environment, with the intial version of Transformer model, it has been found that the inference speed is around *1.5X* to *2X* slower than that of the LSTM version. Several optimizations have been undertaken to improve the inference performance, such as graph-level op fusion, loop invariant code motion, etc. Also it is has been observed that batch matmul is a major performance hot-spot of Transformer model and the current implementation in cuBLAS is not well optimized.

{:center: style="text-align: center"}
![image](/images/nmt-transformer/model_arch.png){: width="90%"}
{:center}

# Batch Matmul

## Why batch matmul
In Transformer model, batch matmul is widely used for the computation of multi-head attention. By using batch matmul, several attention layers from multi-head can run in parallel, which can help improve the computation efficiency of the hardware.

{:center: style="text-align: center"}
![image](/images/nmt-transformer/batchmatmul.png){: width="90%"}
{:center}


After a thorough profiling of the transformer model in inference phase, it is shown that batch matmul computations contribute up to ~ $30%$ of GPU kernel execution time. Also after using nvprof to do some first-principle analysis of cuBLAS's batch matmul kernel，it is clearly that current implementation is under-performing and some interesting phenomena have been observed.

## What is batch matmul
Typically, a batch matmul computation performs the matrix-matrix multiplication over a batch of matrices. The batch is considered to be "uniform", i.e. all instances have the same dimensions ($M$, $N$, $K$), leading dimensions (lda, ldb, ldc) and transpositions for their respective A, B and C matrices. 

Batch matmul computation can be described more vividly as follows: 

```
void BatchedGemm(input A, input B, output C, M, N, K, batch_dimension) {
  for (int i = 0; i < batch_dimension; ++i)  {
    DoGemm(A[i],B[i],C[i],M,K,N)
  } 
}
```

### Batch matmul shapes

In the language translation tasks, the shape of the batch matmul is significantly smaller than normal matmul computation in other workloads. The shape in transformer is relevant to the length of input sentences and that of decoder steps. Normally it is less than $30$.

As to the batch dimension, it is a fixed number given a certain inference batch size. For instance, if $16$ is used as batch size with beam size being $4$, the batch dimension is $16$ \* $4$ \* number of head (heads number in multi-head attention is usually $8$). The $M$, $K$, $N$, i.e. the shape of the matrix, is within the range of  $[1,$ max decode length $]$ or $[1,$ max encode length $]$.

### Performance issue of cuBLAS' batch matmul

First we make a theoretical flops analysis over the batch matmul kernels, the results are quite interesting, all the batch matmul have limited computation intensity(less than 1TFLOPs).

Then we profile the cuBLAS performance of batch matmul with multiple shapes through nvprof. In the followings are some of the metrics obtained on a NVIDIA M40 GPU with CUDA8.0 support.

| kernel |  nvprof observed FLOPs | nvprof observed FLOPs Efficiency |
| --------- |  ----- | ------ |
| maxwell\_sgemmBatched\_128x128\_raggedMn\_tn | 2155872256 | 64.72% |
| maxwell\_sgemmBatched\_64x64\_raggedMn\_tn | 161990822 | 20.86% |

Of all the maxwell_sgemmBatched_128x128_raggedMn_tn calls with various shapes(varing Mn and tn), it can be shown that all these kernels actually execute the same amount of FLOPs. It can be infered that all these different shapes are padded to a certain shape. Among all these various shapes, the extreme case is that the theoretical flops is only $2.7%$ of the actually executed flops, *so most of the computation are purely redundant*. Another set of cuBLAS kernel  maxwell_sgemmBatched_64x64_raggedMn_tn have the same issue.

<b>It is obvious that cuBLAS' batch matmul mplementation is inefficient. Thus TVM is  exploited to generate efficient batch matmul kernels for for our NMT workloads.</b>

## Batch matmul computation

In TVM, a general batch matmul computation can be declared as:

```
# computation representation
A = tvm.placeholder((batch, M, K), name='A')
B = tvm.placeholder((batch, K, N), name='B')
k = tvm.reduce_axis((0, K), 'k')
C = tvm.compute((batch, M, N),
         lambda b, y, x: tvm.sum(A[b, y, k] * B[b, k, x], axis = k),
         name = 'C')
```

# Schedule optimization

After declaring the computation, we need to devise our own schedule carefully to squeeze performance potential.

## Tuning parameters of block/thread numbers

```
  # thread indices
  block_y = tvm.thread_axis("blockIdx.y")
  block_x = tvm.thread_axis("blockIdx.x")
  thread_y = tvm.thread_axis((0, num_thread_y), "threadIdx.y")
  thread_x = tvm.thread_axis((0, num_thread_x), "threadIdx.x")
  thread_yz = tvm.thread_axis((0, vthread_y), "vthread", name="vy")
  thread_xz = tvm.thread_axis((0, vthread_x), "vthread", name="vx")

  # block partitioning
  BB, FF, MM, PP = s[C].op.axis
  BBFF = s[C].fuse(BB, FF)
  MMPP = s[C].fuse(MM, PP)
  by, ty_block = s[C].split(BBFF, factor = num_thread_y * vthread_y)
  bx, tx_block = s[C].split(MMPP, factor = num_thread_x * vthread_x)
  s[C].bind(by, block_y)
  s[C].bind(bx, block_x)
  vty, ty = s[C].split(ty_block, nparts = vthread_y)
  vtx, tx = s[C].split(tx_block, nparts = vthread_x)
  s[C].reorder(by, bx, vty, vtx, ty, tx)
  s[C].reorder(by, bx, ty, tx)
  s[C].bind(ty, thread_y)
  s[C].bind(tx, thread_x)
  s[C].bind(vty, thread_yz)
  s[C].bind(vtx, thread_xz)
```
We fuse the outer dimensions of the batch matmul, i.e. the BB and FF of the op's dimension, normally known as "batch" dimension in batch matmul computation. Then we split the outer and the inner dimensions by a factor of (number_thread * vthread).

Strided pattern is not needed in batch matmul, thus the virtual thread number (vthread\_y & vthread\_x) are both set to $1$.

### Finding the best combination of number\_thread

The results below are obtained on a NVIDIA M40 GPU device with CUDA8.0 support.

| Input Shape [batch,features,M,N,K] | num\_thread\_y, num\_thread\_x | num\_vthread\_y, num\_vthread\_x | Time(us) |
| --------- | ---------- | ----- | ------ |
| [64,8,1,17,128] | 8,1 | 32,1 | 37.62 |
| [64,8,1,17,128] | 4,1 | 32,1 | 39.30 |
| [64,8,1,17,128] | 1,1 | 32,1 | 38.82 |
| [64,8,1,17,128] | 1,1 | 256,1 | 41.95 |
| [64,8,1,17,128] | 32,1 | 1,1 | 94.61 |

As learned from [past experience](http://tvmlang.org/2017/08/22/Optimize-Deep-Learning-GPU-Operators-with-TVM-A-Depthwise-Convolution-Example.html), the method to find the best combination of num\_thread\_y and num\_thread\_x is through brute force search. After a brute force search, the best combination for current shape can be found, which in current computation is num\_thread\_y = $8$ and num\_thread\_x = $32$.

# Fuse batch matmul with other operations

Normally, the existing "black-box" cuBLAS library calls play as the boundary of the normally used "op fusion" optimization tactics. However, with the generated efficient batch matmul kernel, the fusion boundary can be easily broken, thus futher performance improvement can be obtained.

By analyzing the computation graph, it has been observed that a batch matmul is always followed by a *broadcast add* operation or a *transpose* operation. By fusing add or transpose operation with batch matmul, kernel launch overhead and redundant memory access can be reduced.

Batch matmul and broadcast add fusion computation can be declared as following:

```
# computation representation
A = tvm.placeholder((batch_size, features, M, K), name='A')
# the shape of B is (N, K) other than (K, N) is because B is transposed is this fusion pattern
B = tvm.placeholder((batch_size, features, N, K), name='B')
ENTER = tvm.placeholder((batch_size, 1, M, N), name = 'ENTER')
k = tvm.reduce_axis((0, K), 'k')
C = tvm.compute(
           (batch_size, features, M, N),
           lambda yb, yf, m, x: tvm.sum(A[yb, yf, m, k] * B[yb, yf, x, k], axis = k),
           name = 'C')
D = topi.broadcast_add(C, ENTER)
```

Batch matmul and transpose fusion computation can be declared as:

```
# computation representation
A = tvm.placeholder((batch_size, features, M, K), name='A')
B = tvm.placeholder((batch_size, features, K, N), name='B')
k = tvm.reduce_axis((0, K), 'k')
C = tvm.compute(
           (batch_size, M, features, N),
           lambda yb, m, yf, x: tvm.sum(A[yb, yf, m, k] * B[yb, yf, k, x], axis = k),
           name = 'C')
```

## Kernel performance

We choose the shape of [batch=$64$, heads=$8$, M=$1$, N=$17$, K=$128$] to elaborate the performance of the generated code. $17$ is chosen as the sequenc length since it is the average input length in our production scenarios.

| Input Shape [batch,heads,M,N,K] | computation | time |
| --------- | ---------- | ----- |
| [64,8,1,17,128] | tf-r1.4 BatchMatMul | 513.9us |
| [64,8,1,17,128] | tf-r1.4 BatchMatMul+Transpose(separate) | 541.9us |
| [64,8,1,17,128] | TVM BatchMatMul | 37.62us |
| [64,8,1,17,128] | TVM BatchMatMul+Transpose(fused) | 38.394us |

It is clearly that TVM generated kernel(with schdule optimization) brings at least $13X$ speed-up. The kernel fusion optimization brings further $1.7X$ speed-up.

# Ahead of time Codegen

The input shape of batch matmul in our workload is finite and can be enumerated easily in advance. With those pre-defined shapes, We can generate highly optimized cuda kernel ahead of time(fixed shape computation could bring the best optimization potential). Meanwhile, a general batch matmul kernel suitable for most shapes will also be generated to provide a fall-back machanism for the shapes which does not have a corresponding ahead-of-time generated kernel. 

The generated efficient kernels for the specific shapes and the fall-back one are integrated into Tensorflow framework. Also a graph optimization pass is developedto automatically replace the origin batch *matmul + add/transpose* pattern with the TVM kernel. 

# Summary
Inside Alibaba, we found that TVM is a very productive tool to develop high performance GPU kernels to meet our in-house requirements. In this blog, NMT Transformer model is taken as an example to illustrate our optimization strategy with TVM. Firstly we locate the hot-spot of Transformer model through first-principle analysis. Then we use TVM to generate highly optimized CUDA kernel to replace cuBLAS version(*13X* speed-up is observed). Next we leverage TVM's kernel fusion mechanism to fuse the preceding/following operations of batch matmul to bring further performance improvement(with *1.7X* further performance improvment). The end2end performance improvement is *1.4X*. Based on those generated kernels a graph optimization pass is developed to replace the original computation pattern with the TVM fused kernels automatically to ensure the optimization is transparent to end users since as AI infrastructure provider we found transparency of optimization strategy is very important to popularize its adoption. In the last, but not the least, all those optimizations are integrated into TensorFlow in a loosely coupled way, demonstrating a potential way for integrating TVM with different deep learning frameworks. Also there is an ongoing work to integrate TVM as a codegen backend for TensorFlow, we hope in the future further result could be shared with the community.

