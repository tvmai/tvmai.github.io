---
layout: page
title: "About"
order : 12
group : navigation
description: "TVM"
---
{% include JB/setup %}

# About TVM


TVM is an open deep learning compiler stack for CPUs, GPUs, and specialized accelerators. It aims to close the gap between the productivity-focused deep learning frameworks,
and the performance- or efficiency-oriented hardware backends. TVM provides the following main features:

- Compilation of deep learning models in Keras, MXNet, PyTorch, Tensorflow, CoreML, DarkNet into minimum deployable modules on diverse hardware backends.
- Infrastructure to automatic generate and optimize tensor operators
  on more backend with better performance.

TVM stack began as a research project at the [SAML group](https://saml.cs.washington.edu/) of
Paul G. Allen School of Computer Science & Engineering, University of Washington. The project is now driven by an open source community involving multiple industry and academic institutions.
The project adopts [Apache-style merit based governace model](https://docs.tvm.ai/contribute/community.html).

TVM provides two level optimizations show in the following figure.
Computational graph optimization to perform tasks such as high-level operator fusion, layout transformation, and memory management.
Then a tensor operator optimization and code generation layer that optimizes tensor operators. More details can be found at the [techreport](https://arxiv.org/abs/1802.04799).

{:center: style="text-align: center"}
![image](/images/main/stack_tvmlang.png){: width="90%"}
{:center}
