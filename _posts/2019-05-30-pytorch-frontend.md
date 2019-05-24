PyTorch now has an official TVM-based backend, [torch_tvm](https://github.com/pytorch/tvm).  Usage is simple:

```
import torch_tvm
torch_tvm.enable()
```

That's it!  PyTorch will then attempt to convert all operators it can to known Relay operators during its JIT compilation process.

### Background

Unlike many other ML frameworks, PyTorch exposes an eager-execution programming interface.  This style of programming avoids graph-based meta-programming and focuses on the direct manipulation of n-dimensional arrays (tensors) in a Pythonic way.  As such, the framework was initially well suited for the experimentation and development of models, but not for automatic performance optimization or deployment.  To leverage optimizing compiler techniques, some large changes were recently introduced to PyTorch to solve this problem.

![TVM Integration](https://i.imgur.com/4XVHbJE.png)

PyTorch 1.0 introduced PyTorch IR, a PyTorch-specific intermediate representation for models similar to Relay.  PyTorch programs can be converted into the IR via model tracing, which records the execution of a model or TorchScript, a subset of Python.  The new TVM backend lowers PyTorch IR to Relay, and is able to transparently improve PyTorch performance with little user involvement.

### Integration and Results

To support Relay, two features were added to the PyTorch JIT: custom transformation passes and custom subgraph interpreters.

When `torch_tvm` is enabled, subgraphs of PyTorch IR that can be converted to Relay `Expr`s will be marked as Relay-compatible.  Since PyTorch IR does not always contain shape information, none of the subgraphs can be compiled in a useful way before invocation.

During user invocation, the PyTorch JIT runtime will determine input shape information and compile the previously marked subgraphs with the new Relay C++ [build system](https://github.com/pytorch/tvm/blob/master/torch_tvm/compiler.cpp#L226-L246).  The compilation is cached based on input shapes for subsequent runs.  More details can be found in the [README](https://github.com/pytorch/tvm/blob/master/README.md).

`torch_tvm` has a continuous benchmark system set up, which is monitoring the performance of ResNet18 on CPU.
Out of the box TVM provides over two times the performance of the default PyTorch JIT backend for various ResNet models.
Below is a graph that details the iterations per second achieved with 16 threads on an AWS c5n.4xlarge instance (larger is better):

![bench](https://i.imgur.com/KfJ7oas.png)

These results are quite encouraging, and the project will continue to focus on improving CPU inference speed across more models.

### Future work

Right now the PyTorch JIT does a lot of work to find pure functional subsets of its IR to feed to Relay.  This avoids the need to map aliasing and control flow information to Relay, but is not necessary.  Mapping more of the PyTorch IR to Relay may yield performance wins and is a goal of the project.  PyTorch IR is rapidly changing as it is being developed, so this must be done carefully.

More work will be done to ensure the hand off between PyTorch and TVM code is efficient.  This includes unifying the threading model, allocators and reducing the overhead associated with copying inputs into TVM.

### Tutorial

If you have an already written PyTorch model, the easiest way to get started comes from using `torch.jit.trace` as follows

```
import torch_tvm
from your_model import model, inputs

torch_tvm.enable(opt_level=3)

iters = 100
warmup = 10

# Ensure your model is in eval mode and also turn of gradients.
with torch.no_grad():
  # Use tuned parameters for better performance.
  with autotvm.apply_history_best("test/autotvm_tuning.log"):
    # This is where all the compilation happens.
    trace_tvm = torch.jit.trace(model, inputs)
    
    # Warmup
    for _ in range(warmup):
      _ = trace_tvm(*inputs)

    # Benchmark
    start = time.time()
    for _ in range(iters):
      _ = trace_tvm(*inputs)
    tvm_time = time.time() - start
    
    print("Took {}s to run {} iters".format(tvm_time, iters))
```

Much of this code comes from [benchmarks.py](https://github.com/pytorch/tvm/blob/master/test/benchmarks.py).  Note that tuned parameters for AVX2 LLVM compilation is in the `test/` folder of the repo.
