# zigtensor <img src="zigtensor_alpha.png" alt="zigtensor logo" height="50"/> (WIP)

**A work-in-progress deep learning library written purely* in Zig, aiming for performance, explicitness, and harnessing the power of CUDA.**

*(Well, mostly Zig! Performance-critical GPU kernels are written in CUDA C/C++ and called via Zig's C FFI).*

---

## 🤔 Why `zigtensor`?

The world of deep learning is incredible, but often the underlying frameworks can feel a bit like magic boxes. At the same time, Zig is emerging as a fascinating language for building robust, performant, and maintainable systems software.

`zigtensor` aims to bridge these worlds! We want to build a deep learning library that:

1.  **Leverages Zig's Strengths:** Uses explicit allocators, `comptime` for metaprogramming and compile-time checks, clear error handling (no hidden exceptions!), and Zig's focus on simplicity to create a robust foundation.
2.  **Is Built "From Scratch":** We're implementing core tensor operations, automatic differentiation, layers, and optimizers ourselves (CPU in Zig, GPU in raw CUDA kernels). This is partly for the learning experience (ours and hopefully yours!) and partly to have fine-grained control.
3.  **Provides GPU Acceleration:** Modern deep learning needs GPUs. We're directly integrating CUDA support for high-performance training and inference, without relying *initially* on high-level libraries like cuDNN or cuBLAS (though we might add options later!).
4.  **Feels Idiomatic for Zig Developers:** If you like Zig, hopefully using `zigtensor` will feel natural.

This is an ambitious project, inspired by giants like PyTorch/TensorFlow but also by minimalist projects like tinygrad and the overall Zig philosophy.

## ✨ Goals & Features (Planned)

* **Idiomatic Zig API:** Explicit allocators, error unions (`try`/`catch`), `comptime` usage.
* **Core `Tensor` struct:** Multi-dimensional arrays on CPU & GPU.
* **CPU Backend:** Operations implemented in pure Zig.
* **GPU Backend:** Raw CUDA C/C++ kernels called via FFI for maximum control (initially).
* **Math Operations:** Element-wise, GEMM, reductions, convolutions, etc., from scratch.
* **Automatic Differentiation:** Tape-based or graph-based reverse-mode autodiff.
* **Neural Network Layers:** Linear, Conv2D, RNNs, Attention, Normalization, Pooling, etc.
* **Optimizers:** SGD, Adam, RMSprop, etc.
* **Loss Functions:** MSE, Cross-Entropy, etc.
* **Serialization:** Saving and loading models/tensors (maybe SafeTensors compatible?).
* **Clear & Testable Code:** A codebase that's relatively easy to understand and contribute to.

## 🚧 Current Status (As of late March 2025)

⚠️ **VERY EARLY STAGES!** ⚠️

`zigtensor` is currently under active development and is **not yet ready for any serious use**. We are working through the foundational pillars – setting up the build system, core tensor types, memory management, and basic operations.

Think of it as the digital clay being molded. Expect rough edges, missing features, and breaking API changes for a while!

## 🚀 Getting Started (Eventually!)

*(Instructions and examples will be added here once the library becomes minimally usable.)*

For now, you'd need the Zig compiler (see [ziglang.org](https://ziglang.org/)) and the NVIDIA CUDA Toolkit installed to build the project.

```zig
// Hypothetical future usage example:
const std = @import("std");
const zt = @import("zigtensor"); // Project's main import

pub fn main() !void {
    // Standard Zig allocator pattern
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    defer _ = gpa.deinit();
    const allocator = gpa.allocator();

    // Create tensors (CPU example)
    var tensor_a = try zt.Tensor.initCpu(allocator, .{2, 2}, .f32);
    defer tensor_a.deinit();
    try tensor_a.fill(1.0); // Fill op needs implementing!

    var tensor_b = try zt.Tensor.initCpu(allocator, .{2, 2}, .f32);
    defer tensor_b.deinit();
    try tensor_b.fill(2.0);

    // Perform operations (API subject to change!)
    var result = try tensor_a.addAlloc(&tensor_b, allocator);
    defer result.deinit();

    // TODO: Add printing / inspection
    std.debug.print("Operation finished (result inspection TBD).\n", .{});
}
