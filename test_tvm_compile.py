#!/usr/bin/env python3
"""
Quick test script to verify TVM installation and basic functionality.
"""

import sys
import numpy as np

print("Testing TVM installation...")

try:
    import tvm
    print(f"✓ TVM version: {tvm.__version__}")
except ImportError as e:
    print(f"✗ Failed to import TVM: {e}")
    sys.exit(1)

try:
    from tvm import relax
    print("✓ Relax module available")
except ImportError:
    print("✗ Relax module not available")

try:
    from tvm.relax.frontend.onnx import from_onnx
    print("✓ ONNX frontend available")
except ImportError:
    print("✗ ONNX frontend not available")

# Check for CUDA support
cuda_available = False
try:
    cuda_dev = tvm.cuda(0)
    if cuda_dev.exist:
        cuda_available = True
        print(f"✓ CUDA device available: {cuda_dev}")
    else:
        print("✗ No CUDA device found")
except:
    print("✗ CUDA support not available")

# Try a simple compilation
print("\nTesting simple compilation...")
try:
    # Create a simple Relax function
    from tvm.script import relax as R
    from tvm.script import ir as I
    
    @I.ir_module
    class SimpleModule:
        @R.function
        def main(x: R.Tensor((1, 4), "float32")) -> R.Tensor((1, 2), "float32"):
            with R.dataflow():
                lv1 = R.nn.linear(x, R.const(np.random.randn(4, 2).astype("float32")))
                lv2 = R.nn.softmax(lv1, axis=-1)
                R.output(lv2)
            return lv2
    
    # Try to compile
    target = "cuda" if cuda_available else "llvm"
    print(f"Compiling for target: {target}")
    
    with tvm.transform.PassContext(opt_level=0):
        ex = tvm.compile(SimpleModule, target=target)
    
    print("✓ Compilation successful")
    
    # Try to run if possible
    dev = tvm.cuda(0) if cuda_available else tvm.cpu(0)
    vm = relax.VirtualMachine(ex, dev)
    
    test_input = tvm.nd.array(np.random.randn(1, 4).astype("float32"), dev)
    output = vm["main"](test_input)
    
    print(f"✓ Execution successful, output shape: {output.shape}")
    
except Exception as e:
    print(f"✗ Compilation/execution failed: {e}")

print("\nTVM installation check complete!")
print("\nRecommended next steps:")
print("1. Run: python create_neural_net.py")
print("2. Run: python compile_onnx_to_tvm.py")
print("3. Check the generated files in tvm_output/")