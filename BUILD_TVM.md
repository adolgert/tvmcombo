# Build Apache TVM

Following https://tvm.apache.org/docs/install/from_source.html.

Checkout tvm's main branch to a directory called `tvm`.

Start the devcontainer in VisualStudio Code.
You may need to do "source /home/vscode/.bashrc".
Note use of cuda <12.2. Clang is compatible with 12.1.1 but no later.
Current version is 12.9 and default version for Conda is 12.4.
```
conda env remove -n tvm-build-venv
conda create -n tvm-build-venv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    "cuda < 12.2" \
    git \
    python=3.11 \
    cython
```
Then start the conda.
```
conda init
source /home/vscode/.bashrc
conda activate tvm-build-venv
```
Make build dir.
```
cd tvm
rm -rf build && mkdir build && cd build
cp ../cmake/config.cmake .
```
Set options:
```
# controls default compilation flags (Candidates: Release, Debug, RelWithDebInfo)
echo "set(CMAKE_BUILD_TYPE RelWithDebInfo)" >> config.cmake

# LLVM is a must dependency for compiler end
echo "set(USE_LLVM \"llvm-config --ignore-libllvm --link-static\")" >> config.cmake
echo "set(HIDE_PRIVATE_SYMBOLS ON)" >> config.cmake

# GPU SDKs, turn on if needed
echo "set(USE_CUDA   ON)" >> config.cmake
echo "set(USE_METAL  OFF)" >> config.cmake
echo "set(USE_VULKAN OFF)" >> config.cmake
echo "set(USE_OPENCL OFF)" >> config.cmake

# cuBLAS, cuDNN, cutlass support, turn on if needed
echo "set(USE_CUBLAS ON)" >> config.cmake
echo "set(USE_CUDNN  OFF)" >> config.cmake
echo "set(USE_CUTLASS OFF)" >> config.cmake
echo "set(USE_THREADS ON)" >> config.cmake
echo "set(USE_OPENMP gnu)" >> config.cmake
```
And build, but you can't use parallel for the whole build because there
is a dependency problem if you do. It's slow so use parallel, but it will fail
so restart it without parallel at the end.
```
export LIBRARY_PATH="/home/vscode/anaconda3/lib:$LIBRARY_PATH"
cmake .. && cmake --build . --parallel 4
cmake .. && cmake --build .
sudo /home/vscode/anaconda3/envs/tvm-build-venv/bin/cmake --install .
```
-    set(CMAKE_C_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_C_FLAGS}")
-    set(CMAKE_CXX_FLAGS "-O2 ${WARNING_FLAG} -fPIC ${CMAKE_CXX_FLAGS}")
+    set(CMAKE_C_FLAGS "-O2 ${WARNING_FLAG} -fPIC -Wno-deprecated-declarations -fno-var-tracking-assignments ${CMAKE_C_FLAGS}")
+    set(CMAKE_CXX_FLAGS "-O2 ${WARNING_FLAG} -fPIC -Wno-deprecated-declarations -fno-var-tracking-assignments ${CMAKE_CXX_FLAGS}") 

When you install the python package, it's important that the current working
directory be the python package.
```
cd tvm/python
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/workspaces/tvmcombo/tvm/build
python setup.py build_ext --inplace
pip install -e .
export PYTHONPATH=/workspaces/tvmcombo/tvm/python:$PYTHONPATH
```

Now prepare for the python work we will do. If the following fails, do you need it?
Maybe you can skip the pytorch.
```
conda install pytorch onnx numpy
```
