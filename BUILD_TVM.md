# Build Apache TVM

Following https://tvm.apache.org/docs/install/from_source.html.

Start the devcontainer in VisualStudio Code.

```
conda env remove -n tvm-build-venv
conda create -n tvm-build-venv -c conda-forge \
    "llvmdev>=15" \
    "cmake>=3.24" \
    git \
    python=3.11
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
echo "set(USE_OPENMP OFF)" >> config.cmake
```
And build.
```
export LIBRARY_PATH="/home/vscode/anaconda3/lib:$LIBRARY_PATH"
cmake .. && cmake --build . --parallel
```
