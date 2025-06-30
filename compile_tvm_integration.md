sudo apt update
sudo apt upgrade
sudo apt install clang
cd tvm && sudo /home/vscode/anaconda3/envs/tvm-build-venv/bin/cmake --install .
# Compilation command options.
Ordered from newest to oldest.
clang++ -std=c++17 \
      -I./tvm/include \
      -I./tvm/ffi/include \
      -I./tvm/3rdparty/dlpack/include \
      -I./tvm/3rdparty/dmlc-core/include \
      -I/home/vscode/anaconda3/envs/tvm-build-venv/targets/x86_64-linux/include \
      -DTVM_EXPORTS \
      -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
      -L./tvm/build -ltvm -ltvm_runtime \
      -L/home/vscode/anaconda3/envs/tvm-build-venv/lib \
      -lcudart \
      src/tvm_integration.cu -o tvm_integration

clang++ -std=c++17 \
      -I./tvm/include \
      -I./tvm/ffi/include \
      -I./tvm/3rdparty/dlpack/include \
      -I./tvm/3rdparty/dmlc-core/include \
      -I/home/vscode/anaconda3/envs/tvm-build-venv/targets/x86_64-linux/include \
      -DTVM_EXPORTS \
      -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
      -L./tvm/build -ltvm -ltvm_runtime \
      -L/home/vscode/anaconda3/envs/tvm-build-venv/lib \
      -lcudart \
      src/tvm_integration.cu -o tvm_integration

$ clang++ -std=c++17 --cuda-gpu-arch=sm_75 \
      -I./tvm/include \
      -I./tvm/ffi/include \
      -I./tvm/3rdparty/dlpack/include \
      -I./tvm/3rdparty/dmlc-core/include \
      -DTVM_EXPORTS \
      -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
      -Wno-unknown-cuda-version \
      -L./tvm/build -ltvm -ltvm_runtime \
      -lcudart \
      src/tvm_integration.cu -o tvm_integration

clang++ -std=c++17 --cuda-gpu-arch=sm_75 \
      -I/home/vscode/anaconda3/envs/tvm-build-venv/targets/x86_64-linux/include \
      -I./tvm/include \
      -I./tvm/ffi/include \
      -I./tvm/3rdparty/dmlc-core/include \
      -I./tvm/3rdparty/dlpack/include \
      -DTVM_EXPORTS \
      -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
      -Wno-unknown-cuda-version \
      -Wno-unused-function \
      -L./tvm/build -ltvm -ltvm_runtime \
      -L/home/vscode/anaconda3/envs/tvm-build-venv/lib \
      -lcudart \
      src/tvm_integration.cu -o tvm_integration

clang++ -std=c++17 --cuda-gpu-arch=sm_75 \
      -I./tvm/include \
      -I./tvm/ffi/include \
      -I./tvm/3rdparty/dmlc-core/include \
      -I./tvm/3rdparty/dlpack/include \
      -I/usr/local/cuda/include \
      -DTVM_EXPORTS \
      -DDMLC_USE_LOGGING_LIBRARY=\<tvm/runtime/logging.h\> \
      -D__CUDA_NO_HALF_OPERATORS__ \
      -D__CUDA_NO_HALF_CONVERSIONS__ \
      -D__CUDA_NO_BFLOAT16_CONVERSIONS__ \
      -Wno-unknown-cuda-version \
      -Wno-unused-function \
      -L./tvm/build -ltvm -ltvm_runtime \
      -lcudart \
      src/tvm_integration.cu -o tvm_integration

clang++ -std=c++17 --cuda-gpu-arch=sm_75 \
      -I./tvm/include \
      -I./tvm/ffi/include \
      -I./tvm/3rdparty/dmlc-core/include \
      -I./tvm/3rdparty/dlpack/include \
      -I/usr/local/cuda/include \
      -DTVM_EXPORTS \
      -Wno-unknown-cuda-version \
      -L./tvm/build -ltvm -ltvm_runtime \
      -lcudart \
      src/tvm_integration.cu -o tvm_integration


clang++ -std=c++17 --cuda-gpu-arch=sm_75 \
       -I./tvm/include \
       -I./tvm/ffi/include \
       -I./tvm/3rdparty/dmlc-core/include  \
       -I./tvm/3rdparty/dlpack/include  \
       -I/usr/local/cuda/include \
       -DTVM_EXPORTS  \
       -L./tvm/build -ltvm -ltvm_runtime -lcudart  src/tvm_integration.cu -o tvm_integration

clang++ -std=c++17 --cuda-gpu-arch=sm_75 \
      -I/usr/local/include \
      -I./tvm/include \
      -I./tvm/ffi/include \
      -I./tvm/3rdparty/dmlc-core/include \
      -I/usr/local/cuda-12.9/targets/x86_64-linux/include \
      -I./src -I./tvm/include -I./tvm/3rdparty/dlpack/include \
      -L./tvm/build -ltvm -ltvm_runtime \
      -lcudart -lcublas \
      src/tvm_integration.cu -o tvm_integration
