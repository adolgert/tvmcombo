sudo apt update
sudo apt upgrade
sudo apt install clang
cd tvm && sudo /home/vscode/anaconda3/envs/tvm-build-venv/bin/cmake --install .

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
