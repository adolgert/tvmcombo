# clang's cuda implementation is compatible with CUDA 12.1 at newest.
ARG BASE_IMAGE=nvidia/cuda:12.1.1-devel-ubuntu22.04
# ARG BASE_IMAGE=nvidia/cuda:12.9.1-devel-ubuntu22.04
FROM ${BASE_IMAGE}

# Avoid warnings by switching to noninteractive
ENV DEBIAN_FRONTEND=noninteractive

# Configure apt and install packages
RUN apt-get update \
    && apt-get -y install --no-install-recommends apt-utils dialog 2>&1 \
    #
    # Verify git, process tools, lsb-release (common in install instructions for CLIs) installed
    && apt-get -y install git openssh-client less iproute2 procps lsb-release gnupg
    #
    # Install C++ tools
RUN apt-get -y install build-essential cmake gdb
    #
    # Install Python 3.11
RUN apt-get -y install software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get -y install python3.11 python3.11-dev python3.11-venv python3-pip
    #
    # Install LLVM 19
# RUN wget -O- https://apt.llvm.org/llvm-snapshot.gpg.key | gpg --dearmor | sudo tee /usr/share/keyrings/llvm-archive-keyring.gpg >/dev/null
# RUN echo "deb [signed-by=/usr/share/keyrings/llvm-archive-keyring.gpg] http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main" | tee /etc/apt/sources.list.d/llvm.list
# # RUN wget -O - https://apt.llvm.org/llvm-snapshot.gpg.key | apt-key add -
# # RUN add-apt-repository "deb http://apt.llvm.org/jammy/ llvm-toolchain-jammy-19 main"
# RUN apt-get update \
#     && apt-get -y install llvm-19 clang-19 clang-tools-19
    #
    # Install CMake (latest version)
RUN pip3 install cmake
    #
    # Common utilities
RUN apt-get -y install curl wget unzip zip nano vim sudo libxml2-dev
    #
    # Clean up
RUN apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

RUN wget https://apt.llvm.org/llvm.sh && \
    chmod +x llvm.sh && \
    ./llvm.sh 19 all && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-19 100 && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-19 100

    # Install Conda
# RUN wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh \
#     && bash /tmp/miniconda.sh -b -p /opt/conda \
#     && rm /tmp/miniconda.sh \
#     && /opt/conda/bin/conda clean -tipsy

# Add conda to path
# ENV PATH="/opt/conda/bin:${PATH}"

# Create a non-root user
ARG USERNAME=vscode
ARG USER_UID=1000
ARG USER_GID=$USER_UID

RUN groupadd --gid $USER_GID $USERNAME \
    && useradd --uid $USER_UID --gid $USER_GID -m $USERNAME -s /bin/bash \
    && echo $USERNAME ALL=\(root\) NOPASSWD:ALL > /etc/sudoers.d/$USERNAME \
    && chmod 0440 /etc/sudoers.d/$USERNAME \
    && mkdir -p /home/$USERNAME/.vscode-server /home/$USERNAME/.vscode-server-insiders \
    && chown -R $USERNAME:$USERNAME /home/$USERNAME

# Set up Python 3.11 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Set up conda for the vscode user
# RUN /opt/conda/bin/conda init bash \
#     && echo 'export PATH="/opt/conda/bin:$PATH"' >> /home/$USERNAME/.bashrc

# Switch back to dialog for any ad-hoc use of apt-get
ENV DEBIAN_FRONTEND=dialog

# Set the working directory to the workspace
WORKDIR /workspaces

USER $USERNAME
