conda deactivate
conda deactivate
unset CPLUS_INCLUDE_PATH
unset C_INCLUDE_PATH
unset CMAKE_INCLUDE_PATH
unset CMAKE_LIBRARY_PATH
sudo apt-get update --fix-missing
sudo apt-get install -y --no-install-recommends \
  python3-pip python3-dev build-essential cmake cmake-curses-gui checkinstall g++ gcc gfortran \
  git vim tmux wget curl bzip2 ca-certificates gnupg software-properties-common \
  libglib2.0-0 libsm6 libxext6 libxrender-dev libgtk2.0-dev qtbase5-dev \
  libblas-dev liblapack-dev libatlas-base-dev libssl-dev libzmq3-dev \
  libboost-filesystem-dev libboost-date-time-dev libboost-iostreams-dev libboost-system-dev libboost-program-options-dev libboost-all-dev \
  libflann-dev libjpeg8-dev libtiff5-dev pkg-config yasm libavcodec-dev libavformat-dev libswscale-dev libdc1394-dev libxine2-dev libv4l-dev libtbb-dev ffmpeg \
  libfaac-dev libmp3lame-dev libtheora-dev libvorbis-dev libxvidcore-dev libopencore-amrnb-dev libopencore-amrwb-dev x264 v4l-utils \
  libprotobuf-dev protobuf-compiler libgoogle-glog-dev libgflags-dev \
  libgphoto2-dev libhdf5-dev doxygen proj-data libproj-dev libyaml-cpp-dev freeglut3-dev \
  rsync lbzip2 pigz zip p7zip-full p7zip-rar

wget http://www.cmake.org/files/v3.25/cmake-3.25.3.tar.gz
tar xf cmake-3.25.3.tar.gz
cd cmake-3.25.3
./configure
make
sudo make install
cd ..
rm -rf cmake-3.25.3 cmake-3.25.3.tar.gz
hash -r

wget https://gitlab.com/libeigen/eigen/-/archive/3.4.0/eigen-3.4.0.tar.gz
tar xf eigen-3.4.0.tar.gz
cd eigen-3.4.0
mkdir build && cd build
/usr/local/bin/cmake ..
sudo make install
cd ../..
rm -rf eigen-3.4.0 eigen-3.4.0.tar.gz

git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv
git clone --depth 1 --branch 4.11.0 https://github.com/opencv/opencv_contrib
mkdir -p opencv/build
cd opencv/build
/usr/bin/cmake .. -DCMAKE_BUILD_TYPE=Release \
  -DBUILD_CUDA_STUBS=OFF -DBUILD_DOCS=OFF -DWITH_MATLAB=OFF -Dopencv_dnn_BUILD_TORCH_IMPORTE=OFF \
  -DCUDA_FAST_MATH=ON -DMKL_WITH_OPENMP=ON -DOPENCV_ENABLE_NONFREE=ON -DWITH_OPENMP=ON -DWITH_QT=ON \
  -DWITH_OPENEXR=ON -DENABLE_PRECOMPILED_HEADERS=OFF -DBUILD_opencv_cudacodec=OFF -DINSTALL_PYTHON_EXAMPLES=OFF \
  -DWITH_TIFF=OFF -DWITH_WEBP=OFF -DWITH_FFMPEG=ON -DOPENCV_EXTRA_MODULES_PATH=../../opencv_contrib/modules \
  -DCMAKE_CXX_FLAGS=-std=c++17 -DENABLE_CXX11=OFF -DBUILD_opencv_xfeatures2d=OFF -DOPENCV_DNN_OPENCL=OFF \
  -DWITH_CUDA=ON -DWITH_OPENCL=OFF -DBUILD_opencv_wechat_qrcode=OFF -DCMAKE_CXX_STANDARD=17 \
  -DCMAKE_CXX_STANDARD_REQUIRED=ON -DOPENCV_CUDA_OPTIONS_opencv_test_cudev=-std=c++17 \
  -DCUDA_ARCH_BIN="7.0 7.5 8.0 8.6 9.0" -DCMAKE_INSTALL_PREFIX=/usr/local -DCMAKE_INSTALL_LIBDIR=lib \
  -DINSTALL_PKGCONFIG=ON -DOPENCV_GENERATE_PKGCONFIG=ON -DPKG_CONFIG_PATH=/usr/local/lib/pkgconfig \
  -DINSTALL_PYTHON_EXAMPLES=OFF -DINSTALL_C_EXAMPLES=OFF -DWITH_QT=OFF -DBUILD_opencv_hdf=ON
make -j$(nproc)
sudo make install
cd ../..
rm -rf opencv opencv_contrib

git clone --depth 1 --branch pcl-1.10.0 https://github.com/PointCloudLibrary/pcl
mkdir -p pcl/build
cd pcl/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_apps=OFF -DBUILD_GPU=OFF -DBUILD_CUDA=OFF -DBUILD_examples=OFF \
  -DBUILD_global_tests=OFF -DBUILD_simulation=OFF -DCUDA_BUILD_EMULATION=OFF -DCMAKE_CXX_FLAGS=-std=c++17 \
  -DPCL_ENABLE_SSE=ON -DPCL_SHARED_LIBS=ON -DWITH_VTK=OFF -DPCL_ONLY_CORE_POINT_TYPES=ON -DPCL_COMMON_WARNINGS=OFF
make -j$(nproc)
sudo make install
cd ../..
rm -rf pcl

git clone --depth 1 --branch v2.13.0 https://github.com/pybind/pybind11
mkdir -p pybind11/build
cd pybind11/build
cmake .. -DCMAKE_BUILD_TYPE=Release -DPYBIND11_INSTALL=ON -DPYBIND11_TEST=OFF
make -j$(nproc)
sudo make install
cd ../..
rm -rf pybind11

git clone --depth 1 --branch 0.8.0 https://github.com/jbeder/yaml-cpp
mkdir -p yaml-cpp/build
cd yaml-cpp/build
cmake .. -DCMAKE_POLICY_VERSION_MINIMUM=3.5 -DBUILD_TESTING=OFF -DCMAKE_BUILD_TYPE=Release \
  -DINSTALL_GTEST=OFF -DYAML_CPP_BUILD_TESTS=OFF -DYAML_BUILD_SHARED_LIBS=ON
make -j$(nproc)
sudo make install
cd ../..
rm -rf yaml-cpp

conda activate mono-artgs
pip3 install --upgrade pip setuptools wheel
pip3 install --no-cache-dir kaolin==0.17.0 -f https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.4.0_cu118.html
pip3 install "git+https://github.com/facebookresearch/pytorch3d.git@stable"
pip3 install --break-system-packages --force-reinstall blinker
pip3 install trimesh wandb matplotlib imageio tqdm open3d ruamel.yaml sacred kornia pymongo pyrender jupyterlab ninja "Cython>=0.29.37" yacs
pip3 install scipy scikit-learn
pip3 install numpy==1.26.4 transformations einops scikit-image awscli-plugin-endpoint gputil xatlas pymeshlab rtree dearpygui pytinyrenderer PyQt5 cython-npm chardet openpyxl

export CUDA_HOME=/usr/local/cuda
export LD_LIBRARY_PATH=/usr/local/cuda/lib64
export OPENCV_IO_ENABLE_OPENEXR=1
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS=1

python3 -c "import imageio; imageio.plugins.freeimage.download()"



export LD_PRELOAD=/usr/lib/x86_64-linux-gnu/libstdc++.so.6



ROOT="reconstruction/BundleSDF"

# Set PyTorch library path
export LD_LIBRARY_PATH="/usr/local/lib/python3.10/dist-packages/torch/lib:$LD_LIBRARY_PATH"
export TORCH_LIBRARIES="/usr/local/lib/python3.10/dist-packages/torch/lib"

# Additional PyTorch environment variables
export TORCH_CUDA_ARCH_LIST="7.0;7.5;8.0;8.6;9.0"
export FORCE_CUDA=1
export TORCH_EXTENSIONS_DIR="/tmp/torch_extensions"

# Ensure PyTorch can be found
export PYTHONPATH="/usr/local/lib/python3.10/dist-packages:$PYTHONPATH"

# Print debug info
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "PYTHONPATH: $PYTHONPATH"
echo "Testing PyTorch import..."
python3 -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"

cd ${ROOT}/mycuda && rm -rf build *egg* && python3 -m pip install -e . --no-build-isolation
cd ${ROOT}/BundleTrack && rm -rf build && mkdir build && cd build && /usr/bin/cmake .. -DFLANN_INCLUDE_DIR=/usr/include/flann -DFLANN_LIBRARY=/usr/lib/x86_64-linux-gnu/libflann.so -DCMAKE_PREFIX_PATH=/usr/local/lib/cmake/yaml-cpp -Dpybind11_DIR=$(python3 -m pybind11 --cmakedir) && make -j11