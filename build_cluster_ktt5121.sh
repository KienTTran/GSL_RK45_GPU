#!/usr/bin/env bash
mkdir -p build
cd build

export PATH=/usr/lpp/mmfs/bin:/usr/local/bin:/usr/local/sbin:/usr/lib64/qt-3.3/bin:/opt/moab/bin:/opt/mam/bin:/usr/local/bin:/usr/bin:/usr/local/sbin:/usr/sbin:/opt/ibutils/bin:/opt/puppetlabs/bin:/storage/home/ktt5121/bin:/usr/local/cuda
export LIBRARY_PATH=/storage/home/ktt5121/work/build_env/postgres/lib:/storage/home/ktt5121/work/build_env/lib/lib:

module load cuda/11.1.0
module load gcc/8.3.1
module load cmake/3.18.4
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_TOOLCHAIN_FILE=/storage/home/ktt5121/work/build_env/vcpkg/scripts/buildsystems/vcpkg.cmake -DBUILD_CLUSTER:BOOL=true ..
make -j 8
