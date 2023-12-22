rm -rf ./build/

wait

engine=$1
problem=$2
device=$3

if test "${engine}" == "pic"; then
    metric="minkowski"
elif test "${engine}" == "grpic"; then
    metric="qkerr_schild"
fi

wait

if test "${device}" == "cpu"; then
    cuda="OFF"
    openmp="ON"
elif test "${device}" == "gpu"; then
    cuda="ON"
    openmp="OFF"
fi

wait

cmake -B build -D engine=${engine} -D pgen=${problem} -D metric=${metric} -D precision=single -D out$

wait

cd build/
make -j

wait 

cd ..
