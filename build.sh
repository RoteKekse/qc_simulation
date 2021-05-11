rm -rf build/
mkdir build

echo "Build HF"
clang++ -std=c++11 -g -o  build/hf.out code/hartree-fock++.cc  -lint2 -lpthread
echo "1 elec"
clang++ -std=c++11 -g -o  build/get_1elec_int.out code/Create1ParticleOperator.cc  -lint2 -lpthread
echo "2 elec"
g++ -std=c++14 -g -o  build/get_2elec_int.out code/Create2ParticleOperator.cc  -lint2 -lpthread -lxerus -lxerus_misc
echo "Transform Basis"
g++ -std=c++14 -g -o  build/basistransform.out code/BasisTransform.cpp  -lint2 -lpthread -lxerus -lxerus_misc
echo "Build Hamil"
g++ -std=c++14 -g -o  build/buildhamil.out code/BuildHamiltonian.cpp   -lxerus -lxerus_misc
echo "Build Diag"
g++ -std=c++14 -g -o  build/buildhamildiag.out code/BuildHamiltonianDiag.cpp   -lxerus -lxerus_misc
echo "Build Precon"
g++ -g -std=c++14  -o build/buildprecon.out code/BuildPrecon.cpp   -lxerus -lxerus_misc 

echo "Build DMRG"
g++ -g -std=c++14 -fext-numeric-literals -DARPACK_LIBRARIES -o build/dmrg.out code/RunDmrg.cpp   -lxerus -lxerus_misc -lboost_regex -fopenmp

echo "Build PPGD"
g++ -g -std=c++14 -o build/ppgd.out code/PPGD_CG.cpp   -lxerus -lxerus_misc  -fopenmp



