rm -rf build/
mkdir build
clang++ -std=c++11 -g -o  build/hf.out code/hartree-fock++.cc  -lint2 -lpthread
clang++ -std=c++11 -g -o  build/get_1elec_int.out code/Create1ParticleOperator.cc  -lint2 -lpthread
g++ -std=c++14 -g -o  build/get_2elec_int.out code/Create2ParticleOperator.cc  -lint2 -lpthread -lxerus -lxerus_misc
g++ -std=c++14 -g -o  build/basistransform.out code/BasisTransform.cpp  -lint2 -lpthread -lxerus -lxerus_misc
g++ -std=c++14 -g -o  build/builhamil.out code/BuildHamiltonian.cpp   -lxerus -lxerus_misc



