#include <xerus.h>
#include <classes/helpers.cpp>
#include <classes/loading_tensors.cpp>
#include <classes/hamiltonian.cpp>



using namespace xerus;
using xerus::misc::operator<<;



int main(int argc, char* argv[]) {

	const auto geom = argv[1];
	const auto basisname = argv[2];
	value_t shift = std::atof(argv[3]);

	Tensor T ,V;
	TTOperator H_bench;
	std::string name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_T.tensor";
	//std::string name = "data/T_H2O_48_bench_single.tensor";
	read_from_disc(name, T);
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_V.tensor";
	//name = "data/V_H2O_48_bench_single.tensor";
	read_from_disc(name, V);

	auto H = BuildHamil(T,V,shift);
	size_t d = H.order()/2;
	std::vector<size_t> hf = {0,1,2,3,4,5,6,7,8,9,10,11,12,13};
	//std::vector<size_t> hf = {0,1,2,3,22,23,30,31};
	std::vector<size_t> idx(2*d,0);
	for (auto  i : hf){
		idx[i] = 1;
		idx[i+d] = 1;
	}
	XERUS_LOG(info, H[idx]);

	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_H.ttoperator";
	write_to_disc(name, H);

}


