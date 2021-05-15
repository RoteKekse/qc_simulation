#include <xerus.h>

#include "classes/loading_tensors.cpp"
#include "classes/helpers.cpp"


using namespace xerus;
using xerus::misc::operator<<;


int main(){
	std::string name1,name2, name3;
	TTOperator H1, H2;
	TTTensor phi;

	name1 = "data/h2o_48_double_H.ttoperator";
	name2 = "data/hamiltonian_H2O_48_full_benchmark.ttoperator";
	name3 = "data/h2o_48_double_r64_f_i25_phi.tttensor";

	read_from_disc(name1,H1);
	read_from_disc(name2,H2);
	read_from_disc(name3,phi);
	Tensor res;
	Index ii,jj,kk;
	XERUS_LOG(info,phi.frob_norm());
	XERUS_LOG(info,H1.order());
	XERUS_LOG(info,H2.order());
	XERUS_LOG(info,phi.order());

	Tensor nuc;
	name1 = "data/h2o_48_double_nuc.tensor";
	read_from_disc(name1,nuc );


	XERUS_LOG(info,contract_TT(H1,phi,phi)+nuc[0]);

	XERUS_LOG(info,contract_TT(H2,phi,phi)+nuc[0]);
	size_t d = 48;
	for (size_t i = 0; i < d; i++){
		for (size_t j = 0; j < d; j++){
			for (size_t k = 0; k < d; k++){
				for (size_t l = 0; l < d; l++){
					std::vector<size_t> idx(2*d,0):
					idx[i,j,d+k,d+l] = 1;
					value_t val1 = H1[idx];
					value_t val2 = H2[idx];
					if (val1 != val2){
						XERUS_LOG(info,idx);
					}
				}
			}
		}
	}






	return 0;
}
