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

	res() = phi(ii&0)*H1(ii/2,jj/2)*phi(jj/2);

	XERUS_LOG(info,res);
	res() = phi(ii&0)*H2(ii/2,jj/2)*phi(jj/2);

	XERUS_LOG(info,res);





	return 0;
}
