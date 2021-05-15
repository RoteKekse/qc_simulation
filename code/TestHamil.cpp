#include <xerus.h>

#include "classes/loading_tensors.cpp"
#include "classes/helpers.cpp"


using namespace xerus;
using namespace Eigen;
using xerus::misc::operator<<;


int main(){
	std::string name1,name2;
	TTOperator H1, H2;

	name1 = "data/h2o_48_double_H.ttoperator";
	name2 = "data/hamiltonian_H2O_48_full_benchmark.ttoperator";

	read_from_disc(name1,H1);
	read_from_disc(name2,H2);

	XERUS_LOG(info,(H1-H2).frob_norm());





	return 0;
}
