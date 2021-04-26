#include <xerus.h>
#include <../classes/helpers.cpp>
#include <../classes/loading_tensors.cpp>

using namespace xerus;
using xerus::misc::operator<<;

value_t returnTValue(size_t p, size_t q);
value_t returnVValue(size_t i, size_t k, size_t j, size_t l);
TTOperator buildHamil(Tensor &T, Tensor &V);

int main(int argc, char* argv[]) {
	/*
	 * !!!!! Change Here !!!!
	 */
	const auto geom = argv[1];
	// Set basis functions
	const auto basisname = argv[2];

	auto T = xerus::Tensor();
	auto V = xerus::Tensor();

	std::string name;
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_T.tensor";
	read_from_disc(name,T);
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_V.tensor";
	read_from_disc(name,V);



	H = buildHamil(T,V);
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_H.ttoperator";
	write_to_disc(name,V);

	return 0;
}
TTOperator buildHamil(Tensor &T, Tensor &V){
	size d = 2*T.dimensions[0];
	TTOperator H(std::vector<size_t>(d,2));

	return H;
}


value_t returnTValue(size_t p, size_t q)
{
	if (p > q)
		return T[{p , q }];
	return T[{q , p }];
}


value_t returnVValue(size_t i, size_t k, size_t j, size_t l){
	//XERUS_LOG(info, i<<j<<k<<l );
	if (j <= i){
		if (k<= i && l <= (i==k ? j : k))
			return V[{i,j,k ,l}];
		if (l<= i && k <= (i==l ? j : l))
			return V[{i,j,l ,k}];
	} else if (i <= j){
		if (k<= j && l <= (j==k ? i : k))
			return V[{j,i,k,l}];
		if (l<= j && k <= (j==l ? i : l))
			return V[{j,i,l,k}];
	}
	if (l <= k){
		if (i<= k && j <= (k==i ? l : i))
			return V[{k,l,i ,j}];
		if (j<= k && i <= (k==j ? l : j))
			return V[{k,l,j ,i}];
	} else if (k <= l) {
		if (i<= l && j <= (l==i ? k : i))
			return V[{l,k,i ,j}];
		if (j<= l && i <= (l==j ? k : j))
			return V[{l,k,j ,i}];
	}

	return 1.0;
}
