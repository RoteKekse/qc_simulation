#include <xerus.h>
#include <classes/helpers.cpp>
#include <classes/loading_tensors.cpp>

using namespace xerus;
using xerus::misc::operator<<;

value_t returnTValue(size_t p, size_t q);
value_t returnVValue(size_t i, size_t k, size_t j, size_t l);
TTOperator buildHamil(Tensor &T, Tensor &V);
Tensor V11f(size_t i,size_t d);
Tensor V22f(size_t i,size_t d);
size_t getsizeV11(size_t i);
size_t getsizeV22(size_t i,size_t d);

int main(int argc, char* argv[]) {
	auto test1 = V11f(0,8);
	auto test2 = V22f(0,8);
	XERUS_LOG(info,getsizeV11(0));
	XERUS_LOG(info,getsizeV22(2,8));
	XERUS_LOG(info,getsizeV22(3,8));
	XERUS_LOG(info,test1.dimensions << "\n" << test1);
	XERUS_LOG(info,test2.dimensions << "\n" << test2);


//	const auto geom = argv[1];
//	// Set basis functions
//	const auto basisname = argv[2];
//
//	auto T = xerus::Tensor();
//	auto V = xerus::Tensor();
//
//	std::string name;
//	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_T.tensor";
//	read_from_disc(name,T);
//	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_V.tensor";
//	read_from_disc(name,V);
//
//
//
//	H = buildHamil(T,V);
//	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_H.ttoperator";
//	write_to_disc(name,V);
//
//	return 0;
//}
//TTOperator buildHamil(Tensor &T, Tensor &V){
//	size d = 2*T.dimensions[0];
//	TTOperator H(std::vector<size_t>(d,2));
//
//	return H;
}

Tensor V11f(size_t i,size_t d){
	bool reverse = i < d/2 ? false : true;
	if (reverse)
		i = d-1-i;
	Tensor comp({i == 0 ? 1 : getsizeV11(i-1),2,2,getsizeV11(i)});
	comp[{0,0,0,0}] = 1; comp[{0,1,1,0}] = 1;
	comp[{0,1,0,i+1}] = 1;
	comp[{0,0,1,2*(i+1)}] = 1;
	comp[{0,1,1,2*(i+1)+1+i*i}] = 1;
	for(size_t j = 0;j<i;++j){
		comp[j+1,0,0,j+1] = 1; comp[j+1,1,1,j+1] = -1;
		comp[j+i+1,0,0,j+i+1] = 1; comp[j+i+1,1,1,j+i+1] = -1;
		comp[j+1,0,1,j+1+i*i+2*(i+1)+1] = 1;
		comp[j+1,1,0,j+1+i*i+ 4*(i+1)+i*(i-1)/2-1] = 1;
		comp[j+i+1,1,0,j+1+i*i+ 3*(i+1)] = 1;
		comp[j+i+1,0,1,j+1+i*i+ 5*(i+1)+i*(i-1)-2] = 1;
	}
	for(size_t j = 0; j < i*i;++j){
		comp[j+1+2*i,0,0,j+1+2*(i+1)] = 1;comp[j+1+2*i,1,1,j+1+2*(i+1)] = 1;
	}
	for(size_t j = 0;  j <i*(i-1)/2; ++j){
		comp[j+1+2*i+i*i,0,0,j+i*i+ 4*(i+1)] = 1; comp[j+1+2*i+i*i,1,1,j+i*i+ 4*(i+1)] = -1;
		comp[j+1+2*i+i*i+i*(i-1)/2,0,0,j+1+i*i+ 5*(i+1)+i*(i-1)/2-2] = 1; comp[j+1+2*i+i*i+i*(i-1)/2,1,1,j+1+i*i+ 5*(i+1)+i*(i-1)/2-2] = -1;
	}

	if (reverse){
		Index i1,j1;
		comp(i1,j1) = comp(j1,i1);
		return comp;
	}
	return comp;
}

Tensor V22f(size_t i,size_t d){
	bool reverse = i < d/2 ? false : true;
	if (reverse)
		i = d-1-i;
	Tensor comp({i == 0 ? 1 : getsizeV22(i-1,d),2,2,getsizeV22(i,d)});
	comp[0,0,1,getsizeV22(i,d)-1] = 1;
	comp[d-i+1,1,0,getsizeV22(i,d)-1] = 1;
	comp[getsizeV22(i,d)-1,0,0,getsizeV22(i,d)-1] = 1;comp[getsizeV22(i,d)-1,1,1,getsizeV22(i,d)-1] = 1;
	for(size_t j = 0; j < d-i;++j){
		comp[j+1,0,0,j] = 1; comp[j+2,1,1,j+1] = -1;
		comp[d-i+2+j,0,0,d-i+j] = 1; comp[d-i+2+j,1,1,d-i+j] = -1;
	}
	if (reverse){
		Index i1,j1;
		comp(i1,j1) = comp(j1,i1);
		return comp;
	}
	return comp;
}


size_t getsizeV11(size_t i){
    return 1+2*(i+1)+(i+1)*(i+1)+(i+1)*(i);
}

size_t getsizeV22(size_t i,size_t d){
    return 1+2*(d-i-1);
}



value_t returnTValue(Tensor T, size_t i, size_t j)
{
	if (i%2 != j%2)
		return 0;
	i /=2;j /=2;
	if (i > j)
		return T[{i , j }];
	return T[{j , i }];
}


value_t returnVValue(Tensor V, size_t i, size_t k, size_t j, size_t l){
	if ((i%2 != j%2) || (k%2!=l%2))
		return 0;
	i /=2;k /=2;j /=2;l /=2;

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
