#include <xerus.h>
#include <classes/helpers.cpp>
#include <classes/loading_tensors.cpp>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

using namespace xerus;
using xerus::misc::operator<<;

value_t returnTValue(Tensor &T, size_t p, size_t q);
value_t returnVValue(Tensor &V, size_t i, size_t k, size_t j, size_t l);
TTOperator BuildHamilDiag(Tensor &T, Tensor &V);
Tensor V11f(size_t i,size_t d);
Tensor V22f(size_t i,size_t d);
size_t getsizeV11(size_t i);
size_t getsizeV22(size_t i,size_t d);
Tensor V12f(size_t n, Tensor &T, Tensor &V);
Tensor V21f(size_t n,Tensor &T, Tensor &V);
Tensor MVf(Tensor &T, Tensor &V);

value_t getV(Tensor &V,size_t i, size_t j, size_t k, size_t l);
value_t getT(Tensor &T,size_t i, size_t j);
xerus::Tensor make_H_CH2(size_t nob);
xerus::Tensor make_V_CH2(size_t nob);
TTOperator BuildHamil(Tensor &T, Tensor &V);

int main(int argc, char* argv[]) {

	Tensor T ,V;
	TTOperator H_bench;
	size_t nob = 24;
//	T = make_H_CH2(nob);
//	V = make_V_CH2(nob);
//

	read_from_disc("data/T_H2O_48_bench_single.tensor", T);
	read_from_disc("data/V_H2O_48_bench_single.tensor", V);

	XERUS_LOG(info, "T dimensions  " << T.dimensions);
	XERUS_LOG(info, "V dimensions  " << V.dimensions);

	auto H = BuildHamilDiag(T,V);

	write_to_disc("data/hamiltonian_diag_H2O_48_full_new.ttoperator", H);
	XERUS_LOG(info, "H ranks       " << H.ranks());
	H.round(0.0);
	XERUS_LOG(info, "H ranks       " << H.ranks());




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

TTOperator BuildHamilDiag(Tensor &T, Tensor &V){
    size_t d = 2*V.dimensions[0];
    TTOperator H(std::vector<size_t>(2*d ,2));

    auto comp1 = V11f(0,d);
	auto comp2 = V12f(0,T,V);

	Tensor comp({comp1.dimensions[0],2,2,comp1.dimensions[3]+comp2.dimensions[3]});
	comp.offset_add(comp1,{0,0,0,0});
	comp.offset_add(comp2,{0,0,0,comp1.dimensions[3]});
	comp.use_sparse_representation();
	H.set_component(0,comp);

	for (size_t i = 1; i < d/2;++i){
		XERUS_LOG(info,i);
		comp1 = V11f(i,d);
		comp2 = V12f(i,T,V);
		auto comp3 = V22f(i,d);
		comp = Tensor({comp1.dimensions[0]+comp3.dimensions[0],2,2,comp1.dimensions[3]+comp3.dimensions[3]});
		comp.offset_add(comp1,{0,0,0,0});
		comp.offset_add(comp2,{0,0,0,comp1.dimensions[3]});
		comp.offset_add(comp3,{comp1.dimensions[0],0,0,comp1.dimensions[3]});
		H.set_component(i,comp);
	}
	comp1 = V11f(d-1,d);
	comp2 = V21f(d-1,T,V);
	comp = Tensor({comp1.dimensions[0]+comp2.dimensions[0],2,2,comp1.dimensions[3]});
	comp.offset_add(comp1,{0,0,0,0});
	comp.offset_add(comp2,{comp1.dimensions[0],0,0,0});
	comp.use_sparse_representation();
	H.set_component(d-1,comp);
	for (size_t i = d-2; i >= d/2;--i){
		XERUS_LOG(info,i);

		comp1 = V11f(i,d);
		comp2 = V21f(i,T,V);
		auto comp3 = V22f(i,d);
		comp = Tensor({comp1.dimensions[0]+comp3.dimensions[0],2,2,comp1.dimensions[3]+comp3.dimensions[3]});
		comp.offset_add(comp1,{0,0,0,0});
		comp.offset_add(comp2,{comp1.dimensions[0],0,0,0});
		comp.offset_add(comp3,{comp1.dimensions[0],0,0,comp1.dimensions[3]});
		H.set_component(i,comp);
	}
	auto M = MVf(T,V);
	Index i,j,k,l,m;
	Tensor tmp;
	tmp(i,j,k,m) = H.get_component(d/2-1)(i,j,k,l)*M(l,m);
	H.set_component(d/2-1,tmp);

    return H;
}

Tensor V11f(size_t i,size_t d){
	bool reverse = i < d/2 ? false : true;
	if (reverse)
		i = d-1-i;
	Tensor comp({i == 0 ? 1 : getsizeV11(i-1),2,2,getsizeV11(i)});

	comp[{0,0,0,0}] = 1; comp[{0,1,1,0}] = 1;
	comp[{0,1,1,getsizeV11(i)-1}] = 1;

	for(size_t j = 0; j < i;++j){
		comp[{j+1,0,0,j+1}] = 1;comp[{j+1,1,1,j+1}] = 1;
	}

	if (reverse){
		Index i1,j1,k1,l1;
		comp(i1,k1,l1,j1) = comp(j1,k1,l1,i1);
		return comp;
	}
	return comp;
}

Tensor V22f(size_t i,size_t d){
	bool reverse = i < d/2 ? false : true;
	if (reverse)
		i = d-1-i;
	Tensor comp({i == 0 ? 1 : getsizeV22(i-1,d),2,2,getsizeV22(i,d)});
	comp[{getsizeV22(i-1,d)-1,0,0,getsizeV22(i,d)-1}] = 1;comp[{getsizeV22(i-1,d)-1,1,1,getsizeV22(i,d)-1}] = 1;
	if (reverse){
		Index i1,j1,k1,l1;
		comp(i1,k1,l1,j1) = comp(j1,k1,l1,i1);
		return comp;
	}
	return comp;
}

Tensor V12f(size_t n, Tensor &T, Tensor &V){
    size_t d = 2*V.dimensions[0];
	//XERUS_REQUIRE(n>=1,"n=0 doesent work");
    Tensor comp({getsizeV11(n-1),2,2,getsizeV22(n,d)});

    comp[{0,1,1,0}] = getT(T,n,n); // A^*A

    for (size_t i = 0; i < n; ++i){
        comp[{i+1,1,1,0}]= getV(V,i,n,i,n); // (val,:AtAr)
    }
    return comp;
}

Tensor V21f(size_t n,Tensor &T, Tensor &V){
    size_t d = 2*V.dimensions[0];


    size_t n1 = d - n - 1;
    Tensor comp({getsizeV11(n1-1),2,2,getsizeV22(n1,d)});

    size_t counti = 0;
    comp[{0,1,1,0}] = getT(T,n,n); // A^*A


    for (size_t i = d-1; i >n;--i){
        comp[{d-i,1,1,0}] = getV(V,i,n,i,n); //  (val,:AtAl)
    }
	Index i1,i2,j1,j2;
	comp(i1,i2,j1,j2) = comp(j2,i2,j1,i1);
    return comp;
}

Tensor MVf(Tensor &T, Tensor &V){
    size_t d   = 2*V.dimensions[0];//2*V.dimensions[0];
    size_t n = getsizeV11(d/2-1)+getsizeV22(d/2-1,d);
    Tensor MV({n,n});

    MV[{n-1,0}] = 1;
    MV[{0,n-1}] = 1;

    for (size_t i = 0; i < d/2;++i){
	    for (size_t j = d-1; j >= d/2;--j){
            MV[{i+1,d-j}] = getV(V,i,j,i,j);
	    }
    }
   return MV;
}


size_t getsizeV11(size_t i){
    return 2+i;
}

size_t getsizeV22(size_t i,size_t d){
    return 1;
}
value_t getV(Tensor &V,size_t i, size_t j, size_t k, size_t l){
	value_t val = returnVValue(V,i,j,k,l)+returnVValue(V,j,i,l,k)-returnVValue(V,j,i,k,l)-returnVValue(V,i,j,l,k);
	bool flip = (i < j && l < k) || (j < i && k < l);
	//value_t val = 100000+1000*i+100*j+10*k+l;
	//return Tensor::random({1})[0];
    return flip ?  -0.5*val : 0.5*val;
}
value_t getT(Tensor &T,size_t i, size_t j){
	value_t val = returnTValue(T, i, j);
	//value_t val = 100000+10*i+j;
	//return Tensor::random({1})[0];
	//return 0;
	return val;
}


value_t returnTValue(Tensor &T, size_t i, size_t j)
{
	if (i%2 != j%2)
		return 0;
	i /=2;j /=2;
	if (i > j)
		return T[{i , j }];
	return T[{j , i }];
}


value_t returnVValue(Tensor &V, size_t i, size_t k, size_t j, size_t l){
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

xerus::Tensor make_H_CH2(size_t nob){
	auto H = xerus::Tensor({nob,nob});
	std::string line;
	std::ifstream input;
	input.open ("FCIDUMP.ch2_13");
	size_t count = 0;
	while ( std::getline (input,line) )
	{
		count++;
		if (count > 4){
			std::vector<std::string> l;
			boost::algorithm::split_regex( l, line, boost::regex( "  " ) ) ;
			if (std::stoi(l[1]) != 0 && std::stoi(l[3]) == 0){
				H[{static_cast<size_t>(std::stoi(l[1]))-1,static_cast<size_t>(std::stoi(l[2]))-1}] = stod(l[0]);
	    }
		}
	}
	input.close();
	return H;
}


xerus::Tensor make_V_CH2(size_t nob){
	auto V = xerus::Tensor({nob,nob,nob,nob});
	std::string line;
	std::ifstream input;
	input.open ("FCIDUMP.ch2_13");
	size_t count = 0;
	while ( std::getline (input,line) )
	{
		count++;
		if (count > 4){
			std::vector<std::string> l;
			boost::algorithm::split_regex( l, line, boost::regex( "  " ) ) ;
			if (std::stoi(l[1]) != 0 && std::stoi(l[3]) != 0){
				V[{static_cast<size_t>(std::stoi(l[1]))-1,static_cast<size_t>(std::stoi(l[2]))-1,static_cast<size_t>(std::stoi(l[3]))-1,static_cast<size_t>(std::stoi(l[4]))-1}] = stod(l[0]);
			}
		}
	}

	return V;
}

