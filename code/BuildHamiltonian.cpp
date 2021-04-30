#include <xerus.h>
#include <classes/helpers.cpp>
#include <classes/loading_tensors.cpp>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>

using namespace xerus;
using xerus::misc::operator<<;

value_t returnTValue(Tensor T, size_t p, size_t q);
value_t returnVValue(Tensor V, size_t i, size_t k, size_t j, size_t l);
TTOperator buildHamil(Tensor &T, Tensor &V);
Tensor V11f(size_t i,size_t d);
Tensor V22f(size_t i,size_t d);
size_t getsizeV11(size_t i);
size_t getsizeV22(size_t i,size_t d);
Tensor V12f(size_t n, Tensor T, Tensor V);
Tensor V21f(size_t n,Tensor T, Tensor V);
Tensor MVf(Tensor T, Tensor V);

value_t getV(Tensor V,size_t i, size_t j, size_t k, size_t l);
value_t getT(Tensor T,size_t i, size_t j);
xerus::Tensor make_H_CH2(size_t nob);
xerus::Tensor make_V_CH2(size_t nob);
TTOperator BuildHamil(Tensor T, Tensor V);

int main(int argc, char* argv[]) {

	Tensor T ,V;
	TTOperator H_bench;
	size_t nob = 13;
	T = make_H_CH2(nob);
	V = make_V_CH2(nob);


	read_from_disc("data/hamiltonian_CH2_26_full.ttoperator", H_bench);

	XERUS_LOG(info, "T dimensions  " << T.dimensions);
	XERUS_LOG(info, "V dimensions  " << V.dimensions);

	auto H = BuildHamil(T,V);
	XERUS_LOG(info, "H_bench ranks " << H_bench.ranks());
	XERUS_LOG(info, "H ranks       " << H.ranks());
	H.round(0.0);
	XERUS_LOG(info, "H ranks       " << H.ranks());

	for (size_t i = 0;i < 2*nob;++i){
		std::vector<size_t> idx(4*nob,0);
		idx[i] = 1; idx[i+2*nob] = 1;
		XERUS_LOG(info,H[idx]<< " " <<H_bench[idx]);
	}


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

TTOperator BuildHamil(Tensor T, Tensor V){
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
	comp[{0,1,0,i+1}] = 1;
	comp[{0,0,1,2*(i+1)}] = 1;
	comp[{0,1,1,2*(i+1)+1+i*i}] = 1;
	for(size_t j = 0;j<i;++j){
		comp[{j+1,0,0,j+1}] = 1; comp[{j+1,1,1,j+1}] = -1;
		comp[{j+i+1,0,0,j+i+2}] = 1; comp[{j+i+1,1,1,j+i+2}] = -1;
		comp[{j+1,0,1,j+1+i*i+2*(i+1)+1}] = 1;
		comp[{j+1,1,0,j+1+i*i+ 4*(i+1)+i*(i-1)/2-1}] = 1;
		comp[{j+i+1,1,0,j+1+i*i+ 3*(i+1)}] = 1;
		comp[{j+i+1,0,1,j+1+i*i+ 5*(i+1)+i*(i-1)-2}] = 1;
	}
	for(size_t j = 0; j < i*i;++j){
		comp[{j+1+2*i,0,0,j+1+2*(i+1)}] = 1;comp[{j+1+2*i,1,1,j+1+2*(i+1)}] = 1;
	}
	for(size_t j = 0;  j <i*(i-1)/2; ++j){
		comp[{j+1+2*i+i*i,0,0,j+i*i+ 4*(i+1)}] = 1; comp[{j+1+2*i+i*i,1,1,j+i*i+ 4*(i+1)}] = -1;
		comp[{j+1+2*i+i*i+i*(i-1)/2,0,0,j+1+i*i+ 5*(i+1)+i*(i-1)/2-2}] = 1; comp[{j+1+2*i+i*i+i*(i-1)/2,1,1,j+1+i*i+ 5*(i+1)+i*(i-1)/2-2}] = -1;
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
	comp[{0,0,1,getsizeV22(i,d)-1}] = 1;
	comp[{d-i,1,0,getsizeV22(i,d)-1}] = 1;
	comp[{getsizeV22(i-1,d)-1,0,0,getsizeV22(i,d)-1}] = 1;comp[{getsizeV22(i-1,d)-1,1,1,getsizeV22(i,d)-1}] = 1;
	for(size_t j = 0; j < d-i-1;++j){
		comp[{j+1,0,0,j}] = 1; comp[{j+1,1,1,j}] = -1;
		comp[{d-i+1+j,0,0,d-i+j-1}] = 1; comp[{d-i+1+j,1,1,d-i+j-1}] = -1;
	}
	if (reverse){
		Index i1,j1,k1,l1;
		comp(i1,k1,l1,j1) = comp(j1,k1,l1,i1);
		return comp;
	}
	return comp;
}

Tensor V12f(size_t n, Tensor T, Tensor V){
    size_t d = 2*V.dimensions[0];
	//XERUS_REQUIRE(n>=1,"n=0 doesent work");
    Tensor comp({getsizeV11(n-1),2,2,getsizeV22(n,d)});

    comp[{0,1,1,getsizeV22(n,d)-1}] = getT(T,n,n); // A^*A
    for (size_t i = 0; i < n;++i){
    	comp[{1+i,0,1,getsizeV22(n,d)-1}] = getT(T,i,n); // A
        for (size_t l = n+1; l < d; ++l)
            comp[{1+i,1,1,l-(n+1)}] = getV(V,i,n,n,l); //(val,:AtAplus)
    }
	for (size_t l = 0; l <n;++l){
    	comp[{l+n+1,1,0,getsizeV22(n,d)-1}] = getT(T,n,l); // A^*
        for (size_t j = n+1; j<d;++j)
            comp[{l+n+1,1,1,d-2*(n+1)+j}] = getV(V,n,j,n,l); //  (val,:AtAminus)
	}
    size_t count = 0;
    std::vector<std::pair<size_t,size_t>> list;
    for (size_t i = 0; i < n;++i){
        list.emplace_back(std::pair<size_t,size_t>(i,i));
        for (size_t k = 0; k < i;++k)
        	list.emplace_back(std::pair<size_t,size_t>(k,i));
        for (size_t k = 0; k < i;++k)
        	list.emplace_back(std::pair<size_t,size_t>(i,k));
    }
    for (auto pair : list){
    	auto i = pair.first;
    	auto k = pair.second;
        for (size_t l =n+1; l<d; ++l)
            comp[{count+ 2*n+1,1,0,l-(n+1)}]  =getV(V,i,n,k,l);//  (val,:Alrstar)
        for (size_t j = n+1; j<d;++j)
            comp[{count+ 2*n+1,0,1,d-2*(n+1)+j}] =getV(V,i,j,k,n);//  (val,:Alr)
        comp[{count+ 2*n+1,1,1,getsizeV22(n,d)-1}]= getV(V,i,n,k,n); // (val,:AtAr)
        count++;
    }

    count = 0;
    for (size_t j = 1; j < n;++j){
        for (size_t i = 0; i < j;++i){
            for (size_t l = n+1; l<d;++l)
                comp[{count+ 2*n+1+n*n,0,1,l-(n+1)}] =getV(V,i,j,n,l); //  (val, :Arm)
            count++;
        }
    }

    count = 0;
    for (size_t l = 1; l<n;++l){
        for (size_t k = 0; k <l;++k){
            for (size_t j = n+1; j < d;++j)
                comp[{count+ 2*n+1+n*n+n*(n-1)/2,1,0,d-2*(n+1)+j}]=getV(V,n,j,k,l); //  (val, :Armstar)
            count++;
        }
    }
    return comp;
}

Tensor V21f(size_t n,Tensor T, Tensor V){
    size_t d = 2*V.dimensions[0];


    size_t n1 = d - n - 1;
    Tensor comp({getsizeV11(n1-1),2,2,getsizeV22(n1,d)});

    size_t counti = 0;
    comp[{0,1,1,getsizeV22(n1,d)-1}] = getT(T,n,n); // A^*A
    for (size_t i = d-1; i>n; --i){
        size_t countl = 0;
        comp[{1+counti,0,1,getsizeV22(n1,d)-1}] = getT(T,i,n); // A
        for (size_t l = n;l>0; --l){
            comp[{1+counti,1,1,countl}] = getV(V,i,n,n,l-1) ;//  (val, :AtAminus)
            countl++;
		}
        counti++;
    }
    size_t countl = 0;
    for (size_t l = d-1; l>n;--l){
        size_t countj = 0;
        comp[{countl+n1+1,1,0,getsizeV22(n1,d)-1}] = getT(T,n,l); // A^*
        for (size_t j = n; j> 0;--j){
            comp[{countl+n1+1,1,1,n+countj}] =  getV(V,n,j-1,n,l); // :  (val,:AtAplus)
            countj++;
		}
        countl++;
    }
    size_t count = 0;
    std::vector<std::pair<size_t,size_t>> list;
    for (size_t i = d-1;i>n;--i){
        list.emplace_back(std::pair<size_t,size_t>(i,i));
        for (size_t k = d-1; k> i;--k)
            list.emplace_back(std::pair<size_t,size_t>(k,i));
        for (size_t k = d-1; k> i;--k)
            list.emplace_back(std::pair<size_t,size_t>(i,k));
    }

    for (auto pair : list){
		auto i = pair.first;
		auto k = pair.second;
        size_t countl = 0;
        for (size_t l = n;l>0;--l){
            comp[{count+ 2*n1+1,1,0,countl}] =  getV(V,i,n,k,l-1); //  (val,:Arlstar)
            countl++;
        }
        size_t countj = 0;
        for (size_t j = n;j>0;--j){
            comp[{count+ 2*n1+1,0,1,n+countj}] =getV(V,i,j-1,k,n); // (val, :Arl)
            countj++;
        }
        comp[{count+ 2*n1+1,1,1,getsizeV22(n1,d)-1}] = getV(V,i,n,k,n); //  (val,:AtAl)
        count++;
    }

    count = 0;
    for (size_t j = d-2; j>n;--j){
        for (size_t i = d-1;i>j;--i){
            size_t countl = 0;
            for (size_t l = n;l>0;--l){
                comp[{count+ 2*n1+1+n1*n1,0,1,countl}] = getV(V,i,j,n,l-1);//  (val,:Alm)
                countl++;
            }
            count++;
        }
    }

    count = 0;
    for (size_t l = d-2;l>n;--l){
        for (size_t k = d-1; k>l;--k){
            size_t countj=0;
            for (size_t j = n;j>0;--j){
                comp[{count+ 2*n1+1+n1*n1+n1*(n1-1)/2,1,0,n+countj}] = getV(V,n,j-1,k,l);//  (val,:Almstar)
                countj++;
            }
            count++;
        }
    }
	Index i1,i2,j1,j2;
	comp(i1,i2,j1,j2) = comp(j2,i2,j1,i1);
    return comp;
}

Tensor MVf(Tensor T, Tensor V){
    size_t d   = 2*V.dimensions[0];//2*V.dimensions[0];
    size_t count,countr, countl;
    size_t n = getsizeV11(d/2-1)+getsizeV22(d/2-1,d);
    Tensor MV({n,n});

    MV[{n-1,0}] = 1;
    MV[{0,n-1}] = 1;
    for (size_t i=0; i< d/2;++i){
        MV[{1+i,n-i-2}] = 1;
        MV[{1+d/2+i,n-2-i-d/2}] = 1;
        MV[{n-2-i,1+i}] = 1;
        MV[{n-2-i-d/2,1+d/2+i}] = 1;
    }

     for (size_t i=0; i< d/2;++i){
    	 countr = 0;
         for (size_t j=d-1; j>= d/2;--j){
     		MV[{1+d/2+i,1+countr}] = getT(T,j,i);
      		countr++;
         }
     }

     for (size_t i=0; i< d/2;++i){
    	 countr=0;
         for (size_t j=d-1; j>= d/2;--j){
     		MV[{1+i,1+d/2 +countr}] = getT(T,i,j);
     		countr++;
         }
     }

    std::vector<std::pair<size_t,size_t>> listl;
    for (size_t i = 0; i< d/2;++i){
    	listl.emplace_back(std::pair<size_t,size_t>(i,i));
    	for (size_t k = 0; k < i;++k)
    		listl.emplace_back(std::pair<size_t,size_t>(k,i));
    	for (size_t k = 0; k < i;++k)
    		listl.emplace_back(std::pair<size_t,size_t>(i,k));
    }

    std::vector<std::pair<size_t,size_t>> listr;
    for (size_t i = d-1; i >= d/2;--i){
    	listr.emplace_back(std::pair<size_t,size_t>(i,i));
        for (size_t k = d-1; k > i;--k)
        	listr.emplace_back(std::pair<size_t,size_t>(k,i));
        for (size_t k = d-1; k > i;--k)
        	listr.emplace_back(std::pair<size_t,size_t>(i,k));
    }

    countl = 0;
    for (auto pair1 : listl){
		auto i = pair1.first;
		auto k = pair1.second;
	    countr = 0;
	    for (auto pair2 : listr){
	    	auto j = pair2.first;
	    	auto l = pair2.second;
            MV[{countl+1+d,countr+1+d}] = getV(V,i,j,k,l);
            countr++;
	    }
        countl++;
    }



    countl = 0;
    for (size_t j = 1; j < d/2;++j){
        for (size_t i = 0;i < j;++i){
            countr=0;
            for (size_t k = d-2; k >= d/2;--k){
                for (size_t l = d-1;l>k;--l){
                    MV[{countl+1+d+(d*d)/4,countr+1+d+(d*d)/4+((d/2)*(d/2-1))/2}] = getV(V,i,j,k,l);
                    countr++;
                }
            }
            countl++;
        }
    }

    countl = 0;
    for (size_t j = d-2;j>=d/2;--j){
        for (size_t i = d-1;i>j;--i){
            countr=0;
            for (size_t k = 1;k< d/2;++k){
                for (size_t l = 0; l<k;++l){
                    MV[{countr+1+d+(d*d)/4+((d/2)*(d/2-1))/2,countl+1+d+(d*d)/4}] = getV(V,i,j,k,l);
                    countr++;
                }
            }
            countl++;
        }
    }

    return MV;
}


size_t getsizeV11(size_t i){
    return 1+2*(i+1)+(i+1)*(i+1)+(i+1)*(i);
}

size_t getsizeV22(size_t i,size_t d){
    return 1+2*(d-i-1);
}
value_t getV(Tensor V,size_t i, size_t j, size_t k, size_t l){
	value_t val = returnVValue(V,i,j,k,l)+returnVValue(V,j,i,l,k)-returnVValue(V,j,i,k,l)-returnVValue(V,i,j,l,k);
	bool flip = (i < j && k < l) || (j < i && l < k);
	//value_t val = 100000+1000*i+100*j+10*k+l;
	//return Tensor::random({1})[0];
    return flip ?  -0.5*val : 0.5*val;
}
value_t getT(Tensor T,size_t i, size_t j){
	value_t val = returnTValue(T, i, j);
	//value_t val = 100000+10*i+j;
	//return Tensor::random({1})[0];
	return val;
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

