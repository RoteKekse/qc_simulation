#include <xerus.h>
#include <classes/helpers.cpp>
#include <classes/loading_tensors.cpp>

using namespace xerus;
using xerus::misc::operator<<;

value_t returnTValue(Tensor T, size_t p, size_t q);
value_t returnVValue(Tensor V, size_t i, size_t k, size_t j, size_t l);
TTOperator buildHamil(Tensor &T, Tensor &V);
Tensor V11f(size_t i,size_t d);
Tensor V22f(size_t i,size_t d);
size_t getsizeV11(size_t i);
size_t getsizeV22(size_t i,size_t d);
//Tensor V12f(size_t n, Tensor T, Tensor V);
Tensor V12f(size_t n,size_t d);
Tensor V21f(size_t n,Tensor T, Tensor V);
Tensor MVf(Tensor T, Tensor V);

value_t getV(Tensor V,size_t i, size_t j, size_t k, size_t l);



int main(int argc, char* argv[]) {
	auto test1 = V12f(3,8);

	Tensor T = Tensor::random({4,4});
	Tensor V = Tensor::random({4,4,4,4});
	auto test2 = V21f(4,T,V);
	auto test3 = MVf(T,V);

	XERUS_LOG(info,test1.dimensions << "\n"  << test1);
	XERUS_LOG(info,test2.dimensions << "\n"  << test2);
	XERUS_LOG(info,test3.dimensions << "\n"  << test3);




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
	XERUS_LOG(info,comp.dimensions);
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

//Tensor V12f(size_t n, Tensor T, Tensor V){
Tensor V12f(size_t n, size_t d){
    //size_t d = 2*V.dimensions[0]
	XERUS_REQUIRE(n>=1,"n=0 doesent work");
    Tensor comp({getsizeV11(n-1),getsizeV22(n,d)});

    for (size_t i = 0; i < n;++i){
        for (size_t l = n+1; l < d; ++l)
            comp[{1+i,l-(n+1)}] =1;// -getV(V,i,n,n,l); //(val,:AtAplus)
    }
	for (size_t l = 0; l <n;++l){
        for (size_t j = n+1; j<d;++j)
            comp[{l+n+1,d-2*(n+1)+j}] = 2;//getV(V,n,j,n,l); //  (val,:AtAminus)
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
    XERUS_LOG(info,list);
    for (auto pair : list){
    	auto i = pair.first;
    	auto k = pair.second;
        for (size_t l =n+1; l<d; ++l)
            comp[{count+ 2*n+1,l-(n+1)}] =3;// -getV(V,i,n,k,l);//  (val,:Alrstar)
        for (size_t j = n+1; j<d;++j)
            comp[{count+ 2*n+1,d-2*(n+1)+j}] =4;//-getV(V,i,j,k,n);//  (val,:Alr)
        comp[{count+ 2*n+1,getsizeV22(n,d)-1}] =5;//-getV(V,i,n,k,n); // (val,:AtAr)
        count++;
    }

    count = 0;
    for (size_t j = 1; j < n;++j){
        for (size_t i = 0; i < j;++i){
            for (size_t l = n+1; l<d;++l)
                comp[{count+ 2*n+1+n*n,l-(n+1)}] =6;//-getV(V,i,j,n,l); //  (val, :Arm)
            count++;
        }
    }

    count = 0;
    for (size_t l = 1; l<n;++l){
        for (size_t k = 0; k <l;++k){
            for (size_t j = n+1; j < d;++j)
                comp[{count+ 2*n+1+n*n+n*(n-1)/2,d-2*(n+1)+j}] =7;//-getV(V,n,j,k,l); //  (val, :Armstar)
            count++;
        }
    }
    return comp;
}

Tensor V21f(size_t n,Tensor T, Tensor V){
    size_t d = 2*V.dimensions[0];


    size_t n1 = d - n - 1;
    Tensor comp({getsizeV11(n1-1),getsizeV22(n1,d)});

    size_t counti = 0;
    for (size_t i = d-1; i>n; --i){
        size_t countl = 0;
        for (size_t l = n;l>0; --l){
            comp[{1+counti,countl}] = -getV(V,i,n,n,l-1) ;//  (val, :AtAminus)
            countl++;
		}
        counti++;
    }
    size_t countl = 0;
    for (size_t l = d-1; l>n;--l){
        size_t countj = 0;
        for (size_t j = n; j> 0;--j){
            comp[{countl+n1+1,n+countj}] =  getV(V,n,j-1,n,l); // :  (val,:AtAplus)
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
    XERUS_LOG(info,list);

    for (auto pair : list){
		auto i = pair.first;
		auto k = pair.second;
        size_t countl = 0;
        for (size_t l = n;l>0;--l){
            comp[{count+ 2*n1+1,countl}] =  -getV(V,i,n,k,l-1); //  (val,:Arlstar)
            countl++;
        }
        size_t countj = 0;
        for (size_t j = n;j>0;--j){
            comp[{count+ 2*n1+1,n+countj}] =-getV(V,i,j-1,k,n); // (val, :Arl)
            countj++;
        }
        comp[{count+ 2*n1+1,getsizeV22(n1,d)-1}] = -getV(V,i,n,k,n); //  (val,:AtAl)
        count++;
    }

    count = 0;
    for (size_t j = d-2; j>n;--j){
        for (size_t i = d-1;i>j;--i){
            size_t countl = 0;
            for (size_t l = n;l>0;--l){
                comp[{count+ 2*n1+1+n1*n1,countl}] = -getV(V,i,j,n,l-1);//  (val,:Alm)
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
                comp[{count+ 2*n1+1+n1*n1+n1*(n1-1)/2,n+countj}] = -getV(V,n,j-1,k,l);//  (val,:Almstar)
                countj++;
            }
            count++;
        }
    }
	Index i1,j1;
	comp(i1,j1) = comp(j1,i1);
    return comp;
}

Tensor MVf(Tensor T, Tensor V){
    size_t d   = 2*V.dimensions[0];

    size_t n = getsizeV11(d/2-1)+getsizeV22(d/2-1,d);
    Tensor MV({n,n});

    MV[n-1,0] = 1;
    MV[0,n-1] = 1;
    for (size_t i=0; i< d/2;++i){
        MV[{1+i,n-i-2}] = 1;
        MV[{1+d/2+i,n-2-i-d/2}] = 2;//1;
        MV[{n-1-i,1+i}] = 3;//1;
        MV[{n-1-i-d/2,1+d/2+i}] = 4;//1;
    }

//    listl = []
//    for i = 1:K÷2
//        push!(listl,Pair(i,i))
//        for k = 1:i-1
//            push!(listl,Pair(k,i))
//        end
//        for k = 1:i-1
//            push!(listl,Pair(i,k))
//        end
//    end
//    listr = []
//    for i = K:-1:K÷2+1
//        push!(listr,Pair(i,i))
//        for k = K:-1:i+1
//            push!(listr,Pair(k,i))
//        end
//        for  k = K:-1:i+1
//            push!(listr,Pair(i,k))
//        end
//    end
//    countl = 1
//    for (i,k) ∈ listl
//        countr = 1
//        signl = i<=k ? 1.0 : -1.0
//        for (j,l) ∈ listr
//            signr = j<=l ? 1.0 : -1.0
//            val = -getV(V,i,j,k,l)
//            MV[countl+1+K,countr+1+K] = abs(val) < 0.0 ? 0.0 :  val
//            countr+=1
//        end
//        countl+=1
//    end
//
//    countl = 1
//    for j = 2:K÷2
//        for i = 1:j-1
//            countr=1
//            for k = K-1:-1:K÷2+1
//                for l = K:-1:k+1
//                    val =  -getV(V,i,j,k,l)
//                    MV[countl+1+K+(K÷2)^2,countr+1+K+(K÷2)^2+binomial(K÷2,2)] = abs(val) < 0.0 ? 0.0 :  val
//                    countr+=1
//                end
//            end
//            countl+=1
//        end
//    end
//
//    countl = 1
//
//    for j = K-1:-1:K÷2+1
//        for i = K:-1:j+1
//            countr=1
//            for k = 2:K÷2
//                for l = 1:k-1
//                    val = -getV(V,i,j,k,l)
//                    MV[countr+1+K+(K÷2)^2+binomial(K÷2,2),countl+1+K+(K÷2)^2] =  abs(val) < 0.0 ? 0.0 : val
//                    countr+=1
//                end
//            end
//            countl+=1
//        end
//    end



    return MV;
}


size_t getsizeV11(size_t i){
    return 1+2*(i+1)+(i+1)*(i+1)+(i+1)*(i);
}

size_t getsizeV22(size_t i,size_t d){
    return 1+2*(d-i-1);
}
value_t getV(Tensor V,size_t i, size_t j, size_t k, size_t l){
	//value_t val = returnVValue(V,i,j,k,l)+returnVValue(V,j,i,l,k)-returnVValue(V,j,i,k,l)-returnVValue(V,i,j,l,k);
    value_t val = 1000*i+100*j+10*k+l;
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
