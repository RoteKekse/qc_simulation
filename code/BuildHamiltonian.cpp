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
Tensor V12f(size_t n.size_t d);
value_t getV(Tensor V,size_t i, size_t j, size_t k, size_t l);



int main(int argc, char* argv[]) {
	auto test1 = V12f(1,8);

	XERUS_LOG(info,test1.dimensions << "\n"  << test1);




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
	XERUS_LOG(info,comp.dimensions);
	comp[{0,0,1,getsizeV22(i,d)-1}] = 1;
	comp[{d-i,1,0,getsizeV22(i,d)-1}] = 1;
	comp[{getsizeV22(i-1,d)-1,0,0,getsizeV22(i,d)-1}] = 1;comp[{getsizeV22(i-1,d)-1,1,1,getsizeV22(i,d)-1}] = 1;
	for(size_t j = 0; j < d-i-1;++j){
		comp[{j+1,0,0,j}] = 1; comp[{j+1,1,1,j}] = -1;
		comp[{d-i+1+j,0,0,d-i+j-1}] = 1; comp[{d-i+1+j,1,1,d-i+j-1}] = -1;
	}
	if (reverse){
		Index i1,j1;
		comp(i1,j1) = comp(j1,i1);
		return comp;
	}
	return comp;
}

//Tensor V12f(size_t n, Tensor T, Tensor V){
Tensor V12f(size_t n, size_t d){
    //size_t d = 2*V.dimensions[0]
	XERUS_REQUIRE(n>=1,"n=0 doesent work");
    Tensor comp(getsizeV11(n-1),getsizeV22(n,d))

    for (size_t i = 1; i <= n;++i){
        for (size_t l = n+2; l <= d; ++l)
            comp[{1+i-1,l-(n+1)-1}] =1;// -getV(V,i,n,n,l); //(val,:AtAplus)
    }
//	for (size_t l = 1; l <=n;++l)
//        for (size_t j = n+2; j<=d;++j)
//            comp[{l+n,K-(n+1)-1+j-(n+1)-1}] = 2;//getV(V,n,j,n,l); //  (val,:AtAminus)

//    size_t count = 1;
//    std::vector<Pair<size_t,size_t>> list;
//    for (size_t i = 1; i <= n-1;++i){
//        list.emplace_back(Pair(i,i));
//        for (size_t k 1; k <= i-1;++k)
//        	list.emplace_back(Pair(k,i));
//        for (size_t k 1; k <= i-1;++k)
//        	list.emplace_back(Pair(i,k));
//    }
//    for (auto pair : list){
//    	auto i = pair.first;
//    	auto k = pair.second;
//        for (size_t l =n+1; l<=d; ++l)
//            comp[{count+ 2*n-1,l-n}] =3;// -getV(V,i,n,k,l);//  (val,:Alrstar)
//        for (size_t j = n+1; j<=d;++j)
//            comp[{count+ 2*n-1,K-n+j-n}] =4;//-getV(V,i,j,k,n);//  (val,:Alr)
//        comp[{count+ 2*n-1,getsizeV22(n,d)}] =5;//-getV(V,i,n,k,n); // (val,:AtAr)
//        count+=1
//    }
//
//    count = 1
//    for (size_t j = 1; j < n;++j){
//        for (size_t i = 0; i < j;++i){
//            for (size_t l = n+1; l<d;++l)
//                comp[{count+ 2*n-1+(n-1)*(n-1),l-n}] =6;//-getV(V,i,j,n,l); //  (val, :Arm)
//            count+=1
//        }
//    }
//
//    count = 1
//    for (size_t l = 2; l<=n-1;++l){
//        for (size_t k = 1; k <=l-1;++k){
//            for (size_t j = n+1; j <= d;++j)
//                comp[{count+ 2*n-1+(n-1)*(n-1)+(n-1)*(n-2)/2,K-n+j-n}] =7;//-getV(V,n,j,k,l); //  (val, :Armstar)
//            count+=1
//        }
//    }
    return comp;
}


size_t getsizeV11(size_t i){
    return 1+2*(i+1)+(i+1)*(i+1)+(i+1)*(i);
}

size_t getsizeV22(size_t i,size_t d){
    return 1+2*(d-i-1);
}
value_t getV(Tensor V,size_t i, size_t j, size_t k, size_t l){
	value_t val = returnVValue(V,i,j,k,l)+returnVValue(V,j,i,l,k)-returnVValue(V,j,i,k,l)-returnVValue(V,i,j,l,k);
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
