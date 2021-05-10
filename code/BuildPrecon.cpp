#include <xerus.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#include "classes/loading_tensors.cpp"
#include "classes/helpers.cpp"


#define build_operator 0

using namespace xerus;
using namespace Eigen;
using xerus::misc::operator<<;

typedef Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
//              this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, std::vector<value_t> shift_vec, size_t rank);
TTOperator build_Fock_op_inv2(std::vector<value_t> coeffs, size_t k1, size_t k2,value_t h, value_t shift, std::vector<value_t> shift_vec);
TTOperator build_Fock_op(std::vector<value_t> coeffs);

std::pair<std::vector<value_t>,std::vector<value_t>> get_a_b(value_t R,size_t rank);


value_t get_hst(size_t k);
value_t get_tj(int j, size_t k);
value_t get_wj(int j, size_t k);
value_t minimal_ev(std::vector<value_t> coeffs);
value_t maximal_ev(std::vector<value_t> coeffs);

value_t get_gamma(int k, size_t dim);
value_t get_beta(int k);

void printError(TTOperator F, TTOperator Fi, std::vector<size_t> idx, size_t nob);


int main(int argc, char* argv[]) {

	const auto geom = argv[1];
	const auto basisname = argv[2];
	value_t shift = std::atof(argv[3]);
	size_t rank = std::atof(argv[4]);


    std::string name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_eps.csv";
	Mat HFev_tmp = load_csv<Mat>(name);
	size_t nob = HFev_tmp.rows();
	//nob = std::atof(argv[7]);
	XERUS_LOG(info, nob);


	std::vector<value_t> HFev;


	for(size_t j = 0; j < nob; ++j){
		auto val = HFev_tmp(j,0);
		HFev.emplace_back(val);
		HFev.emplace_back(val);
	}
	XERUS_LOG(info,HFev);

	std::vector<value_t> shift_vec(2*nob,0.0);
	value_t sum;
	size_t count = 0;
	for (size_t i = 0 ; i < 2*nob; ++i){
		if (HFev[i] < 0){
			shift_vec[i] = -HFev[i]+1;
			sum = sum - HFev[i]+1;
			count++;
		}
	}
	XERUS_LOG(info,sum);
	value_t rest_shift = (shift - sum)/ static_cast<value_t>(2*nob-count);
	for (size_t i = 0 ; i < 2*nob; ++i){
		if (HFev[i] >= 0){
			shift_vec[i] = rest_shift;
			sum+=rest_shift;
		}
	}
	XERUS_LOG(info,"sum = " << sum);
	XERUS_LOG(info,"rest_shift = " << rest_shift);

//  Tensor T,V;
//	read_from_disc("../data/T_H2O_48_bench.tensor",T);
//	read_from_disc("../data/V_H2O_48_bench.tensor",V);
//	for(size_t j = 0; j < 2*nob; ++j){
//		value_t val = 0;
//		val +=T[{j,j}];
//		for (size_t k : {0,1,2,3,22,23,30,31}){
//			val +=(V[{j,k,j,k}]-V[{j,k,k,j}]);
//		}
//		XERUS_LOG(info,j << " value = " <<val);
//		HFev.emplace_back(val);
//	}

	TTOperator Fock_inv = build_Fock_op_inv(HFev, shift_vec,rank);
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv_r"+std::to_string(rank)+".ttoperator";
	//Fock_inv.round(0.0);
	write_to_disc(name,Fock_inv);
	XERUS_LOG(info,Fock_inv.ranks());



	xerus::Index ii,jj,kk,ll,i1,i2,i3,i4,j1,j2,j3,j4;
	TTOperator test, Fock = build_Fock_op(HFev);
	Fock += shift*TTOperator::identity(std::vector<size_t>(4*nob,2));
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Fock.ttoperator";
	write_to_disc(name,Fock);
	test(ii^(2*nob),jj^(2*nob)) = Fock(ii^(2*nob),kk^(2*nob)) * Fock_inv(kk^(2*nob),jj^(2*nob));
	test -= TTOperator::identity(std::vector<size_t>(4*nob,2));
	test.move_core(0);
	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm());


	Tensor test1,test2;
	auto phi =makeUnitVector({0,1,2,3,4,5,6,7,8,9,10,11,12,13},2*nob);

	XERUS_LOG(info, "Test Inv 1");
	printError(Fock, Fock_inv, {0,1,2,3,4,5,6,7,8,9,10,11,12,13}, nob);
	printError(Fock, Fock_inv, {22,45,33,66,77,88,99,73,42,32,12,21,63,2}, nob);
	printError(Fock, Fock_inv, {0,1,2,3,4,5,6,7,8,9,10,11,12,17}, nob);
	printError(Fock, Fock_inv, {24,25,26,27,28,29,30,31,32,33,34,35,36,37}, nob);
	printError(Fock, Fock_inv, {90,91,92,93,94,95,96,97,98,99,100,102,103,104}, nob);


	return 0;
}

void printError(TTOperator F, TTOperator Fi, std::vector<size_t> idx, size_t nob){
	xerus::Index ii,jj;
	Tensor test1,test2;
	auto phi =makeUnitVector(idx,2*nob);
	test1() =  phi(ii^(2*nob))*F(ii^(2*nob),jj^(2*nob)) * phi(jj^(2*nob));
	test2() =  phi(ii^(2*nob))*Fi(ii^(2*nob),jj^(2*nob)) * phi(jj^(2*nob));
	XERUS_LOG(info,"Fock = " <<test1[0] << " Fock inv= " <<test2[0] << " prod= " <<test1[0]*test2[0]);
}

TTOperator build_Fock_op(std::vector<value_t> coeffs){
	size_t dim = coeffs.size();

	TTOperator result(std::vector<size_t>(2*dim,2));
	size_t comp = 0;
	auto id = xerus::Tensor::identity({2,2});
	id.reinterpret_dimensions({1,2,2,1});
	auto aa = xerus::Tensor({1,2,2,1});
	aa[{0,1,1,0}] = 1.0;
	for (size_t comp = 0; comp < dim; ++comp){
		value_t coeff = coeffs[comp];
		if (comp == 0){
				Tensor tmp = Tensor({1,2,2,2});
				tmp.offset_add(id,{0,0,0,0});
				tmp.offset_add(coeff*aa,{0,0,0,1});
				result.set_component(comp,tmp);
		} else if (comp == dim - 1){
			Tensor tmp = Tensor({2,2,2,1});
			tmp.offset_add(coeff*aa,{0,0,0,0});
			tmp.offset_add(id,{1,0,0,0});
			result.set_component(comp,tmp);
		} else {
			Tensor tmp = Tensor({2,2,2,2});
			tmp.offset_add(id,{0,0,0,0});
			tmp.offset_add(coeff*aa,{0,0,0,1});
			tmp.offset_add(id,{1,0,0,1});
			result.set_component(comp,tmp);
		}
	}
	return result;
}


TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, std::vector<value_t> shift_vec,size_t rank){
	xerus::Index ii,jj,kk,ll;
	value_t a_v1,b_v1,a_v2,b_v2,a_v3,b_v3;
	size_t dim = coeffs.size();
	value_t dim_v = static_cast<value_t>(dim);
	TTOperator result(std::vector<size_t>(2*dim,2)),tmp(std::vector<size_t>(2*dim,2));
	value_t coeff1,coeff2,a=0.0,b=0.0,R;

	for(size_t i = 0; i < dim;++i){
		if (shift_vec[i] < coeffs[i]+shift_vec[i]){
			a += shift_vec[i];
			b +=  coeffs[i]+shift_vec[i];
		}
		else {
			b += shift_vec[i];
			a +=  coeffs[i]+shift_vec[i];
		}
	}
	R = b/a;
	auto ab = get_a_b(R,rank);
	auto a_v = ab.first;
	auto b_v = ab.second;
	a_v1 = 0.326884916411528/a;
	b_v1 = 0.123022177451201/a;

	a_v2 =1.04402770744113/a;
	b_v2 = 0.76173209876179/a;

	a_v3 =2.94374564939135/a;
	b_v3 = 2.57995075168948/a;
	XERUS_LOG(info,"a = " << a <<" b = " << b << " R = "<< R  );
	XERUS_LOG(info,"a = " << a_v1 <<" b = " << b_v1 );
	XERUS_LOG(info,"a = " << a_v2 <<" b = " << b_v2 );
	XERUS_LOG(info,"a = " << a_v3 <<" b = " << b_v3 );
	XERUS_LOG(info,"a = " << a_v[0]/a <<" b = " << b_v[0]/a );
	XERUS_LOG(info,"a = " << a_v[1]/a <<" b = " << b_v[1]/a );
	XERUS_LOG(info,"a = " << a_v[2]/a <<" b = " << b_v[2]/a );

	for (size_t i = 0; i < dim; ++i){
		coeff1 = shift_vec[i];
		coeff2 = coeffs[i]+shift_vec[i];
		auto aa = xerus::Tensor({1,2,2,1});
		aa[{0,0,0,0}] =  std::exp(-b_v1*coeff1)  ;
		aa[{0,1,1,0}] =  std::exp(-b_v1*coeff2) ;
		result.set_component(i,aa);
	}
	result *= a_v1;

	for (size_t i = 0; i < dim; ++i){
		coeff1 = shift_vec[i];
		coeff2 = coeffs[i]+shift_vec[i];
		auto aa = xerus::Tensor({1,2,2,1});
		aa[{0,0,0,0}] =  std::exp(-b_v2*coeff1)  ;
		aa[{0,1,1,0}] =  std::exp(-b_v2*coeff2) ;
		tmp.set_component(i,aa);
	}
	result += a_v2*tmp;

	tmp = TTOperator(std::vector<size_t>(2*dim,2));
	for (size_t i = 0; i < dim; ++i){
		coeff1 = shift_vec[i];
		coeff2 = coeffs[i]+shift_vec[i];
		auto aa = xerus::Tensor({1,2,2,1});
		aa[{0,0,0,0}] =  std::exp(-b_v3*coeff1)  ;
		aa[{0,1,1,0}] =  std::exp(-b_v3*coeff2) ;
		tmp.set_component(i,aa);
	}
	result += a_v3*tmp;
		//result.round(0.0);
		//XERUS_LOG(info,"j = " << j << " coeff2 " << coeff2 << " norm " << tmp.frob_norm()<< std::endl << result.ranks());

	return result;
}

//TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, std::vector<value_t> shift_vec, size_t rank){
//	xerus::Index ii,jj,kk,ll;
//	size_t dim = coeffs.size();
//	value_t dim_v = static_cast<value_t>(dim),av,bv;
//	TTOperator result(std::vector<size_t>(2*dim,2)),tmp(std::vector<size_t>(2*dim,2));
//	value_t coeff1,coeff2,a=0.0,b=0.0,R;
//
//	for(size_t i = 0; i < dim;++i){
//		if (shift_vec[i] < coeffs[i]+shift_vec[i]){
//			a += shift_vec[i];
//			b +=  coeffs[i]+shift_vec[i];
//		}
//		else {
//			b += shift_vec[i];
//			a +=  coeffs[i]+shift_vec[i];
//		}
//	}
//	R = b/a;
//	auto ab = get_a_b(R,rank);
//	auto a_v = ab.first;
//	auto b_v = ab.second;
//
//	XERUS_LOG(info,"a = " << a <<" b = " << b << " R = "<< R);
//	XERUS_LOG(info,"a_v = " << a_v <<" b_v = " << b_v);
//	for (size_t j = 0; j < rank; j++){
//		tmp = TTOperator(std::vector<size_t>(2*dim,2));
//		av = a_v[j]/a;
//		bv = b_v[j]/a;
//		XERUS_LOG(info,av<< " " << bv);
//		for (size_t i = 0; i < dim; ++i){
//			coeff1 = shift_vec[i];
//			coeff2 = coeffs[i]+shift_vec[i];
//			auto aa = xerus::Tensor({1,2,2,1});
//			aa[{0,0,0,0}] =  std::exp(-bv*coeff1)  ;
//			aa[{0,1,1,0}] =  std::exp(-bv*coeff2) ;
//			tmp.set_component(i,aa);
//		}
//		result+= av*tmp;
//	}
//
//	return result;
//}

std::pair<std::vector<value_t>,std::vector<value_t>> get_a_b(value_t R,size_t rank){
	if (rank == 2){
		return std::pair<std::vector<value_t>,std::vector<value_t>>({0.512344165699713,2.29531084041227},{0.183443209989993,1.39888652942634});
	}
	if (rank == 3){
		return std::pair<std::vector<value_t>,std::vector<value_t>>({ 0.326884916411528,1.04402770744113,2.94374564939135},{0.123022177451201,0.76173209876179,2.57995075168948});
	}

	//if (rank == 1){
	if (R < 2e0)
		return std::pair<std::vector<value_t>,std::vector<value_t>>({2.00094589050897},{0.715129187969905});
	if (R < 3e0)
		return std::pair<std::vector<value_t>,std::vector<value_t>>({1.7376357425821},{0.597083366966729});
	if (R < 4e0)
		return std::pair<std::vector<value_t>,std::vector<value_t>>({1.60150307236392},{0.5323920576674});
	if (R < 5e0)
		return std::pair<std::vector<value_t>,std::vector<value_t>>({1.52162963033521},{0.493163066904056});
	if (R < 6e0)
		return std::pair<std::vector<value_t>,std::vector<value_t>>({1.47309216505896},{0.468847474995074});
	if (R < 7e0)
		return std::pair<std::vector<value_t>,std::vector<value_t>>({1.44488691232593},{0.454549612297251});
	if (R < 8e0)
		return std::pair<std::vector<value_t>,std::vector<value_t>>({1.43145353412515},{0.447696316846526});
	if (R < 9e0)
		return std::pair<std::vector<value_t>,std::vector<value_t>>({1.42909978697927},{0.446492606298809});
	return std::pair<std::vector<value_t>,std::vector<value_t>>({1.42909978698058},{0.446492606299478});
	//}
}


value_t get_hst(size_t k){
	return M_PI * M_PI / std::sqrt(static_cast<value_t>(k));
}

value_t get_tj(int j, size_t k){
	value_t hst = get_hst(k);
	return std::log(std::exp(static_cast<value_t>(j)*hst) + std::sqrt(1+std::exp(2*static_cast<value_t>(j)*hst)));
}

value_t get_wj(int j, size_t k){
	value_t hst = get_hst(k);
	return hst/std::sqrt(1+std::exp(-2*static_cast<value_t>(j)*hst));
}

value_t minimal_ev(std::vector<value_t> coeffs){
	value_t lambda;
	for (size_t i = 0; i < coeffs.size(); ++i){
		value_t coeff = coeffs[i];
		lambda += (coeff < 0 ? coeff : 0);
	}
	return lambda;
}

value_t maximal_ev(std::vector<value_t> coeffs){
	value_t lambda;
	for (size_t i = 0; i < coeffs.size(); ++i){
		value_t coeff = coeffs[i];
		lambda += (coeff < 0 ? 0 : coeff);
	}
	return lambda;
}

TTOperator build_Fock_op_inv2(std::vector<value_t> coeffs, const size_t k1,const size_t k2,value_t h, value_t shift, std::vector<value_t> shift_vec){
	xerus::Index ii,jj,kk,ll;
	size_t dim = coeffs.size();
	TTOperator result(std::vector<size_t>(2*dim,2));
	int k_int1 = static_cast<int>(k1);
	int k_int2 = static_cast<int>(k2);
	value_t fac,fac2,fac3,beta,gamma,dim_v = static_cast<value_t>(dim),j_v;
	bool s =false;
	for ( int j = -k_int1; j <=k_int2; ++j){
		TTOperator tmp(std::vector<size_t>(2*dim,2));
		j_v =  static_cast<value_t>(j);
		for (size_t i = 0; i < dim; ++i){
			auto aa = xerus::Tensor({1,2,2,1});
			aa[{0,0,0,0}] = std::exp(j_v/dim_v*h-std::exp(h*j_v)*shift_vec[i]);//shift/dim_v );
			aa[{0,1,1,0}] = std::exp(j_v/dim_v*h-std::exp(h*j_v)*(coeffs[i]+shift_vec[i]));//shift/dim_v ));
			tmp.set_component(i,aa);
			//XERUS_LOG(info, "aa[{0,0,0,0}] = "  << aa[{0,0,0,0}] << " aa[{0,1,1,0}] = "  << aa[{0,1,1,0}]);
		}
		result += h*tmp;
//		XERUS_LOG(info, result.ranks());

	}
	return result;
}


value_t get_gamma(int k,size_t dim ){
	value_t h = 0.5;
	return h*std::exp(static_cast<value_t>(k)*h/static_cast<value_t>(dim));
}
value_t get_beta(int k){
	value_t h = 0.5;
	return std::exp(static_cast<value_t>(k)*h);
}

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<value_t> values;
    uint rows = 0;
    while (std::getline(indata, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        while (std::getline(lineStream, cell, ',')) {
            values.push_back(std::stod(cell));
        }
        ++rows;
    }
    return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}



