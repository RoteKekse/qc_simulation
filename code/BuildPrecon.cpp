#include <xerus.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>

#include <classes/loading_tensors.cpp>



#define build_operator 0

using namespace xerus;
using namespace Eigen;
using xerus::misc::operator<<;

typedef Eigen::Matrix<value_t, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
//              this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, size_t k, value_t shift, std::vector<value_t> shift_vec);
TTOperator build_Fock_op_inv2(std::vector<value_t> coeffs, size_t k,value_t h, value_t shift, std::vector<value_t> shift_vec);
TTOperator build_Fock_op(std::vector<value_t> coeffs);



value_t get_hst(size_t k);
value_t get_tj(int j, size_t k);
value_t get_wj(int j, size_t k);
value_t minimal_ev(std::vector<value_t> coeffs);
value_t maximal_ev(std::vector<value_t> coeffs);

value_t get_gamma(int k, size_t dim);
value_t get_beta(int k);



int main(int argc, char* argv[]) {

	const auto geom = argv[1];
	const auto basisname = argv[2];
	value_t shift = std::atof(argv[3]);
	size_t k = std::atof(argv[4]);
	value_t h =std::atof(argv[5]);
    std::string name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_eps.csv";
	Mat HFev_tmp = load_csv<Mat>(name);

	size_t nob = HFev_tmp.rows();
	//size_t nob = 60;
	XERUS_LOG(info, nob);


	std::vector<value_t> HFev;


	for(size_t j = 0; j < nob; ++j){
		auto val = HFev_tmp(j,0);
		HFev.emplace_back(val);
		HFev.emplace_back(val);
	}

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

	TTOperator Fock_inv = build_Fock_op_inv(HFev, k, shift, shift_vec);
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv.ttoperator";
	//Fock_inv.round(0.0);
	write_to_disc(name,Fock_inv);
	XERUS_LOG(info,Fock_inv.ranks());


	TTOperator Fock_inv2 = build_Fock_op_inv2(HFev, k, h, shift, shift_vec);
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv2.ttoperator";
	//Fock_inv2.round(0.0);
	write_to_disc(name,Fock_inv2);
	XERUS_LOG(info,Fock_inv2.ranks());


	xerus::Index ii,jj,kk,ll,i1,i2,i3,i4,j1,j2,j3,j4,k1,k2,k3,k4;
	TTOperator test, Fock = build_Fock_op(HFev);
	Fock += shift*TTOperator::identity(std::vector<size_t>(4*nob,2));

	test(ii^(2*nob),jj^(2*nob)) = Fock(ii^(2*nob),kk^(2*nob)) * Fock_inv(kk^(2*nob),jj^(2*nob));
	test += TTOperator::identity(std::vector<size_t>(4*nob,2));
	test.move_core(0);
	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm());


	test(ii^(2*nob),jj^(2*nob)) = Fock(ii^(2*nob),kk^(2*nob)) * Fock_inv2(kk^(2*nob),jj^(2*nob));
	test -= TTOperator::identity(std::vector<size_t>(4*nob,2));
	test.move_core(0);
	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm());


	XERUS_LOG(info,"Norm Fock " << Fock.frob_norm());
	XERUS_LOG(info,"Norm Fock inv " << Fock_inv.frob_norm());
	XERUS_LOG(info,"Norm Fock inv2 " << Fock_inv2.frob_norm());


	Fock_inv.round(1);
	Fock_inv2.round(1);

	test(ii^(2*nob),jj^(2*nob)) = Fock(ii^(2*nob),kk^(2*nob)) * Fock_inv(kk^(2*nob),jj^(2*nob));
	test += TTOperator::identity(std::vector<size_t>(4*nob,2));
	test.move_core(0);
	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm());


	test(ii^(2*nob),jj^(2*nob)) = Fock(ii^(2*nob),kk^(2*nob)) * Fock_inv2(kk^(2*nob),jj^(2*nob));
	test -= TTOperator::identity(std::vector<size_t>(4*nob,2));
	test.move_core(0);
	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm());

	return 0;
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




TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, const size_t k, value_t shift, std::vector<value_t> shift_vec){
	xerus::Index ii,jj,kk,ll;
	size_t dim = coeffs.size();
	value_t dim_v = static_cast<value_t>(dim);
	TTOperator result(std::vector<size_t>(2*dim,2));
	int k_int = static_cast<int>(k);
	value_t coeff1;

	XERUS_LOG(info, "minimal = " << minimal_ev(coeffs));
	XERUS_LOG(info, "maximal = " << maximal_ev(coeffs));
	value_t lambda_min = maximal_ev(coeffs) + shift;

	for ( int j = -k_int; j <=k_int; ++j){
		TTOperator tmp(std::vector<size_t>(2*dim,2));
		for (size_t i = 0; i < dim; ++i){
			coeff1 = std::exp(2*get_tj(j,k)/lambda_min*(-coeffs[i]-shift_vec[i]));
			auto aa = xerus::Tensor({1,2,2,1});
			aa[{0,1,1,0}] =  coeff1 ;
			aa[{0,0,0,0}] =  std::exp(2*get_tj(j,k)/lambda_min*(-shift_vec[i]))  ;
			tmp.set_component(i,aa);
		}
		value_t coeff2 = 2*get_wj(j,k)/lambda_min;
		result -= coeff2 * tmp;
		//result.round(0.0);
		//XERUS_LOG(info,"j = " << j << " coeff2 " << coeff2 << " norm " << tmp.frob_norm()<< std::endl << result.ranks());
	}
	return result;
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

TTOperator build_Fock_op_inv2(std::vector<value_t> coeffs, const size_t k,value_t h, value_t shift, std::vector<value_t> shift_vec){
	xerus::Index ii,jj,kk,ll;
	size_t dim = coeffs.size();
	TTOperator result(std::vector<size_t>(2*dim,2));
	int k_int = static_cast<int>(k);
	value_t fac,fac2,fac3,beta,gamma,dim_v = static_cast<value_t>(dim),j_v;

	for ( int j = -k_int; j <=k_int; ++j){
		TTOperator tmp(std::vector<size_t>(2*dim,2));
		bool s =false;
		XERUS_LOG(info, "j = " << j);
		j_v =  static_cast<value_t>(j);
		for (size_t i = 0; i < dim; ++i){
			auto aa = xerus::Tensor({1,2,2,1});
			aa[{0,0,0,0}] = std::exp(j_v/dim_v*h-std::exp(h*j_v)*shift_vec[i]);//shift/dim_v );
			aa[{0,1,1,0}] = std::exp(j_v/dim_v*h-std::exp(h*j_v)*(coeffs[i]+shift_vec[i]));//shift/dim_v ));
			tmp.set_component(i,aa);
			//XERUS_LOG(info, "aa[{0,0,0,0}] = "  << aa[{0,0,0,0}] << " aa[{0,1,1,0}] = "  << aa[{0,1,1,0}]);
		}
		if (tmp.frob_norm() > 1e2)
			s = true;
		if (tmp.frob_norm() < 1e1 && s)
			return result;
		//if (j == -k_int || j == k_int)
		XERUS_LOG(info, "tmp norm = "  << tmp.frob_norm());
		if (s)
			result += h*tmp;
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



