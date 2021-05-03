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

typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
//              this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);

TTOperator build_Fock_op_inv(std::vector<value_t>coeffs, size_t k, double shift);



value_t get_hst(size_t k);
value_t get_tj(int j, size_t k);
value_t get_wj(int j, size_t k);
value_t minimal_ev(std::vector<value_t> coeffs);
value_t maximal_ev(std::vector<value_t> coeffs);


int main(int argc, char* argv[]) {

	const auto geom = argv[1];
	const auto basisname = argv[2];
	value_t shift = std::atof(argv[3]);

    std::string name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_eps.csv";
	Mat HFev_tmp = load_csv<Mat>(name);

	size_t nob = HFev_tmp.rows();
	XERUS_LOG(info, nob);
	size_t k = 700;


	std::vector<value_t> HFev;
	for(size_t j = 0; j < nob; ++j){
		auto val = HFev_tmp(j,0);
		HFev.emplace_back(val);
		HFev.emplace_back(val);
	}
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

	TTOperator Fock_inv = build_Fock_op_inv(HFev, k, shift);
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv.ttoperator";
	Fock_inv.round(0.0);
	write_to_disc(name,Fock_inv);

	XERUS_LOG(info,Fock_inv.ranks());


	return 0;
}




TTOperator build_Fock_op_inv(std::vector<value_t> coeffs, const size_t k, value_t shift){
	xerus::Index ii,jj,kk,ll;
	size_t dim = coeffs.size();
	TTOperator result(std::vector<size_t>(2*dim,2));
	int k_int = static_cast<int>(k);
	value_t coeff1;

	XERUS_LOG(info, "minimal = " << minimal_ev(coeffs));
	XERUS_LOG(info, "maximal = " << maximal_ev(coeffs));
	value_t lambda_min = maximal_ev(coeffs) + shift;

	for ( int j = -k_int; j <=k_int; ++j){
		TTOperator tmp(std::vector<size_t>(2*dim,2));
		for (size_t i = 0; i < dim; ++i){
			coeff1 = std::exp(2*get_tj(j,k)/lambda_min*(-coeffs[i]-shift/dim));
			auto aa = xerus::Tensor({1,2,2,1});
			aa[{0,1,1,0}] =  coeff1 ;
			aa[{0,0,0,0}] =  std::exp(2*get_tj(j,k)/lambda_min*(-shift/dim))  ;
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

template<typename M>
M load_csv (const std::string & path) {
    std::ifstream indata;
    indata.open(path);
    std::string line;
    std::vector<double> values;
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



