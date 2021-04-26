#include <xerus.h>
#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <ctime>
using namespace xerus;
using namespace Eigen;

//typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Mat;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage


template<typename M>
M load_csv (const std::string & path);


int main(int argc, char* argv[]) {
	/*
	 * !!!!! Change Here !!!!
	 */
	auto V_AO = xerus::Tensor();
	std::string name = "data/"+std::to_string(argv[1])+"_"+std::to_string(argv[2])+"_VAO_.tensor";
	std::ifstream read(name.c_str());
	misc::stream_reader(read,V_AO,xerus::misc::FileFormat::BINARY);
	read.close();

	size_t nob = V_AO.dimensions[0];
	size_t dim = 2*nob; // 16 electron, 8 electron pairs !!!!! Change Here !!!!
	const auto filename = argv[1];
	const auto basisname  = argv[2];
	std::cout << "-------------------------------------------- Loading Data ----------------------------" << std::endl;
	auto C = xerus::Tensor({nob,nob});
	auto S = xerus::Tensor({nob,nob});
	auto H_AO = xerus::Tensor({nob,nob});




	name = "data/"+std::to_string(argv[1])+"_"+std::to_string(argv[2])+"_TAO_.csv";
	Mat H_Mat = load_csv<Mat>(name);
	name = "data/"+std::to_string(argv[1])+"_"+std::to_string(argv[2])+"_C_.csv";
	Mat C_Mat = load_csv<Mat>(name);
	name = "data/"+std::to_string(argv[1])+"_"+std::to_string(argv[2])+"_SAO_.csv";
	Mat S_Mat = load_csv<Mat>(name);

	std::cout << " C size " << C_Mat.rows() << "x" << C_Mat.cols() << std::endl;
	std::cout << " H size " << H_Mat.rows() << "x" << H_Mat.cols() << std::endl;
	std::cout << " V size " << V_AO.dimensions[0] << "x" << V_AO.dimensions[1] << "x"<< V_AO.dimensions[2] << "x"<< V_AO.dimensions[3] << std::endl;
	for(size_t i = 0; i < nob; ++i){
		for(size_t j = 0; j < nob; ++j){
			C[{i,j}] = C_Mat(i,j);
			S[{i,j}] = S_Mat(i,j);
			H_AO[{i,j}] = H_Mat(i,j);
		}
	}

	std::cout << "-------------------------------------------- Basis Transformation --------------------" << std::endl;

	xerus::Index i1,i2,i3,i4, ii,jj,kk,ll,mm;
	auto V_MO = xerus::Tensor({nob,nob,nob,nob});
	auto H_MO = xerus::Tensor({nob,nob});
	auto S_test = xerus::Tensor({nob,nob});


	V_MO(i1,i2,i3,i4) =  V_AO(ii,jj,kk,ll) * C(ii,i1)*C(jj,i2)*C(kk,i3)*C(ll,i4);

	H_MO(i1,i2) = C(ii,i1)*C(jj,i2)*H_AO(ii,jj);

	S_test(i1,i2) = C(ii,i1)*C(jj,i2)*S(ii,jj);

	std::cout << "V_AO norm " << V_AO.frob_norm() << std::endl;
	std::cout << "V_MO norm " << V_MO.frob_norm() << std::endl;
	std::cout << "H_AO norm " << H_AO.frob_norm() << std::endl;
	std::cout << "H_MO norm " << H_MO.frob_norm() << std::endl;

	std::cout << "S_test norm " << S_test.frob_norm() << std::endl;



	name = "data/"+std::to_string(argv[1])+"_"+std::to_string(argv[2])+"_V_.tensor";
	std::ofstream write(name.c_str());
	misc::stream_writer(write,V_MO,xerus::misc::FileFormat::BINARY);
	write.close();



	name = "data/"+std::to_string(argv[1])+"_"+std::to_string(argv[2])+"_T_.tensor";
	std::ofstream write2(name.c_str());
	misc::stream_writer(write2,H_MO,xerus::misc::FileFormat::BINARY);
	write2.close();



	return 0;
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
