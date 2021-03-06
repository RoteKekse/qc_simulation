#include <xerus.h>

#include "classes/loading_tensors.cpp"
#include "classes/helpers.cpp"

using namespace xerus;
using xerus::misc::operator<<;


int main(){
	Index ii,jj,kk,ll,mm,nn;
	TTOperator H;
	Tensor tmp;
	std::string name = "data/n2_1.05_cc-pvtz_H.ttoperator";
	read_from_disc(name,H);

	auto c1 = H.get_component(60);
	auto c2 = H.get_component(61);
	auto dim1 = c1.dimensions;
	auto dim2 = c2.dimensions;
	auto ones = Tensor::ones({dim1[0]});
	XERUS_LOG(info,c1.dimensions);
	XERUS_LOG(info,c2.dimensions);

	clock_t begin_time = clock();
	tmp(ii,jj,kk) = ones(nn)* c1(nn,ii,jj,kk);
	tmp(ii,jj,ll,mm^2) = tmp(ii,jj,kk)*c2(kk,ll,mm^2);
	XERUS_LOG(info,  (value_t) (clock() - begin_time) / CLOCKS_PER_SEC);

	c1.use_dense_representation();
	c2.use_dense_representation();
	begin_time = clock();
	tmp(ii,jj,kk) = ones(nn)* c1(nn,ii,jj,kk);
	tmp(ii,jj,ll,mm^2) = tmp(ii,jj,kk)*c2(kk,ll,mm^2);
	XERUS_LOG(info,  (value_t) (clock() - begin_time) / CLOCKS_PER_SEC);

	c1 = Tensor::random(dim1);
	c2 	 = Tensor::random(dim2);
	begin_time = clock();
	tmp(ii,jj,kk) = ones(nn)* c1(nn,ii,jj,kk);
	tmp(ii,jj,ll,mm^2) = tmp(ii,jj,kk)*c2(kk,ll,mm^2);
	XERUS_LOG(info,  (value_t) (clock() - begin_time) / CLOCKS_PER_SEC);

	c1 = H.get_component(55);
	c2 = H.get_component(56);
	dim1 = c1.dimensions;
	dim2 = c2.dimensions;
	ones = Tensor::ones({dim1[0]});
	XERUS_LOG(info,c1.dimensions);
	XERUS_LOG(info,c2.dimensions);

	begin_time = clock();
	tmp(ii,jj,kk) = ones(nn)* c1(nn,ii,jj,kk);
	tmp(ii,jj,ll,mm^2) = tmp(ii,jj,kk)*c2(kk,ll,mm^2);
	XERUS_LOG(info,  (value_t) (clock() - begin_time) / CLOCKS_PER_SEC);

	c1.use_dense_representation();
	c2.use_dense_representation();
	begin_time = clock();
	tmp(ii,jj,kk) = ones(nn)* c1(nn,ii,jj,kk);
	tmp(ii,jj,ll,mm^2) = tmp(ii,jj,kk)*c2(kk,ll,mm^2);
	XERUS_LOG(info,  (value_t) (clock() - begin_time) / CLOCKS_PER_SEC);
	begin_time = clock();

	c1 = Tensor::random(dim1);
	c2 = Tensor::random(dim2);
	XERUS_LOG(info,  (value_t) (clock() - begin_time) / CLOCKS_PER_SEC);

	begin_time = clock();
	tmp(ii,jj,kk) = ones(nn)* c1(nn,ii,jj,kk);
	tmp(ii,jj,ll,mm^2) = tmp(ii,jj,kk)*c2(kk,ll,mm^2);
	XERUS_LOG(info,  (value_t) (clock() - begin_time) / CLOCKS_PER_SEC);

	return 0;
}
