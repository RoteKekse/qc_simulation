#include <xerus.h>
#include <classes/hamiltonian.cpp>

#include <queue>

#include <boost/regex.hpp>
#include <boost/algorithm/string_regex.hpp>


#define debug 0

using namespace xerus;
using xerus::misc::operator<<;

xerus::TTOperator return_annil(size_t i, size_t d);
xerus::TTOperator return_create(size_t i, size_t d);
xerus::TTOperator return_one_e_ac(size_t i, size_t j, size_t d);
xerus::TTOperator return_two_e_ac(size_t i, size_t j, size_t k, size_t l, size_t d);
xerus::TTOperator return_two_e_ac_partial(size_t i, size_t j, size_t k, size_t d);
xerus::TTOperator return_two_e_ac_full(size_t a, size_t b, size_t c, size_t d, size_t dim);



class BuildingOperatorL2R{
	public:
		size_t d;
		Tensor V;
		Tensor T;
		Tensor N;
		std::vector<std::vector<TTOperator>> P;
		std::vector<std::vector<TTOperator>> Q;
		std::vector<TTOperator> S;
		TTOperator H;

		/*
		 * Constructor
		 */
		BuildingOperatorL2R(size_t _d,Tensor _T, Tensor _V)
		: d(_d), T(_T), V(_V){
			XERUS_LOG(BuildOpL2R, "---- Initializing Building Operator L2R ----");
			XERUS_LOG(BuildOpL2R, "- Loading 1e and 2e Integrals and Nuclear Potential");
			XERUS_LOG(BuildOpL2R, "- Initializing Storage Operators H,S,Q,P");
			ini_4_site_operator_H();
			ini_4_site_operator_S();
			ini_4_site_operator_Q();
			ini_4_site_operator_P();
		}

		/*
		 * Build Operator
		 */
		void build(){
			XERUS_LOG(BuildOpL2R, "---- Building Operator ----");
			for (size_t k = 1; k < d; ++k){
				step(k);
				XERUS_LOG(BuildOpL2R, "- Ranks of Operator " << H.ranks());
			}
		}

		/*
		 * One step from k to k + 1
		 */
		void step(const size_t k){
			XERUS_LOG(BuildOpL2R, "- Step " << k);
			update_4_site_operator_H(k+1);

			for (size_t s = k; s < d; ++s)
				update_3_site_operator_S(s,k+1);

			for (size_t r = k; r < d; ++r){
				for (size_t s = k; s < d; ++s){
					update_2_site_operator_Q(r,s,k+1);
					update_2_site_operator_P(r,s,k+1);
				}
			}


		}

		/*
		 * initializes operator H (k = 0)
		 */
		void ini_4_site_operator_H(){
			H = TTOperator({2,2});
			auto tmp = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			tmp[{0,1,1,0}] = T[{0,0}];
			H.set_component(0,tmp);
		}

		/*
		 * initializes operator S (k = 0)
		 */
		void ini_4_site_operator_S(){
			for(size_t s = 0; s < d; ++s){
				S.emplace_back(TTOperator({2,2}));
				auto tmp = xerus::Tensor({1,2,2,1});
				tmp[{0,1,0,0}] = T[{0,s}];
				S[s].set_component(0,tmp);
			}
		}

		/*
		 * initializes operator S (k = 0)
		 */
		void ini_4_site_operator_Q(){
			for (size_t r = 0; r < d; ++r){
				std::vector<TTOperator> tmp2;
				Q.emplace_back(tmp2);
				for(size_t s = 0; s < d; ++s){
					Q[r].emplace_back(TTOperator({2,2}));
					auto tmp = xerus::Tensor({1,2,2,1});
					tmp[{0,1,1,0}]  = V[{0,r,0,s}] - V[{r,0,0,s}];
					Q[r][s].set_component(0,tmp);
				}
			}
		}

		/*
		 * initializes operator S (k = 0)
		 */
		void ini_4_site_operator_P(){
			for (size_t r = 0; r < d; ++r){
				std::vector<TTOperator> tmp2;
				P.emplace_back(tmp2);
				for(size_t s = 0; s < d; ++s){
					P[r].emplace_back(TTOperator({2,2}));
				}
			}
		}

		/*
		 * Builds the hamiltonian from left to right, uses S and Q
		 * overwrites H from the last site
		 */
		void update_4_site_operator_H(const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;
			XERUS_LOG(info,"k = " << k);

			//first summand
			TTOperator H_tmp = add_identity(H);

			//second summand only k
			TTOperator H_tmp2 = TTOperator(std::vector<size_t>(2*dim,2));
			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			for (size_t i = 0; i < k; ++i)
				H_tmp2.set_component(i,id);
			value_t coeff = T[{k,k}];
			auto aa = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			aa[{0,1,1,0}] = coeff;
			H_tmp2.set_component(k,aa);

			//third summand
			TTOperator H_tmpS,H_tmpS_t,annil;
			H_tmpS = add_identity(S[k]);
			annil = return_annil(k,dim);
			H_tmpS(ii/2,jj/2) =  H_tmpS(ii/2,kk/2)*annil(kk/2,jj/2);
			H_tmpS_t(ii/2,jj/2) = H_tmpS(jj/2,ii/2); // transpose

			//fourth summand
			TTOperator H_tmpQ;
			H_tmpQ = add_identity(Q[k][k]);
			auto particle = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			particle[{0,1,1,0}] = 1.0;
			H_tmpQ.set_component(k,particle);


			H = H_tmp + H_tmp2 + H_tmpS + H_tmpS_t + H_tmpQ;
			H.round(0.0);
		}

		/*
		 * Builds the sum over three indices, sum_pqr w_prqs a_p^* a_q^* a_r and includes 1e parts
		 * overwrites S from the last site
		 */
		void update_3_site_operator_S(const size_t s, const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;

			//first summand, til k - 1
			TTOperator S_tmp = add_identity(S[s]);

			//second summand only k, i.e. only 1e part
			TTOperator S_tmp2 = T[{k,s}]*return_create(k, dim);

			//third summand
			TTOperator S_tmpP,annil;
			S_tmpP = add_identity(P[k][s]);
			annil = return_annil(k,dim);
			S_tmpP(ii/2,jj/2) =  S_tmpP(ii/2,kk/2)*annil(kk/2,jj/2);

			//fourth summand
			TTOperator S_tmpQ,create;
			S_tmpQ = add_identity(Q[k][s]);
			create = return_create(k,dim);
			S_tmpQ(ii/2,jj/2) = create(ii/2,kk/2) *  S_tmpQ(kk/2,jj/2);

			//fifth summand
			std::queue<value_t> coeffs;
			for (size_t p = 0; p < k; ++p){
				auto coeff = V[{p,k,k,s}] - V[{p,k,s,k}];
				coeffs.push(coeff);
			}
			TTOperator tmp = build_1_site_operator(coeffs, dim, true);
			auto aa = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			aa[{0,1,1,0}] = 1.0;
			tmp.set_component(k,aa);
			tmp.require_correct_format();



			//final sum
			S[s] = S_tmp + S_tmp2 - S_tmpP + S_tmpQ - tmp;
			S[s].round(0.0);

		}



		/*
		 * Builds the sum over two indices, sum_pq w_prqs a_q^* a_p
		 * overwrites Q from the last site
		 */
		void update_2_site_operator_Q(const size_t r, const size_t s, const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;

			//first summand, til k - 1
			TTOperator Q_tmp = add_identity(Q[r][s]);

			//second summand only k
			TTOperator Q_tmp2 = TTOperator(std::vector<size_t>(2*dim,2));
			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			for (size_t i = 0; i < k; ++i)
				Q_tmp2.set_component(i,id);
			value_t coeff = V[{k,r,k,s}] - V[{r,k,k,s}];
			auto aa = xerus::Tensor({1,2,2,1}); //as 2x2 Matrix:  {{0,0},{0,1}}
			aa[{0,1,1,0}] = coeff;
			Q_tmp2.set_component(k,aa);

			//third summand interaction terms
			TTOperator tmp,tmp_t, create,annil;
			std::queue<value_t> coeffs,coeffs2;
			for (size_t p = 0; p < k; ++p){
				coeff = V[{p,r,k,s}] - V[{r,p,k,s}];
				coeffs.push(coeff);
				coeff = V[{k,r,p,s}] - V[{r,k,p,s}];
				coeffs2.push(coeff);
			}
			tmp = build_1_site_operator(coeffs, dim,true);
			annil = return_annil(k,dim);
			tmp(ii/2,jj/2) = tmp(ii/2,kk/2) * annil(kk/2,jj/2);

			//fourth summand interaction terms
			tmp_t = build_1_site_operator(coeffs2, dim);
			create = return_create(k,dim);
			tmp_t(ii/2,jj/2) = create(ii/2,kk/2) * tmp_t(kk/2,jj/2);

			//final sum
			Q[r][s] = Q_tmp +  Q_tmp2 + tmp + tmp_t;
			Q[r][s].round(0.0);
		}


		/*
		 * Builds the sum over two indices, sum_pq w_prqs a_q a_p
		 * overwrites P from the last site
		 */
		void update_2_site_operator_P(const size_t r, const size_t s, const size_t dim){
			Index ii,jj,kk;
			size_t k = dim - 1;

			//first summand, til k - 1, Note there are no summands for only since they are 0
			TTOperator P_tmp = add_identity(P[r][s]);

			//interaction terms
			std::queue<value_t> coeffs;
			for (size_t p = 0; p < k; ++p){
				value_t coeff = V[{p,k,r,s}] - V[{p,k,s,r}];
				coeffs.push(coeff);
			}
			TTOperator tmp = build_1_site_operator(coeffs, dim, true);
			TTOperator create = return_create(k,dim);
			tmp(ii/2,jj/2) =  tmp(ii/2,kk/2) * create(kk/2,jj/2);

			//final sum
			P[r][s] = P_tmp + tmp;
			P[r][s].round(0.0);
		}


		/*
		 * Builds the sum over one index, sum_p w_prqs a_p, as a rank 2 operator
		 */
		TTOperator build_1_site_operator(std::queue<value_t> coeffs, const size_t dim, const bool transpose = false){
			XERUS_REQUIRE(coeffs.size() < dim,"Number of coefficients is larger than dimension of result");
			TTOperator result(std::vector<size_t>(2*dim,2));
			size_t comp = 0;
			auto id = xerus::Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			auto s = xerus::Tensor::identity({2,2});
			s.reinterpret_dimensions({1,2,2,1});
			s[{0,1,1,0}] = -1.0;
			auto a = xerus::Tensor({1,2,2,1});
			if (transpose)
				a[{0,1,0,0}] = 1.0;
			else
				a[{0,0,1,0}] = 1.0;
			while (!coeffs.empty()){ // set the rank 2 blocks
				value_t coeff = coeffs.front();
				coeffs.pop();
				if (comp == 0){
					if (coeffs.empty()){
						Tensor tmp = Tensor({1,2,2,1});
						tmp.offset_add(coeff*a,{0,0,0,0});
						result.set_component(comp,tmp);
					} else {
						Tensor tmp = Tensor({1,2,2,2});
						tmp.offset_add(s,{0,0,0,0});
						tmp.offset_add(coeff*a,{0,0,0,1});
						result.set_component(comp,tmp);
					}
				} else if (coeffs.empty()){
					Tensor tmp = Tensor({2,2,2,1});
					tmp.offset_add(coeff*a,{0,0,0,0});
					tmp.offset_add(id,{1,0,0,0});
					result.set_component(comp,tmp);
				} else {
					Tensor tmp = Tensor({2,2,2,2});
					tmp.offset_add(s,{0,0,0,0});
					tmp.offset_add(coeff*a,{0,0,0,1});
					tmp.offset_add(id,{1,0,0,1});
					result.set_component(comp,tmp);
				}
				++comp;
			}
			while(comp < dim){ // the rest are identities
				result.set_component(comp,id);
				++comp;
			}
			return result;
		}

		// Takes the given operator and adds an identity to the right or left
		TTOperator add_identity(TTOperator A, bool left=false){
			TTOperator tmpA(A.order() + 2);
			for (size_t i = (left ? 1 : 0); i < A.order() / 2 + (left ? 1 : 0); ++i){
				tmpA.set_component(i,A.get_component(left ? i - 1 : i)); //TODO check if component works instead of get_component
			}
			Tensor id = Tensor::identity({2,2});
			id.reinterpret_dimensions({1,2,2,1});
			tmpA.set_component(left ? 0 : A.order() / 2,id);
			return tmpA;
		}

		/*
		 * Annihilation Operator with operator at position i
		 */
		xerus::TTOperator return_annil(size_t i, size_t d){ // TODO write tests for this
			xerus::Index i1,i2,jj, kk, ll;
			auto a_op = xerus::TTOperator(std::vector<size_t>(2*d,2));
			auto id = xerus::Tensor({2,2});
			id[{0,0}] = 1.0;
			id[{1,1}] = 1.0;
			auto s = xerus::Tensor({2,2});
			s[{0,0}] = 1.0;
			s[{1,1}] = -1.0;
			auto annhil = xerus::Tensor({2,2});
			annhil[{0,1}] = 1.0;
			for (size_t m = 0; m < d; ++m){
				auto tmp = m < i ? s : (m == i ? annhil : id );
				auto res = xerus::Tensor({1,2,2,1});
				res[{0,0,0,0}] = tmp[{0,0}];
				res[{0,1,1,0}] = tmp[{1,1}];
				res[{0,1,0,0}] = tmp[{1,0}];
				res[{0,0,1,0}] = tmp[{0,1}];
				a_op.set_component(m, res);
			}
			return a_op;
		}

		/*
		 * Creation Operator with operator at position i
		 */
		xerus::TTOperator return_create(size_t i, size_t d){ // TODO write tests for this
			xerus::Index i1,i2,jj, kk, ll;
			auto c_op = xerus::TTOperator(std::vector<size_t>(2*d,2));

			auto id = xerus::Tensor({2,2});
			id[{0,0}] = 1.0;
			id[{1,1}] = 1.0;
			auto s = xerus::Tensor({2,2});
			s[{0,0}] = 1.0;
			s[{1,1}] = -1.0;
			auto create = xerus::Tensor({2,2});
			create[{1,0}] = 1.0;
			for (size_t m = 0; m < d; ++m){
				auto tmp = m < i ? s : (m == i ? create : id );
				auto res = xerus::Tensor({1,2,2,1});
				res[{0,0,0,0}] = tmp[{0,0}];
				res[{0,1,1,0}] = tmp[{1,1}];
				res[{0,1,0,0}] = tmp[{1,0}];
				res[{0,0,1,0}] = tmp[{0,1}];
				c_op.set_component(m, res);
			}
			return c_op;
		}

};


int main(){
	XERUS_LOG(info, "---- Start building operator left to right! ----");

	size_t d = 6;
	size_t nob = d/2;
	Tensor T1 = Tensor::random({d/2,d/2});
	Tensor T2({d,d});
	for (size_t i = 0; i < nob; i++){
		for (size_t j = 0; j < nob; j++){
			auto val = T1[{i,j}];
			T1[{j,i}] = val;
			T1[{i,j}] = val;
			T2[{2*i,2*j}] = val;
			T2[{2*j,2*i}] = val;
			T2[{2*i+1,2*j+1}] = val;
			T2[{2*j+1,2*i+1}] = val;
		}
	}
	Tensor Vres = Tensor::random({nob,nob,nob,nob});
	Tensor V1({nob,nob,nob,nob});
	Tensor V2({d,d,d,d});
	for (size_t i = 0; i < nob; i++){
		for (size_t j = 0; j <= i; j++){
			for (size_t k = 0; k<= i; k++){
				for (size_t l = 0; l <= (i==k ? j : k); l++){
					auto value = Vres[{i,j,k,l}];
					V1[{i,k,j,l}] = value;
					V1[{j,k,i,l}] = value;
					V1[{i,l,j,k}] = value;
					V1[{j,l,i,k}] = value;
					V1[{k,i,l,j}] = value;
					V1[{l,i,k,j}] = value;
					V1[{k,j,l,i}] = value;
					V1[{l,j,k,i}] = value;
				}
			}
		}
	}
	for (size_t i = 0; i < nob; i++){
		for (size_t j = 0; j < nob; j++){
			for (size_t k = 0; k < nob; k++){
				for (size_t l = 0; l < nob; l++){
					auto value = V1[{i,j,k,l}];
					V2[{2*i,2*j,2*k,2*l}] = value;
					V2[{2*i+1,2*j,2*k+1,2*l}] = value;
					V2[{2*i,2*j+1,2*k,2*l+1}] = value;
					V2[{2*i+1,2*j+1,2*k+1,2*l+1}] = value;
				}
			}
		}
	}

	Index i1,i2,i3,i4;
	V1(i1,i2,i3,i4) = V1(i1,i3,i2,i4);



	BuildingOperatorL2R builder(d,T2,V2);




	XERUS_LOG(test,"Build 1");
	builder.build();

	auto H1 = builder.H;
	auto H2 = BuildHamil(T1,V1,0.0);

	XERUS_LOG(info,H1.frob_norm());
	XERUS_LOG(info,H2.frob_norm());
	XERUS_LOG(info,(H1-H2).frob_norm());

	auto H1T = Tensor(H1);
	auto H2T = Tensor(H2);
	H1T.reinterpret_dimensions({64,64});
	H2T.reinterpret_dimensions({64,64});

	for (size_t i =0; i < 64; ++i){
		for (size_t j =0; j < 64; ++j){

			if (std::abs(H1T[{i,j}]-H2T[{i,j}])>1e-10){
				std::vector<size_t> idx(2*d,0);
				size_t i_tmp = i, j_tmp = j;
				for (size_t k = d; k> 0; --k){
					idx[k-1] = i_tmp % 2;
					idx[k-1+d] = j_tmp % 2;
					i_tmp /=2;
					j_tmp/=2;
				}
				XERUS_LOG(info,i << " " << j << " " << H1T[{i,j}] <<  " " << H2T[{i,j}]<< "\n" << idx);
				XERUS_LOG(info,H1[idx] <<  " " << H2[idx]);
			}
		}

	}

//	size_t count = 0;
//	size_t count2 = 0;
//	for (size_t i = 0; i < d; i=i+1){
//		for (size_t j = 0; j <= i; j=j+1){
//			for (size_t k = 0; k < d; k=k+1){
//				for (size_t l = 0; l <= k; l=l+1){
//					count++;
//					std::vector<size_t> idx(2*d,0);
//					idx[i] = 1;
//					idx[j] = 1;
//					idx[d+k] = 1;
//					idx[d+l] = 1;
//					value_t val1 = H1[idx];
//					value_t val2 = H2[idx];
//					if (std::abs(val1-val2) > 1e-9){
//						XERUS_LOG(info,i << " " << j<< " " << k<< " " << l );
//						value_t val = 0.5*(V2[{i,j,k,l}]+V2[{j,i,l,k}]-V2[{j,i,k,l}]-V2[{i,j,l,k}]);
//						XERUS_LOG(info,val1 << " " << val2 <<" "  << std::abs(val1-val2)<< " " << getV(V1,i, j, k, l)<< " " << val   <<  "\n" );
//						count2++;
//					}
//				}
//			}
//		}
//	}
//	XERUS_LOG(info,count);
//	XERUS_LOG(info,count2);


	return 0;
}




