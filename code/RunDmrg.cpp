#include <xerus.h>

#include <vector>
#include <fstream>
#include <ctime>
#include <queue>
#include <math.h>



#include "classes/loading_tensors.cpp"
#include "classes/helpers.cpp"


using namespace xerus;
using xerus::misc::operator<<;


class InternalSolver2;
double simpleMALS(const TTOperator& _A, TTTensor& _x, value_t shift, double _eps, size_t _maxRank, size_t _nosw, value_t _nuc, std::	string _out_file);
/*
 * Main!!
 */
int main(int argc, char* argv[]) {
	const auto geom = argv[1];
	const auto basisname = argv[2];
	value_t shift = std::atof(argv[3]);
	size_t max_rank = std::atof(argv[4]);
	size_t number_of_sweeps = std::atof(argv[5]);
	value_t eps  = std::atof(argv[6]);
	size_t ini  = std::atof(argv[7]);
	bool initial = (ini == 0 ? false : true);

	std::string out_name = "results/DMRG_" +static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+ "_r"+ std::to_string(max_rank)+ "_f" +"_i"+std::to_string(number_of_sweeps) +"_results.csv";

	TTOperator H;
	std::string name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_H.ttoperator";
	//std::string name = "data/H_H2O_48_bench_single.ttoperator";
	//std::string name = "data/hamiltonian_H2O_48_full_benchmark.ttoperator";
	read_from_disc(name,H );
	XERUS_LOG(info, "The ranks of H are " << H.ranks() );

	Tensor nuc;
	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_nuc.tensor";
	read_from_disc(name,nuc );
	//nuc[{0}] = 	-52.4190597253;

	XERUS_LOG(info, "nuc " << nuc );
	Index ii,jj;
	//Set Parameters
	size_t d = H.order()/2;

//	//Load Intial Value
//	std::vector<size_t> hf = {0,1,2,3,4,5,6,7,8,9,10,11,12,13};
//	//std::vector<size_t> hf = {0,1,2,3,22,23,30,31};
//	TTTensor phi = makeUnitVector(hf,  d);
//	Tensor E;
//	E() = H(ii/2,jj/2)*phi(ii&0)*phi(jj&0);
//	XERUS_LOG(info,"Initial Energy " << E[0]+nuc[0]-shift);
	TTTensor phi;
	if (!initial){
		name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_phi_ini.tttensor";
		read_from_disc(name,phi);
	} else {
		auto noise = TTTensor::random(std::vector<size_t>(d,2),std::vector<size_t>(d-1,1));
		phi = noise/noise.frob_norm();
	}
	//Calculate initial energy



	//Perfrom ALS/DMRG
	double lambda = simpleMALS(H, phi,shift, eps, max_rank,number_of_sweeps, nuc[0]-shift,out_name);
	phi.round(10e-14);


	XERUS_LOG(info, "The ranks of phi are " << phi.ranks() );
	XERUS_LOG(info, "Size of Solution " << phi.datasize() );

	XERUS_LOG(info, "Final Energy =  " << std::setprecision(16) << lambda 	+nuc[0]-shift);

	if (initial){
		name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_phi_ini.tttensor";
		write_to_disc(name,phi);
	} else {
		name = "data/" +static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+ "_r"+ std::to_string(max_rank)+ "_f" +"_i"+std::to_string(number_of_sweeps) +"_phi.tttensor";
		write_to_disc(name,phi);
	}



	return 0;
}

/*
 *
 *
 *
 * Functions
 *
 *
 */


class InternalSolver2 {
	const size_t d;
	double lambda;
	double eps;
	size_t maxRank;
	value_t shift;
	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;
	value_t nuc;
	TTTensor& x;
	const TTOperator& A;
	TTOperator P;
	std::string out_name;
public:
	size_t maxIterations;

	InternalSolver2(const TTOperator& _A, TTTensor& _x, value_t _shift, double _eps, size_t _maxRank, size_t _nosw, value_t _nuc,std::string _out_name)
		: d(_x.order()), x(_x), A(_A), shift(_shift), maxIterations(_nosw), lambda(1.0), eps(_eps), maxRank(_maxRank),nuc(_nuc),out_name(_out_name)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
		P = particleNumberOperator(d);
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);


		Tensor tmpA;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);

		Tensor tmpA;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
	}

	double calc_residual_norm() { // TODO improve this by (A-lamdaI)
		Index ii,jj,kk,ll,mm,nn,oo,i1,i2,i3,i4;
		auto ones = Tensor::ones({1,1,1});
		xerus::Tensor tmp = ones;
		XERUS_LOG(info,"lambda = " << lambda - 28.1930439210);

		for (size_t i = 0; i < d; i++){
			auto Ai = A.get_component(i);
			auto xi = x.get_component(i);
			tmp(i1,i2,i3) = tmp(ii,jj,kk) * xi(ii,ll,i1) * Ai(jj,ll,mm,i2) * xi(kk,mm,i3);
		}
		tmp() = tmp(ii,jj,kk) * ones(ii,jj,kk);
		XERUS_LOG(info,"xAx = " << tmp);

		auto ones2 = Tensor::ones({1,1,1,1});
		xerus::Tensor tmp2 = ones2;
		for (size_t i = 0; i < d; i++){
			auto Ai = A.get_component(i);
			auto xi = x.get_component(i);
			//XERUS_LOG(info,i);
			//XERUS_LOG(info,tmp2.dimensions);
			tmp2(i1,i2,i3,i4) = tmp2(ii,jj,kk,ll) * xi(ii,mm,i1) * Ai(jj,mm,nn,i2) * Ai(kk,nn,oo,i3) * xi(ll,oo,i4);
		}
		tmp2() = tmp2(ii,jj,kk,ll) * ones2(ii,jj,kk,ll);

	//	XERUS_LOG(info,"xAAx = " << tmp2);

		xerus::TTTensor tmp3;
//		tmp3(ii&0) = A(ii/2,jj/2) * x(jj&0);
//		XERUS_LOG(info,tmp3.ranks());
		return std::sqrt(std::abs(tmp2[0]-lambda*lambda));
	}


	double solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 1; --pos) {
			push_right_stack(pos);
		}

	  std::ofstream outfile;
	  outfile.open(out_name);
	  outfile <<  "Iteration ,Eigenvalue, Calculation Time" <<  std::endl;
	  outfile.close();
	  clock_t begin_time,global_time = clock();
	  value_t stack_time = 0,solving_time = 0;

		Index i1, i2, i3, i4, j1 , j2, j3, j4, k1, k2, k3;
		Index a1, a2, a3, a4, a5, r1, r2;
		std::vector<double> residuals_ev(10, 1000.0);
		std::vector<double> residuals(10, 1000.0);
		XERUS_LOG(info,"A = " << A.ranks());
		std::vector<value_t> result;

		for (size_t itr = 0; itr < maxIterations; ++itr) {
//			if (itr % 5 == 0 and itr != 0)
//				maxRank = 250 > maxRank + 10 ? maxRank + 10 : 250;
			stack_time = 0,solving_time = 0;


			// Calculate residual and check end condition
			residuals_ev.push_back(lambda);
			//residuals.push_back(calc_residual_norm());
			if (itr > 1 and std::abs(residuals_ev[residuals_ev.size()-10] - residuals_ev.back()) < eps) {
				XERUS_LOG(info, residuals_ev[residuals_ev.size()-10]);
				XERUS_LOG(info, residuals_ev.back());
				XERUS_LOG(info, eps);
				return lambda; // We are done!
			}

			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d-1; ++corePosition) {
				Tensor  rhs,pn;
				TensorNetwork op;
				//Tensor op;

				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &Ai1 = A.get_component(corePosition+1);

				Tensor &xi = x.component(corePosition);
				Tensor &xi1 = x.component(corePosition+1);


				auto x_rank = xi.dimensions[2];



				//XERUS_LOG(info, "Operator Size = (" << (leftAStack.back()).dimensions[0] << "x" << Ai.dimensions[1] << "x" << Ai1.dimensions[1] << "x" << rightAStack.back().dimensions[0] << ")x("<< leftAStack.back().dimensions[2] << "x" << Ai.dimensions[2] << "x" << Ai1.dimensions[2] << "x" << rightAStack.back().dimensions[2] <<")");


				Tensor sol, xright,xleft;
				sol(a1,a2,a4,a5) = xi(a1,a2,a3)*xi1(a3,a4,a5);

				//lambda = xerus::get_smallest_eigenvalue(sol, op);
				begin_time = clock();
				op(i1, i2, i3, i4, j1, j2, j3, j4) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2) * Ai1(k2,i3,j3,k3)*rightAStack.back()(i4, k3, j4);
				stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;

				begin_time = clock();
				lambda = xerus::get_eigenpair_iterative(sol,op, true,false, 100000, eps);
				//XERUS_LOG(info,corePosition << " " <<lambda << "\n" << op);

				auto xnew = split1(sol,maxRank,1e-6);
				solving_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;


				xright = xnew.second;
				xleft  = xnew.first;



				x.set_component(corePosition, xleft);
				x.set_component(corePosition+1, xright);

				x/=x.frob_norm();


				begin_time = clock();
				if (corePosition+2 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
				}
				stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
				//XERUS_LOG(info, "After move " << x.ranks());

			}
			// Sweep Right -> Left : only move core and update stacks
			begin_time = clock();
			x.move_core(0, true);

			for (size_t corePosition = d-1; corePosition > 1; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
			}
			stack_time += (value_t) (clock() - begin_time) / CLOCKS_PER_SEC;
			XERUS_LOG(simpleMALS, "Iteration: " << itr  << " Eigenvalue " << lambda <<" " << std::setprecision(16) <<  lambda+nuc);
			XERUS_LOG(simpleMALS, "Ranks: " << x.ranks());

			outfile.open(out_name,std::ios::app);
			outfile <<  itr << "," << std::setprecision(12) <<  lambda+nuc-shift<<","<< (value_t) (clock() - global_time) / CLOCKS_PER_SEC <<  std::endl;
			outfile.close();


		}
		return lambda;
	}

	std::pair<Tensor,Tensor> split1(Tensor& sol, size_t maxRank, value_t eps){
		Tensor  U, S, Vt,xright;
		Index i1,i2,r1,r2,j1,j2;
		(U(i1,i2,r1), S(r1,r2), Vt(r2,j1,j2)) = SVD(sol(i1,i2,j1,j2),maxRank,eps);
		xright(r1,j1,j2) = S(r1,r2)*Vt(r2,j1,j2);
		return std::pair<Tensor,Tensor>(U,xright);
	}

	std::pair<Tensor,Tensor> split2(const TensorNetwork op, Tensor& sol, size_t maxRank, value_t eps){
		Tensor  U, S, Vt,xright;
		Index i1,i2,i3,i4,r1,r2,j1,j2,j3,j4;
		(U(i1,i2,r1), S(r1,r2), Vt(r2,j1,j2)) = SVD(sol(i1,i2,j1,j2),maxRank,eps);
		xright(r1,j1,j2) = S(r1,r2)*Vt(r2,j1,j2);

		std::pair<Tensor,Tensor> x = std::pair<Tensor,Tensor>(U,xright);
		Tensor oploc,rhs;
		for (size_t i = 0; i < 10; ++i){
			oploc(i1,i2,r1,j1,j2,r2)  = op(i1,i2,i3,i4,j1,j2,j3,j4) * x.second(r2,j3,j4) * x.second(r1,i3,i4);

			rhs(i1,i2,r1) = op(i1,i2,i3,i4,j1,j2,j3,j4) * sol(j1,j2,j3,j4) * x.second(r1,i3,i4);

			xerus::solve(x.first,oploc,rhs,0);

			oploc(r1,i3,i4,r2,j3,j4)  = op(i1,i2,i3,i4,j1,j2,j3,j4) * x.first(j1,j2,r2) * x.first(i1,i2,r1);
			rhs(r1,i3,i4) = op(i1,i2,i3,i4,j1,j2,j3,j4) * sol(j1,j2,j3,j4) * x.first(i1,i2,r1);

			xerus::solve(x.second,oploc,rhs,0);


		}


		return x;
	}

};

double simpleMALS(const TTOperator& _A, TTTensor& _x, value_t _shift, double _eps, size_t _maxRank, size_t _nosw, value_t _nuc,std::string _out_name)  {
	InternalSolver2 solver(_A, _x, _shift,_eps, _maxRank,_nosw, _nuc,_out_name);
	return solver.solve();
}




