	#include <xerus.h>
	#include "classes/GradientMethods/tangentialOperation.cpp"
	#include "classes/GradientMethods/ALSres.cpp"
	#include "classes/GradientMethods/basic.cpp"

	#include "classes/loading_tensors.cpp"
	#include "classes/helpers.cpp"

	using namespace xerus;
	using xerus::misc::operator<<;

	double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx);
	value_t getParticleNumber(const TTTensor& x);
	value_t getParticleNumberUp(const TTTensor& x);
	value_t getParticleNumberDown(const TTTensor& x);

	int main(int argc, char* argv[]) {
		const auto geom = argv[1];
		const auto basisname = argv[2];

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

		size_t nob = H.order()/2;
		size_t num_elec = 14;
		size_t max_iter = 10;
		size_t max_rank = 5;
		Index ii,jj,kk,ll,mm;
		value_t eps = 10e-6;
		value_t alpha_start = 0.1; bool optimal = false;
		std::string out_name = "results/PPGD_CG_" +static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+ "_"+ std::to_string(max_rank) +"_results.csv";

		// Load operators
		XERUS_LOG(info, "--- Loading operators ---");
		XERUS_LOG(info, "Loading inverse of Fock Operator");
		TTOperator id,Finv;
		name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv3.ttoperator";
		read_from_disc(name,Finv);
		XERUS_LOG(info,Finv.ranks());




		XERUS_LOG(info, "--- Initializing Start Vector ---");
		XERUS_LOG(info, "Setting Startvector");
		xerus::TTTensor phi,phi_tmp,phi2;
		name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_phi_5.tttensor";
		read_from_disc(name,phi);
		//project(phi,num_elec,2*nob);
		XERUS_LOG(info,phi.ranks());

		XERUS_LOG(info,"--- starting gradient descent ---");
		value_t rHx,rHr,rx,rr,xx,xHx;
		double alpha,beta,residual;


		TTTensor res, res2, res_last;
		std::vector<value_t> result;
		std::vector<Tensor> res_tangential, res_last_tangential;
		TangentialOperation Top(phi);
		std::ofstream outfile;

		outfile.open(out_name);
		outfile.close();

		xx = phi.frob_norm();
		phi /= xx; //normalize
		xHx = contract_TT(H,phi,phi);
		result.emplace_back(xHx  + nuc[{0}]);
		for (size_t iter = 0; iter < max_iter; ++iter){
			//update phi
			XERUS_LOG(info, "------ Iteration = " << iter);
			XERUS_LOG(info,"Projected Gradient step with PC (non symmetric)");
			XERUS_LOG(info,"Particle Number Up   phi " <<  getParticleNumberUp(phi));
			XERUS_LOG(info,"Particle Number Down phi " <<  getParticleNumberDown(phi));
			XERUS_LOG(info,"Particle Number      phi " << getParticleNumber(phi));

			res_tangential.clear();
			res_tangential = Top.localProduct(H,Finv,xHx,true);

			XERUS_LOG(info,"Particle Number res " <<  getParticleNumber(Top.builtTTTensor(res_tangential)));

			if (iter == 0){
				res = Top.builtTTTensor(res_tangential);
			} else {
				res_last_tangential = Top.localProduct(res,id);
				XERUS_LOG(info,"Particle Number res " << std::setprecision(13) << getParticleNumber(Top.builtTTTensor(res_last_tangential)));
				beta = frob_norm(res_tangential)/frob_norm(res_last_tangential); //Fletcher Reeves update
				XERUS_LOG(info,"Beta = " << beta);
				add(res_tangential,res_last_tangential, beta);
				res = Top.builtTTTensor(res_tangential);
			}

			residual = res.frob_norm();

			XERUS_LOG(info,residual);
			XERUS_LOG(info,"Norm res " << res.frob_norm());



			//XERUS_LOG(info,"Particle Number res (after projection) " << std::setprecision(13) << getParticleNumber(res));
			XERUS_LOG(info,"\n" << res.ranks());
			//Calculate optimal stepsize
			if (optimal){
				rHx = contract_TT(H,res,phi);
				rHr = contract_TT(H,res,res);
				rx = contract_TT(id,res,phi);
				rr = contract_TT(id,res,res);
				alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
			}
			phi = phi - alpha* res;
			phi.round(std::vector<size_t>(2*nob-1,max_rank),eps);
			xx = phi.frob_norm();
			phi /= xx;
			xHx = contract_TT(H,phi,phi);

			Top.update(phi);

			res_last = res;

			result.emplace_back(xHx  + nuc[{0}]);
			XERUS_LOG(info,std::setprecision(8) <<result);
			XERUS_LOG(info, phi.ranks());

			//Write to file
			outfile.open(out_name,std::ios::app);
			outfile.close();

		}
		return 0;
	}

	double get_stepsize(double xHx, double rHr, double rHx, double xFx, double rFr, double rFx){
		double a = rFr*rHx-rHr*rFx;
		double b = rHr*xFx-rFr*xHx;
		double c = rFx*xHx-rHx*xFx;

		double disc = b*b-4*a*c;
		double alpha1 = (-b + std::sqrt(disc))/(2*a);
		double alpha2 = (-b - std::sqrt(disc))/(2*a);
		return alpha1;
	}





	value_t getParticleNumber(const TTTensor& x){
		Index ii,jj;
		size_t d = x.order();
		auto P = particleNumberOperator(d);

		Tensor pn,nn;
		pn() = P(ii/2,jj/2)*x(ii&0)*x(jj&0);
		nn() = x(ii&0)*x(ii&0);
		return pn[0]/nn[0];
	}

	value_t getParticleNumberUp(const TTTensor& x){
			Index ii,jj;
			size_t d = x.order();
			auto P = particleNumberOperatorUp(d);

			Tensor pn,nn;
			pn() = P(ii/2,jj/2)*x(ii&0)*x(jj&0);
			nn() = x(ii&0)*x(ii&0);
			return pn[0]/nn[0];
		}

	value_t getParticleNumberDown(const TTTensor& x){
			Index ii,jj;
			size_t d = x.order();
			auto P = particleNumberOperatorDown(d);

			Tensor pn,nn;
			pn() = P(ii/2,jj/2)*x(ii&0)*x(jj&0);
			nn() = x(ii&0)*x(ii&0);
			return pn[0]/nn[0];
		}




