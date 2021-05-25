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
		size_t finvrank = std::atof(argv[3]);
		value_t shift = std::atof(argv[4]);
		size_t max_rank = std::atof(argv[5]);
		size_t max_iter = std::atof(argv[6]);
		value_t eps  = std::atof(argv[7]);
		size_t  opti  = std::atof(argv[8]);
		value_t alpha_start  = std::atof(argv[9]);

		TTOperator H;
		std::string name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_H.ttoperator";
		//std::string name = "data/H_H2O_48_bench_single.ttoperator";
		//std::string name = "data/hamiltonian_H2O_48_full_benchmark.ttoperator";
		read_from_disc(name,H );
		XERUS_LOG(info, "The ranks of H are " << H.ranks() );

		Tensor nuc;
		name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_nuc.tensor";
		read_from_disc(name,nuc );
		nuc[0] = nuc[0] - shift;
		//nuc[{0}] = 	-52.4190597253;

		size_t nob = H.order()/4;
		Index ii,jj,kk,ll,mm;
		bool optimal = (opti == 0 ? false : true);
		std::string out_name = "results/PPGD_CG_" +static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+ "_r"+ std::to_string(max_rank)+ "_f" +std::to_string(finvrank)+"_i"+std::to_string(max_iter)+"_opt"+std::to_string(opti) +"_results.csv";

		// Load operators
		XERUS_LOG(info, "--- Loading operators ---");
		XERUS_LOG(info, "Loading inverse of Fock Operator");
		TTOperator id=TTOperator::identity(std::vector<size_t>(4*nob,2)),Finv;

		if (finvrank == 0)
			Finv = id;
		else {
			name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv_r"+std::to_string(finvrank)+".ttoperator";
			//name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv.ttoperator";
			read_from_disc(name,Finv);
			XERUS_LOG(info,Finv.ranks());
		}
		XERUS_LOG(info,Finv.ranks());

		XERUS_LOG(info, "--- Initializing Start Vector ---");
		XERUS_LOG(info, "Setting Startvector");
		xerus::TTTensor phi,phi_tmp,phi2;
		name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_phi_ini.tttensor";
		read_from_disc(name,phi);
		XERUS_LOG(info,phi.ranks());

		XERUS_LOG(info,"--- starting gradient descent ---");
		value_t rHx,rHr,rx,rr,xx,xHx,alpha_last;
		double alpha=alpha_start,beta,residual;


		TTTensor res, res2, res_last;
		clock_t begin_time,global_time = clock();
		std::vector<Tensor> res_tangential, res_last_tangential;
		TangentialOperation Top(phi);
		std::ofstream outfile;

		outfile.open(out_name);
		outfile <<  "Iteration ,Eigenvalue, Projected Residual,Calculation Time,Step Size" <<  std::endl;
		outfile.close();

		xx = phi.frob_norm();
		phi /= xx; //normalize
		xHx = contract_TT(H,phi,phi);
		for (size_t iter = 0; iter < max_iter; ++iter){
			//update phi
			XERUS_LOG(info, "------ Iteration = " << iter);
			XERUS_LOG(info,"Projected Gradient step with PC (non symmetric)");

			res_tangential.clear();
			XERUS_LOG(info,"Project Gradient");

			res_tangential = Top.localProduct(H,Finv,xHx,true);

			XERUS_LOG(info,"Calc update direction");

			if (iter == 0){
				res = Top.builtTTTensor(res_tangential);
			} else {
				XERUS_LOG(info,"Project last update");
				res_last_tangential = Top.localProduct(res,id);
				beta = frob_norm(res_tangential)/frob_norm(res_last_tangential); //Fletcher Reeves update
				XERUS_LOG(info,"Beta = " << beta);
				add(res_tangential,res_last_tangential, beta);
				XERUS_LOG(info,"Assemble update");

				res = Top.builtTTTensor(res_tangential);
			}

			residual = res.frob_norm();
			XERUS_LOG(info,"Norm res " << residual);

			XERUS_LOG(info, "Max rank Res = " << res.ranks()[nob]);
			//Calculate optimal stepsize
			//res /= residual;
			if (optimal || iter < 2){
				rHx = contract_TT(H,res,phi);
				rHr = contract_TT(H,res,res);
				rx = contract_TT(id,res,phi);
				rr = contract_TT(id,res,res);
				alpha = get_stepsize(xHx,rHr,rHx,xx,rr,rx);
			}
			auto xHx_tmp = xHx;
			size_t count = 0;
			while(xHx_tmp>= xHx){
				XERUS_LOG(info,"count = " << count  << " " << xHx_tmp<< " " << xHx );

				if (count >=1){
					alpha_start *= 0.8;
					alpha = alpha_start;
				}
				XERUS_LOG(info,"alpha = " << alpha);

				if (alpha < 1e-7)
					return 0;
				phi_tmp = phi - alpha* res;
				alpha_last = alpha;
				alpha = alpha_start;
				if (optimal)
					break;

				phi_tmp.round(std::vector<size_t>(2*nob-1,max_rank),eps);

				xx = phi_tmp.frob_norm();
				phi_tmp /= xx;
				XERUS_LOG(info,"Calculate new eigenvalue");
				xHx_tmp = contract_TT3(H,phi_tmp);
				count++;

			}
			phi = phi_tmp;
			xHx = xHx_tmp;

			XERUS_LOG(info,"Particle Number Phi " <<  getParticleNumber(phi));

			Top.update(phi);

			res_last = res;

			XERUS_LOG(info,std::setprecision(8) <<xHx  + nuc[{0}]-shift);
			XERUS_LOG(info, "Max rank Phi = " << phi.ranks()[nob]);

			//Write to file
			outfile.open(out_name,std::ios::app);
			outfile <<  iter << "," << std::setprecision(12) <<  xHx+nuc[0]-shift<<","<<residual << ","<< (value_t) (clock() - global_time) / CLOCKS_PER_SEC << "," << alpha_last << std::endl;
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




