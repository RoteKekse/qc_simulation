#include <xerus.h>
#include <classes/helpers.cpp>
#include <classes/loading_tensors.cpp>

using namespace xerus;

class InternalSolver {
	const size_t d;

	std::vector<Tensor> leftAStack;
	std::vector<Tensor> rightAStack;

	std::vector<Tensor> leftBStack;
	std::vector<Tensor> rightBStack;

	TTTensor& x;
	const TTOperator& A;
	const TTTensor& b;
	const double solutionsNorm;
public:
	size_t maxIterations;

	InternalSolver(const TTOperator& _A, TTTensor& _x, const TTTensor& _b)
		: d(_x.order()), x(_x), A(_A), b(_b), solutionsNorm(frob_norm(_b)), maxIterations(1000)
	{
		leftAStack.emplace_back(Tensor::ones({1,1,1}));
		rightAStack.emplace_back(Tensor::ones({1,1,1}));
		leftBStack.emplace_back(Tensor::ones({1,1}));
		rightBStack.emplace_back(Tensor::ones({1,1}));
	}


	void push_left_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);

		Tensor tmpA, tmpB;
		tmpA(i1, i2, i3) = leftAStack.back()(j1, j2, j3)
				*xi(j1, k1, i1)*Ai(j2, k1, k2, i2)*xi(j3, k2, i3);
		leftAStack.emplace_back(std::move(tmpA));
		tmpB(i1, i2) = leftBStack.back()(j1, j2)
				*xi(j1, k1, i1)*bi(j2, k1, i2);
		leftBStack.emplace_back(std::move(tmpB));
	}


	void push_right_stack(const size_t _position) {
		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		const Tensor &xi = x.get_component(_position);
		const Tensor &Ai = A.get_component(_position);
		const Tensor &bi = b.get_component(_position);

		Tensor tmpA, tmpB;
		tmpA(i1, i2, i3) = xi(i1, k1, j1)*Ai(i2, k1, k2, j2)*xi(i3, k2, j3)
				*rightAStack.back()(j1, j2, j3);
		rightAStack.emplace_back(std::move(tmpA));
		tmpB(i1, i2) = xi(i1, k1, j1)*bi(i2, k1, j2)
				*rightBStack.back()(j1, j2);
		rightBStack.emplace_back(std::move(tmpB));
	}

	double calc_residual_norm() {
		Index i,j;
		return frob_norm(A(i/2, j/2)*x(j&0) - b(i&0)) / solutionsNorm;
	}


	void solve() {
		// Build right stack
		x.move_core(0, true);
		for (size_t pos = d-1; pos > 0; --pos) {
			push_right_stack(pos);
		}

		Index i1, i2, i3, j1 , j2, j3, k1, k2;
		std::vector<double> residuals(10, 1000.0);

		for (size_t itr = 0; itr < maxIterations; ++itr) {
			// Calculate residual and check end condition
			residuals.push_back(calc_residual_norm());
			if (residuals.back()/residuals[residuals.size()-10] > 0.99) {
				XERUS_LOG(simpleALS, "Done! Residual decrease from " << std::scientific << residuals[10] << " to " << std::scientific << residuals.back() << " in " << residuals.size()-10 << " iterations.");
				return; // We are done!
			}
			XERUS_LOG(simpleALS, "Iteration: " << itr << " Residual: " << residuals.back());


			// Sweep Left -> Right
			for (size_t corePosition = 0; corePosition < d; ++corePosition) {
				Tensor op, rhs;

				const Tensor &Ai = A.get_component(corePosition);
				const Tensor &bi = b.get_component(corePosition);

				op(i1, i2, i3, j1, j2, j3) = leftAStack.back()(i1, k1, j1)*Ai(k1, i2, j2, k2)*rightAStack.back()(i3, k2, j3);
				rhs(i1, i2, i3) =            leftBStack.back()(i1, k1) *   bi(k1, i2, k2) *   rightBStack.back()(i3, k2);

				xerus::solve(x.component(corePosition), op, rhs);

				if (corePosition+1 < d) {
					x.move_core(corePosition+1, true);
					push_left_stack(corePosition);
					rightAStack.pop_back();
					rightBStack.pop_back();
				}
			}


			// Sweep Right -> Left : only move core and update stacks
			x.move_core(0, true);
			for (size_t corePosition = d-1; corePosition > 0; --corePosition) {
				push_right_stack(corePosition);
				leftAStack.pop_back();
				leftBStack.pop_back();
			}

		}
	}

};

void simpleALS(const TTOperator& _A, TTTensor& _x, const TTTensor& _b)  {
	InternalSolver solver(_A, _x, _b);
	return solver.solve();
}

TTTensor makeTT(TTOperator F,size_t d){
	TTTensor res = TTTensor(std::vector<size_t>(d,2));
	for (size_t i = 0; i < d; ++i){
		Tensor tmp({1,2,1});
		tmp[{0,0,0}] = F.get_component(i)[{0,0,0,0}];
		tmp[{0,1,0}] = F.get_component(i)[{0,1,1,0}];
		res.set_component(i,tmp);
	}
	return res;
}


int main(int argc, char* argv[]) {
	const auto geom = argv[1];
	const auto basisname = argv[2];

	TTOperator D,F1,F2;
	std::string name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_H_diag.ttoperator";
	read_from_disc(name,D );

	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv.ttoperator";
	read_from_disc(name,F1 );

	name = "data/"+static_cast<std::string>(geom)+"_"+static_cast<std::string>(basisname)+"_Finv2.ttoperator";
	read_from_disc(name,F2 );

	F1.round(1);
	F2.round(1);

	size_t d = D.order()/2;
	XERUS_LOG(info, d);
	XERUS_LOG(info,"Norm D " << D.frob_norm());


	TTTensor test,test1,test2,b = TTTensor::ones(std::vector<size_t>(d,2));
	xerus::Index ii,jj;
	auto x1 = makeTT(F1,d);
	x1 *= -1.0;
	auto x2 = makeTT(F2,d);
	test1(ii^d) = D(ii^d,jj^d) * x1(jj^d);
	test1 -=b;
	test2(ii^d) = D(ii^d,jj^d) * x2(jj^d);
	test2 -=b;

	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test1.frob_norm());
	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test2.frob_norm());

	simpleALS(D, x1, b);

	test(ii^d) = D(ii^d,jj^d) * x1(jj^d);
	test -=b;

	XERUS_LOG(info,"Approximation error = " <<std::setprecision(12) <<test.frob_norm());


}

