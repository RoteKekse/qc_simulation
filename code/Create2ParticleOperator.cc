/*
 * Preparing Xerus TT operator for TT optimization, derived from a given closed shell Molecule (e.g. H2O)
 */



// standard C++ headers
#include <atomic>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <iterator>
#include <memory>
#include <mutex>
#include <sstream>
#include <thread>
#include <unordered_map>
#include <vector>

// Libint Gaussian integrals library
#include <libint2/diis.h>
#include <libint2/util/intpart_iter.h>
#include <libint2/chemistry/sto3g_atomic_density.h>
#include <libint2/lcao/molden.h>
#include <libint2.hpp>

// Eigen matrix algebra library
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>

// xerus
#include <xerus.h>

// used data types
using libint2::Shell;
using libint2::Atom;
using libint2::BasisSet;
using libint2::Operator;
using std::cout;
using std::cerr;
using std::endl;

using shellpair_list_t = std::unordered_map<size_t, std::vector<size_t>>;
shellpair_list_t obs_shellpair_list;  // shellpair list for OBS
using shellpair_data_t = std::vector<std::vector<std::shared_ptr<libint2::ShellPair>>>;  // in same order as shellpair_list_t
shellpair_data_t obs_shellpair_data;  // shellpair data for OBS
const auto max_engine_precision = std::numeric_limits<double>::epsilon() / 1e10;

//typedefs
typedef Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>
    Matrix;  // import dense, dynamically sized Matrix type from Eigen;
             // this is a matrix with row-major storage

//forward declaration
std::vector<Atom> read_geometry(const std::string& filename);
Matrix compute_shellblock_norm(const BasisSet& obs, const Matrix& A);

/// computes non-negligible shell pair list; shells \c i and \c j form a
/// non-negligible
/// pair if they share a center or the Frobenius norm of their overlap is
/// greater than threshold
std::tuple<shellpair_list_t,shellpair_data_t>
compute_shellpairs(const BasisSet& bs1,
                   const BasisSet& bs2 = BasisSet(),
                   double threshold = 1e-12);

xerus::Tensor compute_2body_fock(
    const BasisSet& obs,
    double precision = std::numeric_limits<
        double>::epsilon(),  // discard contributions smaller than this
    const Matrix& Schwarz = Matrix()  // K_ij = sqrt(||(ij|ij)||_\infty); if
                                       // empty, do not Schwarz screen
    );
template <libint2::Operator Kernel = libint2::Operator::coulomb>
Matrix compute_schwarz_ints(
    const BasisSet& bs1, const BasisSet& bs2 = BasisSet(),
    bool use_2norm = false,  // use infty norm by default
    typename libint2::operator_traits<Kernel>::oper_params_type params =
        libint2::operator_traits<Kernel>::default_params());
/*
 * Parallel stuff
 */
namespace libint2 {
int nthreads;

/// fires off \c nthreads instances of lambda in parallel
template <typename Lambda>
void parallel_do(Lambda& lambda) {
#ifdef _OPENMP
#pragma omp parallel
  {
    auto thread_id = omp_get_thread_num();
    lambda(thread_id);
  }
#else  // use C++11 threads
  std::vector<std::thread> threads;
  for (int thread_id = 0; thread_id != libint2::nthreads; ++thread_id) {
    if (thread_id != nthreads - 1)
      threads.push_back(std::thread(lambda, thread_id));
    else
      lambda(thread_id);
  }  // threads_id
  for (int thread_id = 0; thread_id < nthreads - 1; ++thread_id)
    threads[thread_id].join();
#endif
}
}


/*
 * Main routine
 */
int main(int argc, char* argv[]) {

	const auto filename = argv[1]+argv[2];
	// Set basis functions
	const auto basisname = argv[3];

  // Import Geometry
  std::vector<Atom> atoms = read_geometry(filename);

  // set up thread pool
  {
    using libint2::nthreads;
    auto nthreads_cstr = getenv("LIBINT_NUM_THREADS");
    nthreads = 1;
    if (nthreads_cstr && strcmp(nthreads_cstr, "")) {
      std::istringstream iss(nthreads_cstr);
      iss >> nthreads;
      if (nthreads > 1 << 16 || nthreads <= 0) nthreads = 1;
    }
  }

  // count the number of electrons
	auto nelectron = 0;
	for (auto i = 0; i < atoms.size(); ++i) nelectron += atoms[i].atomic_number;
	const auto ndocc = nelectron / 2; // number of molecular basis functions
	cout << "# of electrons = " << nelectron << endl;

  // compute the nuclear repulsion energy
  auto enuc = 0.0;
  for (auto i = 0; i < atoms.size(); i++)
    for (auto j = i + 1; j < atoms.size(); j++) {
      auto xij = atoms[i].x - atoms[j].x;
      auto yij = atoms[i].y - atoms[j].y;
      auto zij = atoms[i].z - atoms[j].z;
      auto r2 = xij * xij + yij * yij + zij * zij;
      auto r = sqrt(r2);
      enuc += atoms[i].atomic_number * atoms[j].atomic_number / r;
    }
  cout << "Nuclear repulsion energy = " << std::setprecision(15) << enuc << endl;

  libint2::Shell::do_enforce_unit_normalization(false);

  cout << "Atomic Cartesian coordinates (a.u.):" << endl;
  for (const auto& a : atoms)
    std::cout << a.atomic_number << " " << a.x << " " << a.y << " " << a.z
              << std::endl;

  BasisSet obs(basisname, atoms);
  cout << "orbital basis set rank = " << obs.nbf() << endl;


  /*** =========================== ***/
  /*** compute 2-e integrals       ***/
  /*** =========================== ***/

  // initializes the Libint integrals library ... now ready to compute
  libint2::initialize();

  // compute OBS non-negligible shell-pair list
  {
    std::tie(obs_shellpair_list, obs_shellpair_data) = compute_shellpairs(obs);
    size_t nsp = 0;
    for (auto& sp : obs_shellpair_list) {
      nsp += sp.second.size();
    }
    std::cout << "# of {all,non-negligible} shell-pairs = {"
              << obs.size() * (obs.size() + 1) / 2 << "," << nsp << "}"
              << std::endl;
  }

  auto K = compute_schwarz_ints<>(obs);

  const auto F = compute_2body_fock(obs);

  std::string name = "Data/" + static_cast<std::string>(basisname) +std::to_string(obs.nbf())+ ".tensor";
	std::ofstream write(name.c_str() );
	xerus::misc::stream_writer(write,F,xerus::misc::FileFormat::BINARY);
	write.close();


  libint2::finalize();  // done with libint

  return 0;
}




std::vector<Atom> read_geometry(const std::string& filename) {
  std::cout << "Will read geometry from " << filename << std::endl;
  std::ifstream is(filename);
  if (not is.good()) {
    char errmsg[256] = "Could not open file ";
    strncpy(errmsg + 20, filename.c_str(), 235);
    errmsg[255] = '\0';
    throw std::runtime_error(errmsg);
  }

  // to prepare for MPI parallelization, we will read the entire file into a
  // string that can be
  // broadcast to everyone, then converted to an std::istringstream object that
  // can be used just like std::ifstream
  std::ostringstream oss;
  oss << is.rdbuf();
  // use ss.str() to get the entire contents of the file as an std::string
  // broadcast
  // then make an std::istringstream in each process
  std::istringstream iss(oss.str());

  // check the extension: if .xyz, assume the standard XYZ format, otherwise
  // throw an exception
  if (filename.rfind(".xyz") != std::string::npos)
    return libint2::read_dotxyz(iss);
  else
    throw "only .xyz files are accepted";
}


std::tuple<shellpair_list_t,shellpair_data_t>
compute_shellpairs(const BasisSet& bs1,
                   const BasisSet& _bs2,
                   const double threshold) {
  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  using libint2::nthreads;
  std::cout << "threads = " << nthreads << endl;

  // construct the 2-electron repulsion integrals engine
  using libint2::Engine;
  std::vector<Engine> engines;
  engines.reserve(nthreads);
  engines.emplace_back(Operator::overlap,
                       std::max(bs1.max_nprim(), bs2.max_nprim()),
                       std::max(bs1.max_l(), bs2.max_l()), 0);
  for (size_t i = 1; i != nthreads; ++i) {
    engines.push_back(engines[0]);
  }

  std::cout << "computing non-negligible shell-pair list ... ";

  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

  shellpair_list_t splist;

  std::mutex mx;

  auto compute = [&](int thread_id) {

    auto& engine = engines[thread_id];
    const auto& buf = engine.results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      mx.lock();
      if (splist.find(s1) == splist.end())
        splist.insert(std::make_pair(s1, std::vector<size_t>()));
      mx.unlock();

      auto n1 = bs1[s1].size();  // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
        if (s12 % nthreads != thread_id) continue;

        auto on_same_center = (bs1[s1].O == bs2[s2].O);
        bool significant = on_same_center;
        if (not on_same_center) {
          auto n2 = bs2[s2].size();
          engines[thread_id].compute(bs1[s1], bs2[s2]);
          Eigen::Map<const Matrix> buf_mat(buf[0], n1, n2);
          auto norm = buf_mat.norm();
          significant = (norm >= threshold);
        }

        if (significant) {
          mx.lock();
          splist[s1].emplace_back(s2);
          mx.unlock();
        }
      }
    }
  };  // end of compute

  libint2::parallel_do(compute);

  // resort shell list in increasing order, i.e. splist[s][s1] < splist[s][s2] if s1 < s2
  // N.B. only parallelized over 1 shell index
  auto sort = [&](int thread_id) {
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      if (s1 % nthreads == thread_id) {
        auto& list = splist[s1];
        std::sort(list.begin(), list.end());
      }
    }
  };  // end of sort

  libint2::parallel_do(sort);

  // compute shellpair data assuming that we are computing to default_epsilon
  // N.B. only parallelized over 1 shell index
  const auto ln_max_engine_precision = std::log(max_engine_precision);
  shellpair_data_t spdata(splist.size());
  auto make_spdata = [&](int thread_id) {
    for (auto s1 = 0l; s1 != nsh1; ++s1) {
      if (s1 % nthreads == thread_id) {
        for(const auto& s2 : splist[s1]) {
          spdata[s1].emplace_back(std::make_shared<libint2::ShellPair>(bs1[s1],bs2[s2],ln_max_engine_precision));
        }
      }
    }
  };  // end of make_spdata

  libint2::parallel_do(make_spdata);

  timer.stop(0);
  std::cout << "done (" << timer.read(0) << " s)" << std::endl;

  return std::make_tuple(splist,spdata);
}



xerus::Tensor compute_2body_fock(const BasisSet& obs,
                          double precision, const Matrix& Schwarz) {
  const auto n = obs.nbf();
  const auto nobf =  static_cast<size_t>(n);
  const auto nshells = obs.size();
  xerus::Tensor G =  xerus::Tensor({nobf,nobf,nobf,nobf});

  const auto do_schwarz_screen = Schwarz.cols() != 0 && Schwarz.rows() != 0;
//  Matrix D_shblk_norm =
//      compute_shellblock_norm(obs, D);  // matrix of infty-norms of shell blocks
//
  auto fock_precision = precision;
  // engine precision controls primitive truncation, assume worst-case scenario
  // (all primitive combinations add up constructively)
//  auto max_nprim = obs.max_nprim();
//  auto max_nprim4 = max_nprim * max_nprim * max_nprim * max_nprim;
//  auto engine_precision = std::min(fock_precision / D_shblk_norm.maxCoeff(),
//                                   std::numeric_limits<double>::epsilon()) /
//                          max_nprim4;
//  assert(engine_precision > max_engine_precision &&
//      "using precomputed shell pair data limits the max engine precision"
//  " ... make max_engine_precision smalle and recompile");
//
  // construct the 2-electron repulsion integrals engine pool
  using libint2::Engine;
  Engine engine(Operator::coulomb, obs.max_nprim(), obs.max_l(), 0);
  engine.set_precision(fock_precision);  // shellset-dependent precision
                                               // control will likely break
                                               // positive definiteness
                                               // stick with this simple recipe
  std::cout << "compute_2body_fock:precision = " << precision << std::endl;
  std::cout << "Engine::precision = " << engine.precision() << std::endl;

  std::atomic<size_t> num_ints_computed{0};

  auto shell2bf = obs.shell2bf();

  const auto& buf = engine.results();


  std::cout << "Number of shells " << nshells << std::endl;

  // loop over permutationally-unique set of shells
  for (auto s1 = 0l, s1234 = 0l; s1 != nshells; ++s1) {
    auto bf1_first = shell2bf[s1];  // first basis function in this shell
    auto n1 = obs[s1].size();       // number of basis functions in this shell

    auto sp12_iter = obs_shellpair_data.at(s1).begin();

    for (const auto& s2 : obs_shellpair_list[s1]) {
      auto bf2_first = shell2bf[s2];
      auto n2 = obs[s2].size();

      const auto* sp12 = sp12_iter->get();
      ++sp12_iter;

      const auto Dnorm12 = 0.;

      for (auto s3 = 0; s3 <= s1; ++s3) {
        auto bf3_first = shell2bf[s3];
        auto n3 = obs[s3].size();

        const auto Dnorm123 = 0.;

        auto sp34_iter = obs_shellpair_data.at(s3).begin();

        const auto s4_max = (s1 == s3) ? s2 : s3;
        for (const auto& s4 : obs_shellpair_list[s3]) {
          if (s4 > s4_max)
            break;  // for each s3, s4 are stored in monotonically increasing
                    // order

          // must update the iter even if going to skip s4
          const auto* sp34 = sp34_iter->get();
          ++sp34_iter;


          const auto Dnorm1234 = 0.;
          auto bf4_first = shell2bf[s4];
          auto n4 = obs[s4].size();

          num_ints_computed += n1 * n2 * n3 * n4;

          // compute the permutational degeneracy (i.e. # of equivalents) of
          // the given shell set
          auto s12_deg = (s1 == s2) ? 1.0 : 2.0;
          auto s34_deg = (s3 == s4) ? 1.0 : 2.0;
          auto s12_34_deg = (s1 == s3) ? (s2 == s4 ? 1.0 : 2.0) : 2.0;
          auto s1234_deg = s12_deg * s34_deg * s12_34_deg;

          engine.compute2<Operator::coulomb, libint2::BraKet::xx_xx, 0>(
                obs[s1], obs[s2], obs[s3], obs[s4], sp12, sp34);
          const auto* buf_1234 = buf[0];
          if (buf_1234 == nullptr)
            continue; // if all integrals screened out, skip to next quartet

          // 1) each shell set of integrals contributes up to 6 shell sets of
          // the Fock matrix:
          //    F(a,b) += (ab|cd) * D(c,d)
          //    F(c,d) += (ab|cd) * D(a,b)
          //    F(b,d) -= 1/4 * (ab|cd) * D(a,c)
          //    F(b,c) -= 1/4 * (ab|cd) * D(a,d)
          //    F(a,c) -= 1/4 * (ab|cd) * D(b,d)
          //    F(a,d) -= 1/4 * (ab|cd) * D(b,c)
          // 2) each permutationally-unique integral (shell set) must be
          // scaled by its degeneracy,
          //    i.e. the number of the integrals/sets equivalent to it
          // 3) the end result must be symmetrized
          for (auto f1 = 0, f1234 = 0; f1 != n1; ++f1) {
            const auto bf1 = f1 + bf1_first;
            for (auto f2 = 0; f2 != n2; ++f2) {
              const auto bf2 = f2 + bf2_first;
              for (auto f3 = 0; f3 != n3; ++f3) {
                const auto bf3 = f3 + bf3_first;
                for (auto f4 = 0; f4 != n4; ++f4, ++f1234) {
                  const auto bf4 = f4 + bf4_first;

                  const auto value = buf_1234[f1234];

                  //const auto value_scal_by_deg = value; // * s1234_deg;
//
              		G[{bf1,bf3,bf2,bf4}] = value;
              		G[{bf2,bf3,bf1,bf4}] = value;
              		G[{bf1,bf4,bf2,bf3}] = value;
              		G[{bf2,bf4,bf1,bf3}] = value;
              		G[{bf3,bf1,bf4,bf2}] = value;
              		G[{bf4,bf1,bf3,bf2}] = value;
              		G[{bf3,bf2,bf4,bf1}] = value;
              		G[{bf4,bf2,bf3,bf1}] = value;

//                    g(bf1, bf2) += D(bf3, bf4) * value_scal_by_deg;
//                    g(bf3, bf4) += D(bf1, bf2) * value_scal_by_deg;
//                    g(bf1, bf3) -= 0.25 * D(bf2, bf4) * value_scal_by_deg;
//                    g(bf2, bf4) -= 0.25 * D(bf1, bf3) * value_scal_by_deg;
//                    g(bf1, bf4) -= 0.25 * D(bf2, bf3) * value_scal_by_deg;
//                    g(bf2, bf3) -= 0.25 * D(bf1, bf4) * value_scal_by_deg;
                }
              }
            }
          }
        }
      }
    }
  }


//
//  Matrix GG = 0.5 * (G[0] + G[0].transpose());
//
  std::cout << "# of integrals = " << num_ints_computed << std::endl;

  // symmetrize the result and return
  return G;
}

Matrix compute_shellblock_norm(const BasisSet& obs, const Matrix& A) {
  const auto nsh = obs.size();
  Matrix Ash(nsh, nsh);

  auto shell2bf = obs.shell2bf();
  for (size_t s1 = 0; s1 != nsh; ++s1) {
    const auto& s1_first = shell2bf[s1];
    const auto& s1_size = obs[s1].size();
    for (size_t s2 = 0; s2 != nsh; ++s2) {
      const auto& s2_first = shell2bf[s2];
      const auto& s2_size = obs[s2].size();

      Ash(s1, s2) = A.block(s1_first, s2_first, s1_size, s2_size)
                        .lpNorm<Eigen::Infinity>();
    }
  }

  return Ash;
}

template <libint2::Operator Kernel>
Matrix compute_schwarz_ints(
    const BasisSet& bs1, const BasisSet& _bs2, bool use_2norm,
    typename libint2::operator_traits<Kernel>::oper_params_type params) {
  const BasisSet& bs2 = (_bs2.empty() ? bs1 : _bs2);
  const auto nsh1 = bs1.size();
  const auto nsh2 = bs2.size();
  const auto bs1_equiv_bs2 = (&bs1 == &bs2);

  Matrix K = Matrix::Zero(nsh1, nsh2);

  // construct the 2-electron repulsion integrals engine
  using libint2::Engine;
  using libint2::nthreads;
  std::vector<Engine> engines(nthreads);

  // !!! very important: cannot screen primitives in Schwarz computation !!!
  auto epsilon = 0.;
  engines[0] = Engine(Kernel, std::max(bs1.max_nprim(), bs2.max_nprim()),
                      std::max(bs1.max_l(), bs2.max_l()), 0, epsilon, params);
  for (size_t i = 1; i != nthreads; ++i) {
    engines[i] = engines[0];
  }

  std::cout << "computing Schwarz bound prerequisites (kernel=" << (int)Kernel
            << ") ... ";

  libint2::Timers<1> timer;
  timer.set_now_overhead(25);
  timer.start(0);

  auto compute = [&](int thread_id) {

    const auto& buf = engines[thread_id].results();

    // loop over permutationally-unique set of shells
    for (auto s1 = 0l, s12 = 0l; s1 != nsh1; ++s1) {
      auto n1 = bs1[s1].size();  // number of basis functions in this shell

      auto s2_max = bs1_equiv_bs2 ? s1 : nsh2 - 1;
      for (auto s2 = 0; s2 <= s2_max; ++s2, ++s12) {
        if (s12 % nthreads != thread_id) continue;

        auto n2 = bs2[s2].size();
        auto n12 = n1 * n2;

        engines[thread_id].compute2<Kernel, libint2::BraKet::xx_xx, 0>(bs1[s1], bs2[s2],
                                                              bs1[s1], bs2[s2]);
        assert(buf[0] != nullptr &&
               "to compute Schwarz ints turn off primitive screening");

        // to apply Schwarz inequality to individual integrals must use the diagonal elements
        // to apply it to sets of functions (e.g. shells) use the whole shell-set of ints here
        Eigen::Map<const Matrix> buf_mat(buf[0], n12, n12);
        auto norm2 = use_2norm ? buf_mat.norm()
                               : buf_mat.lpNorm<Eigen::Infinity>();
        K(s1, s2) = std::sqrt(norm2);
        if (bs1_equiv_bs2) K(s2, s1) = K(s1, s2);
      }
    }
  };  // thread lambda

  libint2::parallel_do(compute);

  timer.stop(0);
  std::cout << "done (" << timer.read(0) << " s)" << std::endl;

  return K;
}


