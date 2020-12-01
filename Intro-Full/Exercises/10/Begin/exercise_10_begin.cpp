#include<Kokkos_Core.hpp>
//EXERCISE: include the right header (later Kokkos will include this)
#include<simd.hpp>

void test_simd(int N_in, int M, int R, double a) {

  //EXERCISE: get the right type here for CUDA/Non-Cuda
  #ifdef KOKKOS_ENABLE_CUDA
  using simd_t = simd::simd<double,simd::simd_abi::cuda_warp<32>>;
  #else
  using simd_t = simd::simd<double,simd::simd_abi::native>;
  #endif
  using simd_storage_t = simd_t::storage_type;

  //EXERCISE: What will the N now be?
  int N = N_in/simd_t::size();

  //EXERCISE: create SIMD Views instead
  Kokkos::View<simd_storage_t**> data("D",N,M);
  Kokkos::View<simd_storage_t*> results("R",N);

  // EXERCISE: create correctly a scalar view of results and data
  // For the final reduction we gonna need a scalar view of the data for now
  // Relying on knowing the data layout, we will add SIMD Layouts later
  // so that simple copy construction/assgnment would work
  Kokkos::View<double**> data_scalar((double*)data.data(),N_in,M);
  Kokkos::View<double*> results_scalar((double*)results.data(),N_in);

  // Lets fill the data deep_copy into scalar types doesn't work correctly for cuda_warp right now
  Kokkos::deep_copy(data_scalar,a);
  Kokkos::deep_copy(results_scalar,0.0);

  Kokkos::Timer timer;
  for(int r = 0; r<R; r++) {
    //EXERCISE: use TeamPolicy here
    // TeamPolicy<>(league_size=N,team_size=1,vector_size)
    // This TeamPolicy gives us a team_size of 1, so each league_rank is the thread id
    Kokkos::parallel_for("Combine",Kokkos::TeamPolicy<>(data.extent(0),1,simd_t::size()),
			 KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      //EXERCISE Use the correct type here
      simd_t tmp = 0.0;
      double b = a;
      
      // i is the thread id s.t. every kokkos thread accumulate values into tmp
      // on cuda, this means that every time tmp is accumulated into, all 32 vector lanes are reducing products into separate "elements" of tmp.
      const int i = team.league_rank();
      for(int j=0; j<data.extent(1); j++) {
        //EXERCISE: add storage_type to temporary type conversion
        tmp += b * simd_t(data(i,j));
        b+=a+1.0*(j+1);
      }
      results(i) = tmp;
    });
    Kokkos::fence();
  }

  double time = timer.seconds();

  double value = 0.0;
  // Lets do the reduction here
  Kokkos::parallel_reduce("Reduce",results_scalar.extent(0), KOKKOS_LAMBDA(const int i, double& lsum) {
    lsum += results_scalar(i);
  },value);

  printf("SIMD Time: %lf ms ( %e )\n",time*1000,value);
}

void test_team_vector(int N, int M, int R, double a) {

  constexpr int V = 32;
  Kokkos::View<double**> data("D",N,M);
  Kokkos::View<double*> results("R",N);

  Kokkos::deep_copy(data,a);
  Kokkos::deep_copy(results,0.0);

  Kokkos::Timer timer;
  for(int r = 0; r<R; r++) {
    Kokkos::parallel_for("Combine",Kokkos::TeamPolicy<>(data.extent(0)/V,1,V), 
      KOKKOS_LAMBDA(const Kokkos::TeamPolicy<>::member_type& team) {
      double b = a;
      const int i = team.league_rank()*V;
      for(int j=0; j<data.extent(1); j++) {
        Kokkos::parallel_for(Kokkos::ThreadVectorRange(team,V), 
          [&] (const int ii) {  
            results(i+ii) += b * data(i+ii,j);
        });
        b+=a+1.0*(j+1);
      }
    });
    Kokkos::fence();
  }

  double time = timer.seconds();

  double value = 0.0;
  Kokkos::parallel_reduce("Reduce",N, KOKKOS_LAMBDA(const int i, double& lsum) {
    lsum += results(i);
  },value);

  printf("ThreadVector Time: %lf ms ( %e )\n",time*1000,value/R);
}


int main(int argc, char* argv[]) {
  Kokkos::initialize(argc,argv);
  
  int N = argc>1?atoi(argv[1]):320000;
  int M = argc>2?atoi(argv[2]):3;
  int R = argc>3?atoi(argv[3]):10;
  double scal = argc>4?atof(argv[4]):1.5;
  
  if(N%32) {
    printf("Please choose an N dividable by 32\n");
    return 0;
  }

  test_team_vector(N,M,R,scal);
  test_simd(N,M,R,scal);

  Kokkos::finalize();
}
