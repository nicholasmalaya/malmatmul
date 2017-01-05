/*
    malmatmul
    MALaya MATrix MULtiply

    Use:

    malmatmul <A matrix filename to read> <B matrix filename to read> <C matrix filename to write>
    Submitter's Name: Nick

*/

#include <sys/time.h>
#include <stdio.h>
#include <iostream>
#include <fstream>  
#include <string>
#include <vector>
#include <stdlib.h>

using std::vector;
using std::string;

#include <hcc/hc.hpp>

vector<vector<double> > ReadMatrixFromFile(FILE *fp)
{
  int err, rval;
  long n, i, j;

  // grab first line, sizeof matrix (n)
  err = fscanf(fp, "%ld\n", &n);
  printf("File contains %lu x %lu matrix\n", n, n);

  // build 2d matrix
  vector<vector<double> > m;
  m.resize(n);
  for (i = 0; i < n; ++i)
    m[i].resize(n);

  // MARK IT ZERO
  for(i=0; i< n; i++)
    for(j=0; j<n; j++)
      m[i][j] = 0.0;

  // read in new values
  for(i=0; i<n; i++)
    {
      for(j=0; j<n; j++)
	{
	  rval = fscanf(fp, "%lf,", &m[i][j]);
	}
      err += fscanf(fp, "\n");
    }

  return(m);

}
 
vector<vector<double> > build_C(long N)
{
  long i,j;
  
  vector<vector<double> > m;
  m.resize(N);
  for (i = 0; i < N; ++i)
    m[i].resize(N);
  
  // // MARK IT ZERO
  for(i=0; i< N; i++)
    for(j=0; j<N; j++)
      m[i][j] = 0.0;
  
  return(m);
}

int WriteMatrixToFile(FILE *fp, hc::array_view<double,2> mat, long N)
{
  long i,j;
  if(fp == NULL) return(1);
  
  fprintf(fp, "%ld\n", N);
  for(i=0; i< N; i++)
    {
      for(j=0; j< N; j++)
	{
	  fprintf(fp, "%lf, ", mat[i][j]);
	}
      fprintf(fp, "\n");
    }
  
  return(0);
}

//
// GPU ADD KERNEL
//
void GPU_ADD(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c)
{  

  c.discard_data();
  hc::parallel_for_each(c.get_extent(), [=](hc::index<2> idx) [[hc]]
		{
		  c[idx] = a[idx] + b[idx];
   		});
  c.synchronize();

}

//
// GPU SUBTRACT KERNEL
//
void GPU_SUB(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c)
{  

  c.discard_data();
  hc::parallel_for_each(c.get_extent(), [=](hc::index<2> idx) [[hc]]
		{
		  c[idx] = a[idx] - b[idx];
   		});
  c.synchronize();

}


//
// GPU MULTIPLY KERNEL
//
void GPU_MULT(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c)
{  

  c.discard_data();
  hc::parallel_for_each(c.get_extent(), [=](hc::index<2> idx) [[hc]]
		{
		  int row = idx[0];
		  int col = idx[1];
 		  double sum = 0;
		  
		  for(long i = 0; i < b.get_extent()[0]; i++)
		    {
		      sum += a(row, i) * b(i, col);
		    }
		  
		  c[idx] = sum;
   		});
  c.synchronize();

}
// kernel
//
//
// https://msdn.microsoft.com/en-us/library/hh873133.aspx

//
// GPU MULTIPLY W/ TILES 
//
template <const int TS> void GPU_TILE(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c, long N)
{  

  hc::extent<2> ex(N,N);
  hc::tiled_extent<2> t_ex = ex.tile_with_dynamic(TS,TS,TS);
  
  c.discard_data();
  hc::parallel_for_each(t_ex, [=](hc::tiled_index<2> t_idx) [[hc]]
		{
		  
		  // local 
		  int row  = t_idx.local[0];
		  int col  = t_idx.local[1];
		  
		  // global
		  int rowG = t_idx.global[0];
		  int colG = t_idx.global[1];

 		  double sum = 0;		  
		  for(long i = 0; i < N; i += TS)
		    {
		      tile_static double locA[TS][TS]; 
		      tile_static double locB[TS][TS];
		      locA[row][col] = a(rowG, col + i);
		      locB[row][col] = b(row + i, colG);
		      
		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{  
			  sum += locA[row][k] * locB[k][col];  
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  c[t_idx] = sum;		  
   		});
  // c.synchronize();
  // copying is implicit when array_view is out of scope, c++ ftw

}
// kernel
//

//
// CORE COMPUTE KERNEL
//
template <const int TS> void GPU_STRASSEN(hc::array_view<const double,2> a, hc::array_view<const double,2> b, hc::array_view<double,2> c, long N)
{
  //
  // if you are here, we KNOW we are divisible by two!
  //
  long Nh = N/2;
  hc::extent<2> ex(Nh,Nh);
  hc::tiled_extent<2> t_ex = ex.tile_with_dynamic(TS,TS,TS);

  // Strassen matrices
  hc::array<double,2> P1(ex);
  hc::array<double,2> P2(ex);
  hc::array<double,2> P3(ex);
  hc::array<double,2> P4(ex);
  hc::array<double,2> P5(ex);
  hc::array<double,2> P6(ex);
  hc::array<double,2> P7(ex);

  // legit temp matricies
  hc::array<double,2> T1(ex);
  hc::array<double,2> T2(ex);
  
  //c.discard_data();
  hc::parallel_for_each(t_ex, [=](hc::tiled_index<2> t_idx) [[hc]]
		{
		  
		  // local 
		  int row  = t_idx.local[0];
		  int col  = t_idx.local[1];
		  
		  // global
		  int rowG = t_idx.global[0];
		  int colG = t_idx.global[1];

		  // -----------------------------
		  // Calculate P1!
		  // -----------------------------
		  // can we add tiling here?
		  // need four tiles!
		  double sum = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA11[TS][TS]; 
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      tile_static double locB22[TS][TS];
		      locA11[row][col] = a(rowG, col + i);
		      locB11[row][col] = b(row + i, colG);
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum += (locA11[row][k]+locA22[row][k])*(locB11[k][col]+locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  //P1[t_idx] = sum;
		  //c[t_idx] = sum;		  

		  // -----------------------------
		  // Calculate P2!
		  // -----------------------------
		  sum = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA21[TS][TS]; 
		      tile_static double locA22[TS][TS]; 
		      tile_static double locB11[TS][TS];
		      locA21[row][col] = a(rowG + Nh, col + i);
		      locB11[row][col] = b(row + i, colG);
		      locA22[row][col] = a(rowG + Nh, col + i + Nh);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum += (locA21[row][k]+locA22[row][k])*(locB11[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  //P2[t_idx] = sum;
		  //c[t_idx] = sum;		  

		  // -----------------------------
		  // Calculate P3!
		  // -----------------------------
		  sum = 0;		  
		  for(long i = 0; i < Nh; i += TS)
		    {
		      tile_static double locA11[TS][TS]; 
		      tile_static double locB12[TS][TS]; 
		      tile_static double locB22[TS][TS];
		      locA11[row][col] = a(rowG, col + i);
		      locB12[row][col] = b(row + i, colG + Nh);
		      locB22[row][col] = b(row + i + Nh, colG + Nh);

		      // threads in tile all wait until locA,locB are filled.  
		      t_idx.barrier.wait();
		      for (long k = 0; k < TS; k++)
			{
			  sum += (locA11[row][k])*(locB12[k][col]-locB22[k][col]); 
			}
		      // all threads wait until sums are calculated. 
		      t_idx.barrier.wait();
		      
		    }  
		  //P3[t_idx] = sum;
		  c[t_idx] = sum;		  

		  
		  
		  // //
		  // // final matrix assembly
		  // if( (rowG < N/2 ) and (colG < N/2 ) )              // C_11
		  //   {
		  //     // C11 = P1 + P4 - P5 + P7
		  //     c[t_idx] = a(rowG,colG) + b(rowG,colG);
		  //   }  
		  // else if( (rowG < N/2 ) and (colG > N/2 - 1 ) )     // C_12
		  //   {
		  //     // C12 = P3 + P5
		  //     c[t_idx] = a(rowG,colG) + b(rowG,colG);
		  //   }
		  // else if( (rowG > N/2 - 1 ) and (colG < N/2 ) )     // C_21
		  //   {
		  //     // C21 = P2 + P4
		  //     c[t_idx] = a(rowG,colG) + b(rowG,colG);
		  //   }
		  // else if( (rowG > N/2 - 1 ) and (colG > N/2 - 1 ) ) // C_22
		  //   {
		  //     // C22 = P1 + P3 - P2 + P6
		  //     c[t_idx] = a(rowG,colG) + b(rowG,colG);
		  //   }
		  
		});
  c.synchronize();

}
//
// https://github.com/arbenson/fast-matmul/blob/master/codegen/algorithms/strassen
//


//
//	MSEMat - compute the mean squared error between 
//	two "matrices".
//
double MSECalc(hc::array_view<double, 2> C, hc::array_view<double, 2> G, long N)
{
  double mse = 0.0, err;
  long i,j;
  
  for(i = 0; i < N; i++)
    {
      for(j=0; j < N; j++)
	{
	  err = C[i][j] - G[i][j];
	  mse += err * err;
	}
    }
  
  return(mse);
}


//
// get time information
//
double get_wtime()
{
	double t;
	struct timeval tm;
	gettimeofday(&tm, NULL);
	t = tm.tv_sec + (tm.tv_usec/1.0E6);
	return(t);
}


int main(int argc, char *argv[])
{
  // simple error handling
  if(argc < 4)
    {
      printf("Filenames must be specified on the command line\n");
      printf("Filenames A and B will be given and read in\n");
      printf("Filename C will be written by this code\n");
      printf("Filename G (if present) will be 'true' solution\n");
      printf("%s <A mat> <B mat> <C mat> <G mat>\n", argv[0]);
      exit(1);
    }
  
  FILE *fp_A;
  FILE *fp_B;
  FILE *fp_C;
  FILE *fp_G;
  char *filename_A = argv[1];
  char *filename_B = argv[2];
  char *filename_C = argv[3];
  
  fp_A = fopen(filename_A, "r");
  fp_B = fopen(filename_B, "r");
  fp_C = fopen(filename_C, "w");

  // modest error handling for file IO
  if(fp_A == NULL || fp_B == NULL)
    {
      std::cout << filename_A;
      printf("\nUnable to open input matrices for reading\n");
      exit(1);
    }

  if(fp_C == NULL)
    {
      printf("Unable to open output file for matrix C\n");
      exit(1);
    }

  vector<vector<double> > A = ReadMatrixFromFile(fp_A);
  vector<vector<double> > B = ReadMatrixFromFile(fp_B);
  long N = A.size(); // assumes square!
  
  // hack it! 
  vector<double> my_composed_vector;
  for(int i = 0, ie = A.size(); i != ie; ++i)
    my_composed_vector.insert(my_composed_vector.end(), A[i].begin(), A[i].end());
  hc::array_view<const double, 2> a(N, N, &my_composed_vector.front());

  vector<double> my_composed_vector_b;
  for(int i = 0, ie = B.size(); i != ie; ++i)
    my_composed_vector_b.insert(my_composed_vector_b.end(), B[i].begin(), B[i].end());
  hc::array_view<const double, 2> b(N, N, &my_composed_vector_b.front());
  
  // build C matrix  
  vector<vector<double> > C = build_C(N);
  vector<double> my_composed_vector_c;
  for(int i = 0, ie = C.size(); i != ie; ++i)
    my_composed_vector_c.insert(my_composed_vector_c.end(), C[i].begin(), C[i].end());
  hc::array_view<double, 2> c(N, N, &my_composed_vector_c.front());
  
  // Provide Accelerator Details -- if not AMD device it is in ERROR
  vector<hc::accelerator> accs = hc::accelerator::get_all();
  hc::accelerator chosen_one;
  std::wcout << chosen_one.get_description() << std::endl;  // you were the chosen one! 
  
  //
  // find largest power of two I can fit... 
  //
  int TS = 1;
  //static const int TS = 8; // equivalent to warp of 64
  int tiles[4] = {2,4,8,16};   
  for(int i = 0; i < 3; i++) // NOT a bug: 8 was best, 16 always degraded performance
    {
      if(N%tiles[i] == 0)
	{
	  TS=tiles[i];
	}
    }
  assert( (N%TS) == 0); // mild sanity check
  std::cout << "Using Tile Size of: " << TS << std::endl;
  
  //
  // EXECUTE
  //
  std::cout << "Execute\n";
  double start = get_wtime();
  //GPU_MULT(a, b, c);

  if(N%2==0) // can use at least one level of strassen
    {
      GPU_STRASSEN<2>(a,b,c,N);
    }
  else
    {
      if(TS == 16)
	GPU_TILE<16>(a, b, c, N);
      else if(TS == 8)
	GPU_TILE<8>(a, b, c, N);
      else if(TS == 4)
	GPU_TILE<4>(a, b, c, N);
      else if(TS == 2)
	GPU_TILE<2>(a, b, c, N);
      else
	GPU_TILE<1>(a, b, c, N);
    }

  
  double end = get_wtime();
  std::cout << "FOM (sec) = " <<  end - start << std::endl;    
  //
  // FINISHED WITH EXECUTION 
  //
  
  if(argc == 5) // if gold matrix present
    {
      std::cout << "Gold Matrix Present" << std::endl;
      std::cout << "Error Checking Enabled\n";
      
      char *filename_G = argv[4];
      fp_G = fopen(filename_G, "r");	
      vector<vector<double> > G = ReadMatrixFromFile(fp_G);
      
      // actually do i even need this?
      vector<double> my_composed_vector_g;
      for(int i = 0, ie = C.size(); i != ie; ++i)
	my_composed_vector_g.insert(my_composed_vector_g.end(), G[i].begin(), G[i].end());
      hc::array_view<double, 2> g(N, N, &my_composed_vector_g.front());
      
      // verify solution
      std::cout << "MSE       = " << MSECalc(c,g,N) << std::endl;
      fclose(fp_G);         
    }

  // actung! need to write C matrix!
  WriteMatrixToFile(fp_C, c, N);  

  // clean up
  fclose(fp_A);
  fclose(fp_B);
  fclose(fp_C);
  
  // steady as she goes
  std::cout << "Exiting Normally..." << std::endl;
  exit(0);
}

// nick
// AMD Research
// 12-15-16
//
