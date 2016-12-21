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
  for(int i=0; i< N; i++)
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
// CORE COMPUTE KERNEL
//
void bijk(hc::array_view<double,2> a, hc::array_view<double,2> b, hc::array_view<double,2> c, long n)
{  

  c.discard_data();
  hc::parallel_for_each(c.get_extent(), [=](hc::index<2> idx) [[hc]]
		{
		  int row = idx[0];
		  int col = idx[1];
 		  double sum = 0;
		  
		  for(int i = 0; i < b.get_extent()[0]; i++)
		    sum += a(row, i) * b(i, col);
		  c[idx] = sum;
   		});
  c.synchronize();

}
// kernel
//

//
// STUPID (CPU) CORE COMPUTE KERNEL
//
void bijk_cpu(hc::array_view<double,2> a, hc::array_view<double,2> b, hc::array_view<double,2> c, long n)
{
  long i,j,k,kk,jj;
  double sum;
  int bsize = 8;
  //int en = bsize * (n/bsize); // amount that fits evenly into block size
  int en = 3000;
  //std::cout << b[0][0] << std::endl;
  
  // nested loop
  for(kk = bsize; kk < en; kk += bsize)
    {
      //std::cout << kk << std::endl;
      for(jj = 0; jj < en; jj += bsize)
	{
	  //std::cout << jj << std::endl;
	  for(i=0; i< n; i++)
	    {
	      //std::cout << 'row ' << int(i) << std::endl;
	      for(j = jj; j < jj + bsize; j++)
		{
		  sum = c[i][j];
		  for(k = kk; k < kk + bsize; k++)
		    {
		      sum += a[i][k] * b[k][j];
		    }
		  c[i][j] = sum;
		}
	    }
	}
    }
  
}
// kernel
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
  hc::array_view<double, 2> a(N, N, &my_composed_vector.front());

  vector<double> my_composed_vector_b;
  for(int i = 0, ie = B.size(); i != ie; ++i)
    my_composed_vector_b.insert(my_composed_vector_b.end(), B[i].begin(), B[i].end());
  hc::array_view<double, 2> b(N, N, &my_composed_vector_b.front());
  
  // build C matrix  
  vector<vector<double> > C = build_C(N);
  vector<double> my_composed_vector_c;
  for(int i = 0, ie = C.size(); i != ie; ++i)
    my_composed_vector_c.insert(my_composed_vector_c.end(), C[i].begin(), C[i].end());
  hc::array_view<double, 2> c(N, N, &my_composed_vector_c.front());
  
  //
  // EXECUTE
  //
  std::cout << "execute\n";
  double start = get_wtime();
  bijk(a, b, c, N);
  //bijk_cpu(a, b, c, N);
  double end = get_wtime();
  std::cout << "FOM (sec) = " <<  end - start << std::endl;    
  //
  //
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

  // acthung! need to write C matrix!
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
