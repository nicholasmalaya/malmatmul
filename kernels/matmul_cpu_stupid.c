// Note: you do not need to use the matrix routines included in the Mat.h header.
//       Feel free to implement your own code to tune memory layout, etc. 
#include "Mat.h" 

//#include <hcc/hc.hpp>
//#include "rocblas.h"

// Note: code was copied from http://csapp.cs.cmu.edu/2e/waside/waside-blocking.pdf
//       this would disqualify this entry in the contest. This is an example entry only.
void bijk(MatS *A, MatS *B, MatS *C, long n, int bsize)
{
	long i,j,k,kk,jj;
	double sum;
	int en = bsize * (n/bsize); // amount that fits evenly into block size

	// zeroing matrix
	for(i=0; i<n; i++)
		for(j=0; j<n; j++)
			C->mat[i][j] = 0.0;

	// nested loop
	for(kk = 0; kk < en; kk += bsize)
	{
		for(jj = 0; jj < en; jj += bsize)
		{
			for(i=0; i< n; i++)
			{
				for(j = jj; j < jj + bsize; j++)
				{
					sum = C->mat[i][j];
					for(k = kk; k < kk + bsize; k++)
					{
						sum += A->mat[i][k] * B->mat[k][j];
					}
					C->mat[i][j] = sum;
				}
			}
		}
	}
}
