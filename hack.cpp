/*
    matmul

    Sample contest submission to demonstrate expectations for file I/O, FOM and contest judging

    Note: this code is not optimized and will not be used as an actual submission. 
          This code has been minimally tested and should be correct. Bugs and errors may exist. 

    Use:

    matmul <A matrix filename to read> <B matrix filename to read> <C matrix filename to write>

	 *** Please include your name in the comments ***
    Submitter's Name: Sample Submission

*/

#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include <string.h>
#include <sys/time.h>

#include <hcc/hc.hpp>

#define ERROR (1)
#define OK (0)


typedef struct 
{
	long N; // square NxN matrix
	double **mat;
} MatS;

//
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
// kernel
//

// allocate memory for and zero MatS data
MatS * CreateMat(long N)
{
	long i,j;

	MatS *m = (MatS *)malloc(sizeof(MatS));
	if(m == NULL) 
	{
		printf("Memory allocation error\n");
		exit(1);
	}

	m->N = N;

	m->mat = (double**)malloc(sizeof(double*) * N);
	m->mat[0] = (double*)malloc(sizeof(double) * N * N);
	for(i=1; i< N; i++)
	{
	  m->mat[i] = m->mat[0] + i * N;
	}

	for(i=0; i< N; i++)
		for(j=0; j<N; j++)
			m->mat[i][j] = 0.0;

	return(m);
}

void DestroyMat(MatS *m)
{
	long i,j;

	// defensive clear memory
	for(i=0; i < m->N; i++)
		for(j=0; j < m->N; j++)
			m->mat[i][j] = 0.0;

	free(m->mat[0]);
	free(m->mat);
	m->mat = NULL;

	m->N = 0;
	free(m);
	m = NULL;
}

int WriteMatrixToFile(FILE *fp, MatS *mat)
{
	long i,j;

	if(fp == NULL) return(ERROR);

	fprintf(fp, "%ld\n", mat->N);

	for(i=0; i< mat->N; i++)
	{
		for(j=0; j< mat->N; j++)
		{
			fprintf(fp, "%lf, ", mat->mat[i][j]);
		}
		fprintf(fp, "\n");
	}

	return(OK);
}

MatS * ReadMatrixFromFile(FILE *fp)
{
	MatS *m;
	long N;
	long i,j;
	int rval;

	if(fp == NULL) return NULL;

	fscanf(fp, "%ld\n", &N);

	printf("File contains %lu x %lu matrix\n", N, N);

	m = CreateMat(N);

	if(m==NULL) return NULL;

	for(i=0; i<N; i++)
	{
		for(j=0; j<N; j++)
		{
			rval = fscanf(fp, "%lf,", &m->mat[i][j]);
			if(rval == EOF)
			{
				printf("Unexpected end of file error\n");
				DestroyMat(m);
				return(NULL);
			}
			else if(rval == 0)
			{
				printf("Unexpected character in matrix file at (%ld,%ld). Could not properly read the matrix\n", i,j);
				DestroyMat(m);
				return(NULL);
			}
		}
		fscanf(fp, "\n");
	}

	return(m);
}

//	MSEMat - compute the mean squared error between 
//	two matrices.
double MSECalc(MatS *C, MatS *G)
{
	double mse = 0.0, err;
	long i,j;
	long N;

	if(C->N != G->N)
	{
		printf("Matrices must be the same size: Mat C size = %lu, Gold size = %lu", C->N, G->N);
		exit(ERROR);
	}

	N = C->N;

	for(i = 0; i < N; i++)
	{
		for(j=0; j < N; j++)
		{
			err = C->mat[i][j] - G->mat[i][j];
			mse += err * err;
		}
	}

	return(mse);
}

double get_wtime()
{
	double t;
	struct timeval tm;
	gettimeofday(&tm, NULL);
	t = tm.tv_sec + (tm.tv_usec/1.0E6);
	return(t);
}

// Convert fractional time to nanosecond counter ticks
uint64_t Get_Time_ns_ticks()
{
	return( (uint64_t)(1.0E9 * get_wtime()));
}

// Creates memory for C and computes C = AB
MatS * matmul(MatS *A, MatS *B)
{
	MatS *C = NULL;
	long N;
	double start, end;

	if(A->N != B->N)
	{
		printf("A and B matricies need to be the same size\n");
		return(NULL);
	}

	N = A->N;

	C = CreateMat(N);
	if(C == NULL)
	{
		printf("Could not create C matrix of size %lu x %lu\n", N, N);
		return(C);
	}

	// hard-code block-size. Note: block size must be divisible into matrix size for correctness
	// Don't make assumptions like this in a real submission. The contest judging problem matrix
	// sizes may be even or odd and may or may not be divisible. The judges will not modify code
	// to account for this. Judges will not re-compile code specifically for a problem size. 
	// handle these edge conditions in your code at runtime. 

	// FOM calculation should be similar to what is done here
	start = get_wtime();
	bijk(A,B,C, N, 8);
	end = get_wtime();

	printf("FOM (sec) = %.4lf\n", end - start);  
	return(C);
}


int main(int argc, char *argv[])
{
	MatS *A = NULL;
	MatS *B = NULL;
	MatS *C = NULL;
	MatS *G = NULL;
	FILE *fp_A;
	FILE *fp_B;
	FILE *fp_C;
	FILE *fp_G;
	char *filename_A = argv[1];
	char *filename_B = argv[2];
	char *filename_C = argv[3];
	char *filename_G = argv[4];

	if(argc < 5)
	{
		printf("Filenames must be specified on the command line\n");
		printf("Filenames A and B will be given and read in\n");
		printf("Filename C will be written by this code according to the matrix format specified\n");
		printf("%s <A mat> <B mat> <C mat> <G mat>\n", argv[0]);
		exit(1);
	}
	fp_A = fopen(filename_A, "r");
	fp_B = fopen(filename_B, "r");
	fp_C = fopen(filename_C, "w");
	fp_G = fopen(filename_G, "r");

	if(fp_A == NULL || fp_B == NULL || fp_G == NULL)
	{
		printf("Unable to open input matrices for reading\n");
		exit(1);
	}

	if(fp_C == NULL)
	{
		printf("Unable to open output file for matrix C\n");
		exit(1);
	}

	if( NULL == (A = ReadMatrixFromFile(fp_A)))
	{
		printf("Unable to read matrix A data\n");
		exit(1);
	}

	if( NULL == (B = ReadMatrixFromFile(fp_B)))
	{
		printf("Unable to read matrix B data\n");
		exit(1);
	}

	if( NULL == (G = ReadMatrixFromFile(fp_G)))
	{
		printf("Unable to read matrix G data\n");
		exit(1);
	}
	printf("Gentlemen, start your engines!\n");  
	C = matmul(A,B);

	// verify solution
	printf("MSE       = %.4lf\n", MSECalc(C,G));  
	       
	if(C != NULL) 
		WriteMatrixToFile(fp_C, C);
	else
		printf("Unable to write matrix C to file\n");

	fclose(fp_A);
	fclose(fp_B);
	fclose(fp_C);
	fclose(fp_G);       

	DestroyMat(A);
	DestroyMat(B);
	DestroyMat(C);
	DestroyMat(G);

	// steady as she goes
	return(0);
}
// done
// nick
// amd research
// 12-15-16
