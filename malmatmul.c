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

// Note: you do not need to use the matrix routines included in the Mat.h header.
//       Feel free to implement your own code to tune memory layout, etc. 
#include "Mat.h" 

// Note: you don't need to use the timing routines, but they are here for convenience. 
//       Timing should be computed in seconds wall clock time for the FOM
#include "fom_timer.h"  

// solver
#include "matmul.h"

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

