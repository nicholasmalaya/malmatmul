/*
	Mat.h

	Matrix handling routines and data structures. 

	Note: these are not optimized for performance in any way. Use this as "documentation". These
	are not required by the contest. 


*/

#include <stdio.h>
#include <stdlib.h>

#ifndef __AMD_MATRIX_CONTEST__
#define __AMD_MATRIX_CONTEST__

#define ERROR (1)
#define OK (0)

typedef struct 
{
	long N; // square NxN matrix
	double **mat;
} MatS;

MatS * CreateMat(long N);
void DestroyMat(MatS *m);
int WriteMatrixToFile(FILE *fp, MatS *mat);
MatS * ReadMatrixFromFile(FILE *fp);

#endif