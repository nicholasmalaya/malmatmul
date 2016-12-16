#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <math.h>
#include "Mat.h"


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
