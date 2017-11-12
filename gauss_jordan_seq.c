#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MATRIX_MAX_DIMENSION	10000

double time_elapsed_milis(clock_t end, clock_t begin) {
	return ((double) (end - begin) / (double) (CLOCKS_PER_SEC / 1000));
}

int main(int argc, char *argv[]) {
	int i, j, k;
	clock_t begin, end;

	begin = clock();

	double *matrix = (double *) malloc(MATRIX_MAX_DIMENSION * MATRIX_MAX_DIMENSION * sizeof(double));
	unsigned int dim;

	FILE *file = fopen("matriz.txt", "r");
	unsigned int read = 0;
	while(!feof(file))
		fscanf(file, "%lf", &matrix[read++]);
	fclose(file);

	dim = (unsigned int) sqrt((double) read);

	double **row_pointers = (double**) malloc(dim * sizeof(double *));
	for(i = 0; i < dim; i++) {
		row_pointers[i] = &matrix[i * dim];
	}

	file = fopen("vetor.txt", "r");
	double *vector = (double *) malloc(dim * sizeof(double));
	read = 0;
	while(!feof(file))
		fscanf(file, "%lf", &vector[read++]);
	fclose(file);

	double **vector_pointers = (double **) malloc(dim * sizeof(double *));
	for(i = 0; i < dim; i++) {
		vector_pointers[i] = &vector[i];
	}

	for(i = 0; i < dim; i++) {
		if(matrix[i * dim + i] == 0.0f) {
			for(j = 0; j < dim; j++) {
				if((j != i) && (matrix[j * dim + i] != 0) && (matrix[i * dim + j] != 0)) {
					double *aux = (double *) malloc(dim * sizeof(double));
					double aux2;

					memcpy(aux, &matrix[j * dim], dim * sizeof(double));
					memcpy(&matrix[j * dim], &matrix[i * dim], dim * sizeof(double));
					memcpy(&matrix[i * dim], aux, dim * sizeof(double));
					free(aux);

					aux2 = vector[i];
					vector[i] = vector[j];
					vector[j] = aux2;

					aux = row_pointers[i];
					row_pointers[i] = row_pointers[j];
					row_pointers[j] = aux;

					aux = vector_pointers[i];
					vector_pointers[i] = vector_pointers[j];
					vector_pointers[j] = aux;
					break;
				}
			}
		}
	}

	double *updated_row = (double *) malloc(dim * sizeof(double));
	for(i = 0; i < dim; i++) {
		for(j = 0; j < dim; j++) {
			if(j != i && (matrix[j * dim + i] != 0.0f)) {
				for(k = 0; k < dim; k++) {
					updated_row[k] = (matrix[i * dim + i] * matrix[j * dim + k]) - (matrix[j * dim + i] * matrix[i * dim + k]);

				}
				vector[j] = (matrix[i * dim + i] * vector[j]) - (matrix[j * dim + i] * vector[i]);
				memcpy(&matrix[j * dim], updated_row, dim * sizeof(double));
			}
		}
	}
	free(updated_row);

	for(i = 0; i < dim; i++) {
		double multiplier = (1.0f / matrix[i * dim + i]);
		vector[i] *= multiplier;
	}

	file = fopen("resultado.txt", "w");
	for(i = 0; i < dim; i++) {
		fprintf(file, "%.3lf\n", *(vector_pointers[i]));
	}
	fclose(file);

	end = clock();

	printf("Matrix_length: %u\tSequential: %lfms\n", dim, time_elapsed_milis(end, begin));

	free(matrix);
	free(row_pointers);
	free(vector);
	free(vector_pointers);


	return 0;
}