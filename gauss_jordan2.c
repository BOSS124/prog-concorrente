#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>

#define RANK_ROOT				0
#define MATRIX_MAX_DIMENSION	10000

#define ERROR_INCORRECT_ARGC	-1

int main(int argc, char **argv) {
	int i, j, k;

	/* Número de threads por processo passado como parâmetro ou igual a 2 */
	int thread_count;
	thread_count = (argc == 2) ? atoi(argv[1]) : 2;

	/* Inicializa o ambiente de execução MPI */
	MPI_Init(NULL, NULL);

	/* world_size guarda o número de processos criados */
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	/* world_rank guarda o rank do processo */
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	double *matrix = NULL;
	double **row_pointers = NULL;
	double *vector = NULL;
	unsigned int dim;

	if(world_rank == RANK_ROOT) {
		matrix = (double *) malloc(MATRIX_MAX_DIMENSION * MATRIX_MAX_DIMENSION * sizeof(double));
		FILE *file = fopen("matriz.txt", "r");
		unsigned int read = 0;
		while(!feof(file))
			fscanf(file, "%lf", &matrix[read++]);

		fclose(file);

		dim = (unsigned int) sqrt((double) read);
		row_pointers = (double**) malloc(dim * sizeof(double *));
		for(i = 0; i < dim; i++) {
			row_pointers[i] = &matrix[i * dim];
		}

		file = fopen("vetor.txt", "r");
		vector = (double *) malloc(dim * sizeof(double));
		read = 0;
		while(!feof(file))
			fscanf(file, "%lf", &vector[read++]);

		fclose(file);

		for(i = 0; i < dim; i++) {
			if(matrix[i * dim + i] == 0) {
				for(j = 0; j < dim; j++) {
					if((j != i) && (matrix[j * dim + i] != 0) && (matrix[i * dim + j] != 0)) {
						double *aux = (double *) malloc(dim * sizeof(double));

						memcpy(aux, &matrix[j * dim], dim * sizeof(double));
						memcpy(&matrix[j * dim], &matrix[i * dim], dim * sizeof(double));
						memcpy(&matrix[i * dim], aux, dim * sizeof(double));

						free(aux);

						aux = row_pointers[i];
						row_pointers[i] = row_pointers[j];
						row_pointers[j] = aux;
						break;
					}
				}
			}
		}
	}

	MPI_Bcast(&dim, 1, MPI_UNSIGNED, RANK_ROOT, MPI_COMM_WORLD);

	double *pivot_row = (double *) malloc(dim * sizeof(double));
	double *up_rows = NULL;
	double *up_vector = NULL;
	double *update_rows = NULL;
	double *update_vector = NULL;
	double pivot_vector_element;
	double *result_buffer;
	double *result_vector;

	int *displs = NULL;
	int *displs2 = NULL;
	int *scounts = NULL;
	int *scounts2 = NULL;

	displs = (int *) calloc(world_size, sizeof(int));
	scounts = (int *) calloc(world_size, sizeof(int));
	unsigned int rows_per_process = (unsigned int) ceil((double) dim / (double) world_size);

	for(j = 0, k = 0; j < world_size; j++) {
		displs[j] = j * rows_per_process * dim;
		if((k + rows_per_process * dim) <= ((dim - 1) * dim)) {
			scounts[j] = rows_per_process * dim;
			k += rows_per_process * dim;
		}
		else {
			scounts[j] = ((dim - 1) * dim) - k;
			k += ((dim - 1) * dim) - k;
			if(scounts[j] < 0) scounts[j] = 0;
		}
	}

	up_rows = (double *) malloc(scounts[world_rank] * sizeof(double));
	result_buffer = (double *) malloc(scounts[world_rank] * sizeof(double));

	displs2 = (int *) calloc(world_size, sizeof(int));
	scounts2 = (int *) calloc(world_size, sizeof(int));

	for(j = 0, k = 0; j < world_size; j++) {
		displs2[j] = j * rows_per_process;
		if((k + rows_per_process) <= (dim -1)) {
			scounts2[j] = rows_per_process;
			k += rows_per_process;
		}
		else {
			scounts2[j] = (dim - 1) - k;
			k += (dim - 1) - k;
			if(scounts2[j] < 0) scounts2[j] = 0;
		}
	}

	up_vector = (double *) malloc(scounts2[world_rank] * sizeof(double));
	result_vector = (double *) malloc(scounts2[world_rank] * sizeof(double));

	for(i = 0; i < dim; i++) {
		if(world_rank == RANK_ROOT) {
			if(!update_rows)
				update_rows = (double *) malloc((dim - 1) * dim * sizeof(double));
			if(!update_vector)
				update_vector = (double *) malloc((dim - 1) * sizeof(double));

			int rindex = 0;

			for(j = 0; j < dim; j++) {
				if(j != i) {
					update_vector[rindex] = vector[j];
					memcpy(&update_rows[(rindex++) * dim], &matrix[j * dim], dim * sizeof(double));
				}
			}

			memcpy(pivot_row, &matrix[i * dim], dim * sizeof(double));
			pivot_vector_element = vector[i];
		}

		MPI_Bcast(pivot_row, dim, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

		MPI_Bcast(&pivot_vector_element, 1, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

		MPI_Scatterv(update_rows, scounts, displs, MPI_DOUBLE, up_rows, scounts[world_rank], MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
		MPI_Scatterv(update_vector, scounts2, displs2, MPI_DOUBLE, up_vector, scounts2[world_rank], MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

		#pragma omp parallel for num_threads(thread_count) shared(dim)
		for(j = 0; j < scounts2[world_rank]; j++) {
			for(k = 0; k < dim; k++) {
				result_buffer[j * dim + k] = (pivot_row[i] * up_rows[j * dim + k]) - (up_rows[j * dim + i] * pivot_row[k]);
				result_vector[j] = (pivot_row[i] * up_vector[j]) - (up_rows[j * dim + i] * pivot_vector_element);
			}
		}

		MPI_Gatherv(result_buffer, scounts[world_rank], MPI_DOUBLE, update_rows, scounts, displs, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
		MPI_Gatherv(result_vector, scounts2[world_rank], MPI_DOUBLE, update_vector, scounts2, displs2, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

		if(world_rank == RANK_ROOT) {
			int rindex = 0;

			for(j = 0; j < dim; j++) {
				if(j != i) {
					vector[j] = update_vector[rindex];
					memcpy(&matrix[j * dim], &update_rows[(rindex++) * dim], dim * sizeof(double));
				}
			}
		}
	}

	if(world_rank == RANK_ROOT) {
		#pragma omp parallel for private(i, j) num_threads(thread_count)
		for(i = 0; i < dim; i++) {
			double multiplier = (1 / (*(row_pointers[i] + i)));
			for(j = 0; j < dim; j++)
				*(row_pointers[i] + j) *= multiplier;
			vector[i] *= multiplier;
		}

		FILE *file = fopen("resultado.txt", "w");
		for(i = 0; i < dim; i++) {
			fprintf(file, "%.3f\n", vector[i]);
		}
		fclose(file);
	}
	
	/* Cleanup */
	free(matrix);
	free(row_pointers);
	free(vector);
	free(pivot_row);
	free(up_rows);
	free(up_vector);
	free(update_rows);
	free(update_vector);
	/* ------- */

	/* Encerra o ambiente de execução MPI */
	MPI_Finalize();

	return 0;
	/* END */
}