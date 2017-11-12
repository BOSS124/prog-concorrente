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

void swap_rows(float **row1_ptr, float **row2_ptr, unsigned int row_length) {
	float *aux = (float *) malloc(row_length * sizeof(float));
	float *aux_ptr = NULL;

	memcpy(aux, *row1_ptr, row_length * sizeof(float));
	memcpy(*row1_ptr, *row2_ptr, row_length * sizeof(float));
	memcpy(*row2_ptr, aux, row_length * sizeof(float));
	free(aux);

	aux_ptr = *row1_ptr;
	*row1_ptr = *row2_ptr;
	*row2_ptr = aux_ptr;
}

int main(int argc, char **argv) {
	int i, j, k;

	/* Inicializa o pseudo-gerador de números aleatórios */
	srand(time(NULL));

	/* Inicializa o ambiente de execução MPI */
	MPI_Init(NULL, NULL);

	/* world_size guarda o número de processos criados */
	int world_size;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);

	/* world_rank guarda o rank do processo */
	int world_rank;
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	float *matrix = NULL;
	float **row_pointers = NULL;
	float *vector = NULL;
	unsigned int dim;

	if(world_rank == RANK_ROOT) {
		matrix = (float *) malloc(MATRIX_MAX_DIMENSION * MATRIX_MAX_DIMENSION * sizeof(float));
		FILE *file = fopen("matriz.txt", "r");
		float input;
		unsigned int read = 0;
		while(!feof(file))
			fscanf(file, "%f", &matrix[read++]);

		fclose(file);

		dim = (unsigned int) sqrt((double) read);
		row_pointers = (float**) malloc(dim * sizeof(float *));
		for(i = 0; i < dim; i++) {
			row_pointers[i] = &matrix[i * dim];
		}

		file = fopen("vetor.txt", "r");
		vector = (float *) malloc(dim * sizeof(float));
		read = 0;
		while(!feof(file))
			fscanf(file, "%f", &vector[read++]);

		for(i = 0; i < dim; i++) {
			if(*(row_pointers[i] + i) == 0) {
				for(j = 0; j < dim; j++) {
					if((j != i) && (*(row_pointers[j] + i) != 0) && (*(row_pointers[i] + j) != 0)) {
						swap_rows(&row_pointers[i], &row_pointers[j], dim);
					}
				}
			}
		}
	}

	MPI_Bcast(&dim, 1, MPI_UNSIGNED, RANK_ROOT, MPI_COMM_WORLD);

	float *pivot_row = (float *) calloc(dim, sizeof(float));
	float *up_rows = NULL;
	float *up_vector = NULL;
	float *update_rows = NULL;
	float pivot_vector_element;

	/* Para cada linha da matriz seleciona o pivô (elemento da pertencente à diagonal principal) e atualiza os valores das outras linhas */
	for(i = 0; i < dim; i++) {

		/* Processo root aloca espaço para o buffer contendo as linhas que serão atualizadas na iteração do algoritmos (update_rows)*/
		if(world_rank == RANK_ROOT) {
			pivot_vector_element = vector[i];

			if(!update_rows)
				update_rows = (float *) malloc((dim - 1) * dim * sizeof(float));

			int rindex = 0;

			for(j = 0; j < dim; j++) {
				if(j != i) {
					memcpy(&update_rows[(rindex++) * dim], row_pointers[j], dim * sizeof(float));
				}
			}

			memcpy(pivot_row, row_pointers[i], dim * sizeof(float));
		}

		MPI_Bcast(pivot_row, dim, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);

		MPI_Bcast(&pivot_vector_element, 1, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);

		unsigned int rows_per_process = (unsigned int) (dim / world_size);
		if(!up_rows) up_rows = (float *) malloc(rows_per_process * dim * sizeof(float));
		MPI_Scatter(update_rows, rows_per_process * dim, MPI_FLOAT, up_rows, rows_per_process * dim, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);

		if(!up_vector) up_vector = (float *) malloc(rows_per_process * sizeof(float));
		MPI_Scatter(vector, rows_per_process, MPI_FLOAT, up_vector, rows_per_process, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);

		for(j = 0; j < rows_per_process; j++) {
			for(k = 0; k < dim; k++) {
				up_rows[j * dim + k] = (pivot_row[i] * up_rows[j * dim + k]) - (up_rows[i * dim + i] * pivot_row[k]);
				up_vector[j] = (pivot_row[i] * up_vector[j]) - (up_rows[i * dim + i] * pivot_vector_element);
			}
		}

		MPI_Gather(up_rows, rows_per_process * dim, MPI_FLOAT, update_rows, rows_per_process * dim, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);
		MPI_Gather(up_vector, rows_per_process, MPI_FLOAT, vector, rows_per_process, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);

		if(world_rank == RANK_ROOT) {
			int rindex = 0;

			for(j = 0; j < dim; j++) {
				if(j != i) {
					memcpy(row_pointers[j], &update_rows[rindex * dim], dim * sizeof(float));
					rindex++;
				}
			}
		}
	}

	if(world_rank == RANK_ROOT) {
		int thread_count;
		thread_count = (argc == 2) ? atoi(argv[1]) : 2;

		#pragma omp parallel for private(i, j) num_threads(thread_count)
		for(i = 0; i < dim; i++) {
			float multiplier = (1 / (*(row_pointers[i] + i)));
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
	/* ------- */

	/* Encerra o ambiente de execução MPI */
	MPI_Finalize();

	return 0;
	/* END */
}