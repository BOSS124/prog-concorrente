#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#include <mpi.h>
#include <omp.h>

#define RANK_ROOT				0
#define MATRIX_MAX_DIMENSION	10000

/* Retorna a diferença de tempo entre begin e end em milisegundos */
double time_elapsed_milis(clock_t end, clock_t begin) {
	return ((double) (end - begin) / (double) (CLOCKS_PER_SEC / 1000));
}

int main(int argc, char **argv) {
	int i, j, k;
	clock_t begin, end;

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
	double **vector_pointers = NULL;
	double *vector = NULL;
	unsigned int dim;

	/* Processo root aloca espaço e recebe a matriz e o vetor como entrada */
	if(world_rank == RANK_ROOT) {
		begin = clock();

		/* Vetor onde a matriz fica salva na memória do processo root */
		matrix = (double *) malloc(MATRIX_MAX_DIMENSION * MATRIX_MAX_DIMENSION * sizeof(double));

		/* Leitura do arquivo da matriz */
		FILE *file = fopen("matriz.txt", "r");
		unsigned int read = 0;
		while(!feof(file))
			fscanf(file, "%lf", &matrix[read++]);

		fclose(file);

		/* dim = sqrt(números lidos) */
		dim = (unsigned int) sqrt((double) read);

		/* Vetor de ponteiros que apontam para as linhas da matriz, utilizado para recuperar o resultado do
		algoritmo na ordem correta no fim da execução */
		row_pointers = (double**) malloc(dim * sizeof(double *));
		for(i = 0; i < dim; i++) {
			row_pointers[i] = &matrix[i * dim];
		}

		/* Leitura do arquivo do vetor */
		file = fopen("vetor.txt", "r");
		vector = (double *) malloc(dim * sizeof(double));
		read = 0;
		while(!feof(file))
			fscanf(file, "%lf", &vector[read++]);

		fclose(file);

		/* Vetor de ponteiros que apontam para os elementos do vetor, utilizado para recuperar o resultado do 
		algoritmo na ordem correta no fim da execução */
		vector_pointers = (double **) malloc(dim * sizeof(double *));
		for(i = 0; i < dim; i++) {
			vector_pointers[i] = &vector[i];
		}

		/* Se um elemento da diagonal principal for igual à 0, é feita a troca de linhas */
		for(i = 0; i < dim; i++) {
			if(matrix[i * dim + i] == 0) { //matriz[i][i] == 0 ?
				for(j = 0; j < dim; j++) { //itera entre todas as outras linhas para encontrar uma possível candidata a troca
					if((j != i) && (matrix[j * dim + i] != 0) && (matrix[i * dim + j] != 0)) {
						/* Troca linha i e j */
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
	}

	/* Todos processos recebem a dimensão da matriz na variável dim */
	MPI_Bcast(&dim, 1, MPI_UNSIGNED, RANK_ROOT, MPI_COMM_WORLD);

	/* Ponteiros para os buffers utilizados nas iterações do algoritmo */
	double *pivot_row = (double *) malloc(dim * sizeof(double));
	double *up_rows = NULL;
	double *up_vector = NULL;
	double *update_rows = NULL;
	double *update_vector = NULL;
	double pivot_vector_element;
	double *result_buffer;
	double *result_vector;
	/* --------------------------------------------------------------- */

	/* Vetores utilizados pelo MPI_Scatterv e MPI_Gatherv para dividir as linhas da matriz entre os
	processos */
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

	/* up_rows: buffer utilizado para receber as linhas da matriz a serem atualizadas pelo processo através do
	MPI_Scatterv */
	up_rows = (double *) malloc(scounts[world_rank] * sizeof(double));
	/* result_buffer: buffer onde as linhas atualizadas serão salvas para serem enviadas de volta para o processo root
	através do MPI_Gatherv */
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

	/* up_vector: buffer utilizado para receber os elementos do vetor a serem atualizados pelo processo através do
	MPI_Scatterv */
	up_vector = (double *) malloc(scounts2[world_rank] * sizeof(double));
	/* result_vector: buffer onde os elementos do vetor atualizados serão salvos para serem enviados de volta para o
	processo root através do MPI_Gatherv */
	result_vector = (double *) malloc(scounts2[world_rank] * sizeof(double));

	/* Itera entre todas as linhas da matriz sequencialmente, devido a dependência de dados entre uma iteração e outra,
	realizando a operação de eliminação Gaussiana */
	for(i = 0; i < dim; i++) {
		/* Processo root separa as linhas não-pivô em um buffer separado(update_rows e update_vector) para serem distribuídas entre os
		processos pelo MPI_Gatherv */
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

			/* A linha pivô da iteração fica num buffer separado para enviar para todos os processos, uma vez
			que todos precisarão dela para efetuar o processo de eliminação */
			memcpy(pivot_row, &matrix[i * dim], dim * sizeof(double));
			pivot_vector_element = vector[i];
		}

		/* Todos os processos recebem a linha pivô da iteração no buffer pivot_row */
		MPI_Bcast(pivot_row, dim, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

		/* Todos os processos recebem o elemento do vetor referente à linha pivô na variável pivot_vector_element */
		MPI_Bcast(&pivot_vector_element, 1, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

		/* Cada processo recebe uma quantia de linhas da matriz e elementos do vetor para realizar o processo de eliminação */
		MPI_Scatterv(update_rows, scounts, displs, MPI_DOUBLE, up_rows, scounts[world_rank], MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
		MPI_Scatterv(update_vector, scounts2, displs2, MPI_DOUBLE, up_vector, scounts2[world_rank], MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

		/* Eliminação: L[i][j] = (L[k][k] * L[i][j]) - (L[i][k] * L[k][j])
		i = index da linha sendo atualizada
		j = index do elemento da linha sendo atualizada
		k = index da linha pivô

		OBS: se L[i][k] for igual a 0 não há necessidade de alterar a linha, portanto apenas copia os valores para os
		buffers de resultado */
		for(j = 0; j < scounts2[world_rank]; j++) {
			if(up_rows[j * dim + i] != 0.0f) {
				for(k = 0; k < dim; k++) {
					result_buffer[j * dim + k] = (pivot_row[i] * up_rows[j * dim + k]) - (up_rows[j * dim + i] * pivot_row[k]);
					result_vector[j] = (pivot_row[i] * up_vector[j]) - (up_rows[j * dim + i] * pivot_vector_element);
				}
			}
			else {
				for(k = 0; k < dim; k++) {
					result_buffer[j * dim + k] = up_rows[j * dim + k];
					result_vector[j] = up_vector[j];
				}
			}
		}

		/* O processo root recebe de volta as linhas da matriz e elementos do vetor atualizados pelos processos */
		MPI_Gatherv(result_buffer, scounts[world_rank], MPI_DOUBLE, update_rows, scounts, displs, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);
		MPI_Gatherv(result_vector, scounts2[world_rank], MPI_DOUBLE, update_vector, scounts2, displs2, MPI_DOUBLE, RANK_ROOT, MPI_COMM_WORLD);

		/* Processo root remonta o buffer da matriz(matrix) com os dados atualizados que acaba de receber pelo MPI_Gatherv */
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

	/* Nesse ponto a matriz reduzida deve possuir apenas os elementos da diagonal principal diferentes de 0 e portanto realiza-se 
	a última etapa do algoritmo onde 
		L[i][j] = L[i][j] * (1 / L[i][i])
	ou seja, cada elemento de cada linha é multiplicado pelo inverso do elemento da diagonal da própria linha,
	dessa maneira não se altera o determinante da matriz e a matriz reduzida se torna a matriz identidade.
	Ao final os elementos do vetor(vector) apresentam a solução do processo de escalonamento */
	if(world_rank == RANK_ROOT) {
		#pragma omp parallel for private(i, j) shared(matrix) num_threads(thread_count)
		for(j = 0; j < dim; j++) {
			double multiplier = (1.0f / matrix[j * dim + j]); //multiplicador = inverso do elemento da diagonal principal
			vector[j] *= multiplier;
		}

		/* Cria-se ou abre para escrita um arquivo resultado.txt onde os elementos de vector serão salvos com um valor por
		linha do arquivo */
		FILE *file = fopen("resultado.txt", "w");
		for(i = 0; i < dim; i++) {
			fprintf(file, "%.3lf\n", *(vector_pointers[i])); //vector_pointers usado pois guarda a sequência original do vetor, mesmo após troca de linhas
		}
		fclose(file);

		end = clock();

		/* Num_process = número de processos que executaram o algoritmo de escalonamento de Gauss-Jordan
		Threads_per_process = número de threads executando por processo
		Matrix_dimension = dimensão da matriz utilizada pelo programa
		Time(ms) = tempo em milisegundo para execução do programa */
		printf("Num_process: %d\tThreads_per_process: %d\tMatrix_dimension: %d\tTime(ms): %lf\n", world_size, thread_count, dim, time_elapsed_milis(end, begin));
	}
	
	/* Liberação de memória */
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