#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include <time.h>
#include <omp.h>

#define ERROR_PARAM_COUNT	-1
#define ERROR_ROW_COL_COUNT	-2
#define ERROR_PROCESS_COUNT	-3
#define ERROR_MAIN_DIAGONAL	-4
#define RANK_ROOT			0

int main(int argc, char **argv) {
	int i, j;

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

	/* Verifica se os parâmetros foram passados corretamente */
	if(argc != 3) {
		printf("Você deve fornecer como parâmetro o número de linhas e colunas da matriz aumentada.\n"
		"Tente novamente...\n");
		MPI_Finalize();
		return ERROR_PARAM_COUNT;
	}

	int rows, cols;
	rows = atoi(argv[1]);
	cols = atoi(argv[2]);

	if(rows > cols) {
		printf("O número de linhas da matriz deve ser menor ou igual ao número de colunas.\n");
		MPI_Finalize();
		return ERROR_ROW_COL_COUNT;
	}

	/* ----------------------------------------------------- */

	/* O número de processos criados deve ser igual ao número de linhas da matriz menos 1 */
	if(world_size != (rows - 1)) {
		printf("O número de processos criados deve ser igual a: número de linhas da matriz - 1.\n");
		MPI_Finalize();
		return ERROR_PROCESS_COUNT;
	}


	/* Processo root aloca espaço na memória para guardar os elementos da matriz */
	float *matrix = NULL;

	if(world_rank == RANK_ROOT) {
		matrix = (float *) calloc(rows * cols, sizeof(float));
		for(i = 0; i < rows; i++) {
			for(j = 0; j < cols; j++) {
				scanf("%f", &matrix[i * cols + j]);
			}
		}
	}

	/* Verifica a ocorrência de elementos nulos na diagonal principal */
	if(world_rank == RANK_ROOT) {
		for(i = 0; i < rows; i++) {
			if(matrix[i * cols + i] == 0) {
				printf("A matriz não pode conter elementos nulos na diagonal principal.\n");
				MPI_Finalize();
				return ERROR_MAIN_DIAGONAL;
			}
		}
	}
	

	/* Aloca espaço para guardar a linha do pivô e a linha que será atualizada */
	float *pivot_row = (float *) calloc(cols, sizeof(float));
	float *up_row = (float *) calloc(cols, sizeof(float));
	float *update_rows = NULL;

	/* Para cada linha da matriz seleciona o pivô (elemento da pertencente à diagonal principal) e atualiza os valores das outras linhas */
	for(i = 0; i < rows; i++) {

		/* Processo root aloca espaço para o buffer contendo as linhas que serão atualizadas na iteração do algoritmos (update_rows)*/
		if(world_rank == RANK_ROOT) {
			if(!update_rows)
				update_rows = (float *) calloc((rows - 1) * cols, sizeof(float));

			int rindex = 0;

			for(j = 0; j < rows; j++) {
				if(j != i) {
					memcpy(&update_rows[rindex * cols], &matrix[j * cols], cols * sizeof(float));
					rindex++;
				}
			}

			memcpy(pivot_row, &matrix[i * cols], cols * sizeof(float));
		}

		/* Processo root envia a linha contendo o elemento pivô da iteração e as respectivas linhas que serão atualizadas para
		todos os processos */
		MPI_Bcast(pivot_row, cols, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);
		MPI_Scatter(update_rows, cols, MPI_FLOAT, up_row, cols, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);

		float *result_row = (float *) calloc(cols, sizeof(float));

		/* Cada processo realiza a operação de eliminação gerando a sua respectiva linha atualizada (result_row) */
		for(j = 0; j < cols; j++) {
			result_row[j] = (pivot_row[i] * up_row[j]) - (up_row[i] * pivot_row[j]);
		}

		/* Processo root recebe o resultado da operação de eliminação de todos os processos em um buffer(update_rows)
		e atualiza a matriz para começar a próxima iteração do algoritmo */
		MPI_Gather(result_row, cols, MPI_FLOAT, update_rows, cols, MPI_FLOAT, RANK_ROOT, MPI_COMM_WORLD);

		if(world_rank == RANK_ROOT) {
			int rindex = 0;

			for(j = 0; j < rows; j++) {
				if(j != i) {
					memcpy(&matrix[j * cols], &update_rows[rindex * cols], cols * sizeof(float));
					rindex++;
				}
			}
		}
	}

	/* Multiplica cada linha da matriz pelo inverso do pivô da respectiva linha para se obter uma
	matriz triangular superiormente/inferiormente */
	if(world_rank == RANK_ROOT) {
		#pragma omp parallel private(i) num_threads(rows)
		{
			int th_id = omp_get_thread_num();
			float multiplier = (1 / matrix[th_id * cols + th_id]);

			for(i = 0; i < cols; i++)
				matrix[th_id * cols + i] *= multiplier;
		}

		/* Apresenta a matriz resultante na tela para o usuário */
		for(i = 0; i < rows; i++) {
			for(j = 0; j < cols; j++) {
				printf("%.2f ", matrix[i * cols + j]);
			}
			printf("\n");
		}
	}

	/* Cleanup */
	free(update_rows);
	free(up_row);
	free(pivot_row);
	free(matrix);
	/* ------- */

	/* Encerra o ambiente de execução MPI */
	MPI_Finalize();

	return 0;
	/* END */
}