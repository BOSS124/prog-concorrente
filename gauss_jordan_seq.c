#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#define MATRIX_MAX_DIMENSION	10000

/* Retorna a diferença de tempo entre begin e end em milisegundos */
double time_elapsed_milis(clock_t end, clock_t begin) {
	return ((double) (end - begin) / (double) (CLOCKS_PER_SEC / 1000));
}

int main(int argc, char *argv[]) {
	int i, j, k;
	clock_t begin, end;

	begin = clock();

	/* Aloca espaço para guardar a matriz a ser escalonada */
	double *matrix = (double *) malloc(MATRIX_MAX_DIMENSION * MATRIX_MAX_DIMENSION * sizeof(double));
	unsigned int dim;

	/* Recebe a matriz como entrada pelo arquivo matriz.txt */
	FILE *file = fopen("matriz.txt", "r");
	unsigned int read = 0;
	while(!feof(file))
		fscanf(file, "%lf", &matrix[read++]); //lê a matriz do arquivo como double
	fclose(file);

	/* dimensão da matriz = raiz_quadrada(elementos lido do arquivo) */
	dim = (unsigned int) sqrt((double) read);

	/* Vetor de ponteiros que apontam para as linhas da matriz, utilizado para recuperar o resultado do
		algoritmo na ordem correta no fim da execução */
	double **row_pointers = (double**) malloc(dim * sizeof(double *));
	for(i = 0; i < dim; i++) {
		row_pointers[i] = &matrix[i * dim];
	}

	/* Leitura do arquivo do vetor (vetor.txt)*/
	file = fopen("vetor.txt", "r");
	double *vector = (double *) malloc(dim * sizeof(double)); //buffer onde fica salvo os elementos do vetor
	read = 0;
	while(!feof(file))
		fscanf(file, "%lf", &vector[read++]); //lê os elementos do arquivo como double
	fclose(file);

	/* Vetor de ponteiros que apontam para os elementos do vetor, utilizado para recuperar o resultado do 
		algoritmo na ordem correta no fim da execução */
	double **vector_pointers = (double **) malloc(dim * sizeof(double *));
	for(i = 0; i < dim; i++) {
		vector_pointers[i] = &vector[i];
	}

	/* Se um elemento da diagonal principal for igual à 0, é feita a troca de linhas */
	for(i = 0; i < dim; i++) {
		if(matrix[i * dim + i] == 0.0f) { //matriz[i][i] == 0 ?
			for(j = 0; j < dim; j++) {	//procura uma possível linha candidata a troca
				if((j != i) && (matrix[j * dim + i] != 0) && (matrix[i * dim + j] != 0)) {
					/* Efetua a troca da linha i pela j */
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

	/* Aloca um buffer temporário do tamanho de uma linha para guardar os valores da linha sendo atualizada no processo
	de eliminação Gaussiana */
	double *updated_row = (double *) malloc(dim * sizeof(double));

	/* Itera entre todas as linhas da matriz sequencialmente, realizando a operação de eliminação Gaussiana */
	for(i = 0; i < dim; i++) {
		/* Eliminação: L[i][j] = (L[k][k] * L[i][j]) - (L[i][k] * L[k][j])
		i = index da linha sendo atualizada
		j = index do elemento da linha sendo atualizada
		k = index da linha pivô

		OBS: se L[i][k] for igual a 0 não há necessidade de alterar a linha, portanto apenas copia os valores para os
		buffers de resultado */
		for(j = 0; j < dim; j++) {
			if(j != i && (matrix[j * dim + i] != 0.0f)) {
				for(k = 0; k < dim; k++) {
					updated_row[k] = (matrix[i * dim + i] * matrix[j * dim + k]) - (matrix[j * dim + i] * matrix[i * dim + k]); //atualiza o elemento da linha
				}
				vector[j] = (matrix[i * dim + i] * vector[j]) - (matrix[j * dim + i] * vector[i]); //atualiza o elemento do vetor
				memcpy(&matrix[j * dim], updated_row, dim * sizeof(double)); //copia os novos valores da linha atualizada para o buffer da matriz (matrix)
			}
		}
	}

	/* Libera a memória do buffer temporário */
	free(updated_row);

	/* Nesse ponto a matriz reduzida deve possuir apenas os elementos da diagonal principal diferentes de 0 e portanto realiza-se 
	a última etapa do algoritmo onde 
		L[i][j] = L[i][j] * (1 / L[i][i])
	ou seja, cada elemento de cada linha é multiplicado pelo inverso do elemento da diagonal da própria linha,
	dessa maneira não se altera o determinante da matriz e a matriz reduzida se torna a matriz identidade.
	Ao final os elementos do vetor(vector) apresentam a solução do processo de escalonamento */
	for(i = 0; i < dim; i++) {
		double multiplier = (1.0f / matrix[i * dim + i]);
		vector[i] *= multiplier;
	}

	/* Cria-se ou abre para escrita um arquivo resultado.txt onde os elementos de vector serão salvos com um valor por
		linha do arquivo */
	file = fopen("resultado.txt", "w");
	for(i = 0; i < dim; i++) {
		fprintf(file, "%.3lf\n", *(vector_pointers[i])); //vector_pointers usado pois guarda a sequência original do vetor, mesmo após troca de linhas
	}
	fclose(file);

	end = clock();

	/* Matrix_length = dimensão da matriz de entrada
	Sequential = tempo em milisegundo para a execução completa do algoritmo sequencial */
	printf("Matrix_length: %u\tSequential: %lfms\n", dim, time_elapsed_milis(end, begin));

	/* Liberação de memória */
	free(matrix);
	free(row_pointers);
	free(vector);
	free(vector_pointers);
	/* ------- */

	return 0;
	/* END */
}