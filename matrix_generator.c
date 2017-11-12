#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc, char *argv[]) {
	srand((unsigned)time(NULL));

	if(argc == 3) {
		int rows = atoi(argv[1]);
		int cols = atoi(argv[2]);
		int i, j;
		FILE *matriz = NULL, *vector = NULL;

		matriz = fopen("matriz.txt", "w");

		for(i = 0; i < rows; i++) {
			for(j = 0; j < cols; j++) {
				fprintf(matriz, "%.3lf ", (double) (rand() % 5 + 1) / 5);
			}
			fprintf(matriz, "\n");
		}

		fclose(matriz);

		vector = fopen("vetor.txt", "w");

		for(i = 0; i < rows; i++) {
			fprintf(vector, "%.3lf\n", (double) (rand() % 5 + 1) / 5);
		}

		fclose(vector);
	}

	return 0;
}