// Full implementation for MPI version of main.c

#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <mpi.h>

#define FILE_TRAIN_IMAGE "train-images.idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels.idx1-ubyte"
#define FILE_TEST_IMAGE  "t10k-images.idx3-ubyte"
#define FILE_TEST_LABEL  "t10k-labels.idx1-ubyte"
#define LENET_FILE       "model.dat"
#define COUNT_TRAIN      60000
#define COUNT_TEST       10000

double wtime(void)
{
    double now_time;
    struct timeval etstart;
    if (gettimeofday(&etstart, NULL) == -1)
        perror("Error: calling gettimeofday() not successful.\n");

    now_time = ((etstart.tv_sec) * 1000 + etstart.tv_usec / 1000.0);
    return now_time;
}

int main(int argc, char *argv[])
{
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // Divide dataset among processes
    int local_train_count = COUNT_TRAIN / size;
    unsigned char(*local_data)[28][28] = malloc(local_train_count * sizeof(*local_data));
    unsigned char local_labels[local_train_count];

    unsigned char(*global_data)[28][28] = NULL;
    unsigned char *global_labels = NULL;

    if (rank == 0) {
        global_data = malloc(COUNT_TRAIN * sizeof(*global_data));
        global_labels = malloc(COUNT_TRAIN * sizeof(unsigned char));
        read_data(global_data, global_labels, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL);
    }

    MPI_Scatter(global_data, local_train_count * sizeof(*local_data), MPI_BYTE,
                local_data, local_train_count * sizeof(*local_data), MPI_BYTE,
                0, MPI_COMM_WORLD);

    MPI_Scatter(global_labels, local_train_count, MPI_BYTE,
                local_labels, local_train_count, MPI_BYTE,
                0, MPI_COMM_WORLD);

    lenet_t model;
    if (rank == 0) init_lenet(&model);
    MPI_Bcast(&model, sizeof(lenet_t), MPI_BYTE, 0, MPI_COMM_WORLD);

    for (int epoch = 0; epoch < 10; epoch++) {
        train_lenet(&model, local_data, local_labels, local_train_count);
        printf("Process %d completed epoch %d\\n", rank, epoch);
    }

    lenet_t global_model;
    MPI_Reduce(&model, &global_model, sizeof(lenet_t), MPI_BYTE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Training complete. Saving model.\\n");
        save_lenet(&global_model, LENET_FILE);
    }

    free(local_data);
    free(global_data);
    MPI_Finalize();
    return 0;
}

mpi_main_c_full_path = '/mnt/data/cnnSeqNew_extracted/cnnSeqNew - Copy/main_mpi_full.c'

with open(mpi_main_c_full_path, 'w') as file:
    file.write(mpi_full_implementation)

mpi_main_c_full_path
