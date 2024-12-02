#include "lenet.h"
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <memory.h>
#include <math.h>

void TrainBatchMPI(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize, int rank, int size) {
    double buffer[GETCOUNT(LeNet5)] = {0};
    double global_buffer[GETCOUNT(LeNet5)] = {0};
    int local_batch_size = batchSize / size;
    int start = rank * local_batch_size;
    int end = (rank == size - 1) ? batchSize : start + local_batch_size;

    for (int i = start; i < end; ++i) {
        Feature features = {0};
        Feature errors = {0};
        LeNet5 deltas = {0};
        load_input(&features, inputs[i]);
        forward(lenet, &features, relu);
        load_target(&features, &errors, labels[i]);
        backward(lenet, &deltas, &errors, &features, relugrad);

        // Accumulate deltas locally
        FOREACH(j, GETCOUNT(LeNet5)) {
            buffer[j] += ((double *)&deltas)[j];
        }
    }

    // Combine weight updates from all processes
    MPI_Reduce(buffer, global_buffer, GETCOUNT(LeNet5), MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        double k = ALPHA / batchSize;
        FOREACH(i, GETCOUNT(LeNet5)) {
            ((double *)lenet)[i] += k * global_buffer[i];
        }
    }

    MPI_Bcast(lenet, sizeof(LeNet5), MPI_BYTE, 0, MPI_COMM_WORLD);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    LeNet5 lenet;
    if (rank == 0) {
        Initial(&lenet);
    }

    // Broadcast the initial weights to all processes
    MPI_Bcast(&lenet, sizeof(LeNet5), MPI_BYTE, 0, MPI_COMM_WORLD);

    image *inputs;  // Assume inputs are allocated and initialized
    uint8 *labels;  // Assume labels are allocated and initialized
    int batchSize = 60000;

    TrainBatchMPI(&lenet, inputs, labels, batchSize, rank, size);

    if (rank == 0) {
        printf("Training complete.\n");
    }

    MPI_Finalize();
    return 0;
}
