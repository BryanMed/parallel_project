#include "lenet.h"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

void TrainBatchOMP(LeNet5 *lenet, image *inputs, uint8 *labels, int batchSize) {
    double buffer[GETCOUNT(LeNet5)] = {0};

    #pragma omp parallel
    {
        double local_buffer[GETCOUNT(LeNet5)] = {0};

        #pragma omp for
        for (int i = 0; i < batchSize; ++i) {
            Feature features = {0};
            Feature errors = {0};
            LeNet5 deltas = {0};
            load_input(&features, inputs[i]);
            forward(lenet, &features, relu);
            load_target(&features, &errors, labels[i]);
            backward(lenet, &deltas, &errors, &features, relugrad);

            // Accumulate deltas locally
            FOREACH(j, GETCOUNT(LeNet5)) {
                local_buffer[j] += ((double *)&deltas)[j];
            }
        }

        #pragma omp critical
        {
            FOREACH(j, GETCOUNT(LeNet5)) {
                buffer[j] += local_buffer[j];
            }
        }
    }

    double k = ALPHA / batchSize;
    FOREACH(i, GETCOUNT(LeNet5)) {
        ((double *)lenet)[i] += k * buffer[i];
    }
}

int main() {
    LeNet5 lenet;
    Initial(&lenet);

    image *inputs;  // Assume inputs are allocated and initialized
    uint8 *labels;  // Assume labels are allocated and initialized
    int batchSize = 60000;

    TrainBatchOMP(&lenet, inputs, labels, batchSize);

    printf("Training complete.\n");
    return 0;
}
