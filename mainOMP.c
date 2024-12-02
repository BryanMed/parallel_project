// Full implementation for OpenMP version of main.c
#include "lenet.h"
#include <stdlib.h>
#include <stdio.h>
#include <omp.h>

#define FILE_TRAIN_IMAGE "train-images.idx3-ubyte"
#define FILE_TRAIN_LABEL "train-labels.idx1-ubyte"
#define FILE_TEST_IMAGE  "t10k-images.idx3-ubyte"
#define FILE_TEST_LABEL  "t10k-labels.idx1-ubyte"
#define LENET_FILE       "model.dat"
#define COUNT_TRAIN      60000
#define COUNT_TEST       10000

int main(int argc, char *argv[])
{
    printf("Initializing OpenMP...\n");

    lenet_t model;
    init_lenet(&model);

    unsigned char(*data)[28][28] = malloc(COUNT_TRAIN * sizeof(*data));
    unsigned char label[COUNT_TRAIN];
    read_data(data, label, COUNT_TRAIN, FILE_TRAIN_IMAGE, FILE_TRAIN_LABEL);

    int batch_size = 300;
    int num_batches = COUNT_TRAIN / batch_size;

    #pragma omp parallel for shared(model)
    for (int i = 0; i < num_batches; i++) {
        int start = i * batch_size;
        train_lenet(&model, data + start, label + start, batch_size);
        printf("Thread %d completed batch %d\\n", omp_get_thread_num(), i);
    }

    printf("Training complete. Saving model.\n");
    save_lenet(&model, LENET_FILE);

    free(data);
    return 0;
}


// Save the full OpenMP implementation
omp_main_c_full_path = '/mnt/data/cnnSeqNew_extracted/cnnSeqNew - Copy/main_omp_full.c'

with open(omp_main_c_full_path, 'w') as file:
    file.write(omp_full_implementation)

omp_main_c_full_path
