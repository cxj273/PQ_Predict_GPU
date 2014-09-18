#include <stdio.h>
#include <cassert>
#include <cuda_runtime.h>
#include "cuPrintf.cu"

#define MAX_LINE_BUFFER 10240
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

#define BLOCK_SIZE (512)
#define FEATURE_CHUNK_SIZE (100000000)

__device__ double sigmoid_predict(double decision_value, double A, double B)
{
    //double fApB = decision_value*A+-B;
    double fApB = decision_value * A + B; //original is this guy
    //fprintf(stderr, "%g %g %g %g\n", decision_value, A, B, fApB);
    if(fApB >= 0)
    {
        return exp(-fApB) / (1.0 + exp(-fApB));
    }
    else
    {
        return 1.0 / (1 + exp(fApB)) ;
    }
}

// Method 2 from the multiclass_prob paper by Wu, Lin, and Weng
__device__ static void multiclass_probability(double r[2][2], double* p)
{
    int t, j;
    const int k = 2;
    int iter = 0, max_iter = max(100, k);
    double Q[k][k];
    double Qp[k];
    double pQp, eps = 0.005 / k;

    for(t = 0; t < k; t++)
    {
        p[t] = 1.0 / k; // Valid if k = 1
        //Q[t]=Malloc(double,k);
        Q[t][t] = 0;
        for(j = 0; j < t; j++)
        {
            Q[t][t] += r[j][t] * r[j][t];
            Q[t][j] = Q[j][t];
        }
        for(j = t + 1; j < k; j++)
        {
            Q[t][t] += r[j][t] * r[j][t];
            Q[t][j] = -r[j][t] * r[t][j];
        }
    }
    for(iter = 0; iter < max_iter; iter++)
    {
        // stopping condition, recalculate QP,pQP for numerical accuracy
        pQp = 0;
        for(t = 0; t < k; t++)
        {
            Qp[t] = 0;
            for(j = 0; j < k; j++)
            {
                Qp[t] += Q[t][j] * p[j];
            }
            pQp += p[t] * Qp[t];
        }
        double max_error = 0;
        for(t = 0; t < k; t++)
        {
            double error = fabs(Qp[t] - pQp);
            if(error > max_error)
            {
                max_error = error;
            }
        }
        if(max_error < eps)
        {
            break;
        }

        for(t = 0; t < k; t++)
        {
            double diff = (-Qp[t] + pQp) / Q[t][t];
            p[t] += diff;
            pQp = (pQp + diff * (diff * Q[t][t] + 2 * Qp[t])) / (1 + diff) / (1 + diff);
            for(j = 0; j < k; j++)
            {
                Qp[j] = (Qp[j] + diff * Q[t][j]) / (1 + diff);
                p[j] /= (1 + diff);
            }
        }
    }
    //if (iter>=max_iter)
    //fprintf(stderr, "Exceeds max_iter in multiclass_prob\n");
    //for(t=0;t<k;t++) free(Q[t]);
    //free(Q);
    //free(Qp);

}

__global__ void k_predict(
    int start_pos,
    int files,
    float* dot_map,
    unsigned char* fs,
    double* label,
    double* s1,
    double* s2,
    int choices,
    int splits,
    float avg,
    float rho,
    int noprobA,
    float probA,
    float probB,
    int first_label
)
{
    int ff = blockIdx.x * blockDim.x + threadIdx.x + start_pos;
    if(ff - start_pos >= files)
    {
        return;
    }
    const int nr_class = 2;
    float sum = 0;
    for(int j = 0; j < splits; j++)
    {
        sum += *(dot_map + j * choices + * (fs + (size_t)(ff - start_pos) * (size_t)splits + j));
    }
    sum /= avg; //division for EK10 (average-normalized distance matrix)
    sum -= rho;
    if(noprobA == 1)    //no probA
    {
        label[ff] = 0;
        s1[ff] = sum;
        s2[ff] = -sum;
        //continue;
        return;
    }

    double min_prob = 1e-7;
    double pairwise_prob[nr_class][nr_class];
    int k = 0;
    for(int i = 0; i < nr_class; i++)
        for(int j = i + 1; j < nr_class; j++)
        {
            pairwise_prob[i][j] = min(max(sigmoid_predict(sum, probA, probB), min_prob), 1 - min_prob);
            pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
            k++;
        }
    //exit(-1);
    double prob_estimates[nr_class];
    multiclass_probability(pairwise_prob, prob_estimates);

    int prob_max_idx = 0;
    for(int i = 1; i < nr_class; i++)
    {
        if(prob_estimates[i] > prob_estimates[prob_max_idx])
        {
            prob_max_idx = i;
        }
    }

    double predict_label = 1 - first_label;
    if(prob_max_idx == 0)
    {
        predict_label = first_label;
    }

    label[ff] = predict_label;
    s1[ff] = prob_estimates[0];
    s2[ff] = prob_estimates[1];
    /*
    fprintf(predict_out, "%g", predict_label);
    for(int i = 0; i < nr_class; i++){
    	fprintf(predict_out, " %g", prob_estimates[i]);
    }
    fprintf(predict_out, "\n");
    */
}


int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        puts("Usage: PQ_predict mapped_data batch_config gpu_id");
        exit(-1);
    }

    clock_t tic, toc;
    cudaError_t cuda_error;
    int gpu_id;
    sscanf(argv[3], "%d", &gpu_id);
    //omp_set_num_threads(threads);
    if(cudaSetDevice(gpu_id) != cudaSuccess)
    {
        fprintf(stderr, "Failed to set gpu_id\n");
        return 1;
    }

    tic = clock();
    FILE* mapped = fopen(argv[1], "r");
    int files, splits;
    fread(&files, sizeof(int), 1, mapped);
    fread(&splits, sizeof(int), 1, mapped);
    fprintf(stderr, "reading mapped features, %d files, %d splits\n", files, splits);
    unsigned char* fs = (unsigned char*)malloc(sizeof(unsigned char) * (size_t)files * (size_t)splits);
    size_t fs_size = sizeof(unsigned char) * (size_t)files * (size_t)splits;
    fread(fs, sizeof(unsigned char), (size_t)files * (size_t)splits, mapped);
    fclose(mapped);
    toc = clock();
    fprintf(stderr, "read data: %8.4f seconds\n", (toc - tic) * 1. / CLOCKS_PER_SEC);

    unsigned char* d_fs;
    if(cudaMalloc(&d_fs, FEATURE_CHUNK_SIZE) != cudaSuccess)
    {
        fprintf(stderr, "Failed to cudaMalloc d_fs:%ld\n", FEATURE_CHUNK_SIZE);
        return 1;
    }

    char model_path[512];
    char predict_path[512];
    FILE* config_file = fopen(argv[2], "r");
    while(fscanf(config_file, "%s %s", model_path, predict_path) != EOF)
    {
        fprintf(stderr, "model_path: %s predict_path: %s\n", model_path, predict_path);
        fprintf(stderr, "reading model\n");
        FILE* modelfp = fopen(model_path, "r");
        assert(modelfp != NULL);
        char buf[1024];
        char buf2[1024];
        float rho = 0, probA = 0, probB = 0;
        int noprobA = 1;
        float avg = 1;
        int total_sv = 0;
        char label_string[1024] = "";
        while(fscanf(modelfp, "%[^\n]\n", buf) > 0)
        {
            strcpy(buf2, buf);
            char* tok = strtok(buf, " ");
            if(strcmp(tok, "w") == 0)
            {
                break;
            }
            else if(strcmp(tok, "avg") == 0)
            {
                tok = strtok(NULL, " ");
                sscanf(tok, "%f", &avg);
            }
            else if(strcmp(tok, "rho") == 0)
            {
                tok = strtok(NULL, " ");
                sscanf(tok, "%f", &rho);
            }
            else if(strcmp(tok, "probA") == 0)
            {
                tok = strtok(NULL, " ");
                sscanf(tok, "%f", &probA);
                noprobA = 0;
            }
            else if(strcmp(tok, "probB") == 0)
            {
                tok = strtok(NULL, " ");
                sscanf(tok, "%f", &probB);
            }
            else if(strcmp(tok, "total_sv") == 0)
            {
                tok = strtok(NULL, " ");
                sscanf(tok, "%d", &total_sv);
            }
            else if(strcmp(tok, "label") == 0)
            {
                strcpy(label_string, buf2);
            }
            else
            {
                fprintf(stderr, "skipping %s\n", buf);
            }
        };
        int tp_splits, choices;
        fscanf(modelfp, "%d %d", &tp_splits, &choices);
        //fprintf(stderr, "%d %d\n", tp_splits, splits);
        assert(tp_splits == splits);
        assert(choices == 256);
        char cl;
        fscanf(modelfp, "%c", &cl);
        assert(cl == '\n');
        float* dot_map = (float*)malloc(sizeof(float) * tp_splits * choices);
        fread(dot_map, sizeof(float), tp_splits * choices, modelfp);
        fclose(modelfp);
        float* d_dot_map;
        if(cudaMalloc(&d_dot_map, sizeof(float) * tp_splits * choices) != cudaSuccess)
        {
            fprintf(stderr, "Failed to cudaMalloc d_dot_map\n");
            return 1;
        }

        int first_label;
        sscanf(label_string, "label %d", &first_label);

        FILE* predict_out = fopen(predict_path, "w");
        fprintf(predict_out, "%s\n", label_string);
        //float sqrt_avg = sqrtf(avg);
        fprintf(stderr, "predicting\n");

        double* label = (double*) malloc(sizeof(double) * files);
        double* d_label;
        if(cudaMalloc(&d_label, sizeof(double) * files) != cudaSuccess)
        {
            fprintf(stderr, "failed to cudaMalloc d_label\n");
            return 1;
        }


        double* s1 = (double*) malloc(sizeof(double) * files);
        double* d_s1;
        if(cudaMalloc(&d_s1, sizeof(double) * files) != cudaSuccess)
        {
            fprintf(stderr, "failed to cudaMalloc d_s1\n");
            return 1;
        }

        double* s2 = (double*) malloc(sizeof(double) * files);
        double* d_s2;
        if(cudaMalloc(&d_s2, sizeof(double) * files) != cudaSuccess)
        {
            fprintf(stderr, "failed to cudaMalloc d_s2\n");
            return 1;
        }

        /*****Copy data to device*****/
        cudaMemcpy(d_dot_map, dot_map, sizeof(float) * tp_splits * choices, cudaMemcpyHostToDevice);

        const size_t feats_per_chunk = FEATURE_CHUNK_SIZE / ((size_t)splits * sizeof(unsigned char));
        const size_t total_feats_num = files;
        size_t start_feat_pos = 0;

        //int count = 1;
        //cudaPrintfInit();
        while(start_feat_pos < total_feats_num)
        {

            int feats_to_predict = 0;
            if(start_feat_pos + feats_per_chunk >= total_feats_num)
            {
                feats_to_predict = total_feats_num - start_feat_pos;
            }
            else
            {
                feats_to_predict = feats_per_chunk;
            }

            int blocks_num = feats_to_predict / BLOCK_SIZE;

            while(blocks_num * BLOCK_SIZE < feats_to_predict)
            {
                blocks_num ++;
            }

            fprintf(stderr, "start_feat_pos: %ld\n", start_feat_pos);
            fprintf(stderr, "feats_to_predict: %ld\n", feats_to_predict);

            /*****Copy data to device*****/
            cudaMemcpy(d_fs, fs + start_feat_pos * splits, sizeof(unsigned char) * (size_t) feats_to_predict * (size_t) splits, cudaMemcpyHostToDevice);
            cudaDeviceSynchronize();

            dim3 blocks(blocks_num, 1, 1);
            dim3 threads(BLOCK_SIZE, 1, 1);
            /*****Kernel call*****/
            k_predict <<< blocks, threads>>>(
                start_feat_pos,
                feats_to_predict,
                d_dot_map,
                d_fs,
                d_label,
                d_s1,
                d_s2,
                choices,
                splits,
                avg,
                rho,
                noprobA,
                probA,
                probB,
                first_label);
            cudaDeviceSynchronize();

            start_feat_pos += feats_per_chunk;

            /*
            if(count -- == 0)
            {
                break;
            }
            */

        }

        /*****Copy data back to host*****/
        cudaMemcpy(s1, d_s1, sizeof(double) * files, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaMemcpy(s2, d_s2, sizeof(double) * files, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
        cudaMemcpy(label, d_label, sizeof(double) * files, cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();

        fprintf(stderr, "prediction complete\n");
        for(int i = 0; i < files; i++)
        {
            fprintf(predict_out, "%g %g %g\n", label[i], s1[i], s2[i]);
        }
        fclose(predict_out);
        free(dot_map);
        free(label);
        free(s1);
        free(s2);
        cudaFree(d_dot_map);
        cudaDeviceSynchronize();
        cudaFree(d_label);
        cudaDeviceSynchronize();
        cudaFree(d_s1);
        cudaDeviceSynchronize();
        cudaFree(d_s2);
        cudaDeviceSynchronize();
    }

    free(fs);
    cudaFree(d_fs);
    fclose(config_file);
    return 0;
}
