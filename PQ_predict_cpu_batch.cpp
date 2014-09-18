#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <omp.h>
#include <vector>
#include <string>
#include <cassert>
#include <cmath>
using namespace std;

#include <iostream>
#include <fstream>

#define MAX_LINE_BUFFER 10240
#define Malloc(type,n) (type *)malloc((n)*sizeof(type))

double sigmoid_predict(double decision_value, double A, double B)
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
static void multiclass_probability(int k, double** r, double* p)
{
    int t, j;
    int iter = 0, max_iter = max(100, k);
    double** Q = Malloc(double*, k);
    double* Qp = Malloc(double, k);
    double pQp, eps = 0.005 / k;

    for(t = 0; t < k; t++)
    {
        p[t] = 1.0 / k; // Valid if k = 1
        Q[t] = Malloc(double, k);
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
    if(iter >= max_iter)
    {
        fprintf(stderr, "Exceeds max_iter in multiclass_prob\n");
    }
    for(t = 0; t < k; t++)
    {
        free(Q[t]);
    }
    free(Q);
    free(Qp);
}

int main(int argc, char* argv[])
{
    if(argc != 4)
    {
        puts("Usage: PQ_predict mapped_data batch_config threads");
        exit(-1);
    }

    clock_t tic, toc;
    int threads;
    sscanf(argv[3], "%d", &threads);
    omp_set_num_threads(threads);

    tic = clock();
    /*****Read Mapped feature****/
    FILE* mapped = fopen(argv[1], "r");
    int files, splits;
    fread(&files, sizeof(int), 1, mapped);
    fread(&splits, sizeof(int), 1, mapped);
    fprintf(stderr, "reading mapped features, %d files, %d splits\n", files, splits);
    unsigned char* fs = (unsigned char*)malloc(sizeof(unsigned char) * (size_t)files * (size_t)splits);
    fread(fs, sizeof(unsigned char), (size_t)files * (size_t)splits, mapped);
    fclose(mapped);
    toc = clock();
    fprintf(stderr, "read data: %8.4f seconds\n", (toc - tic) * 1. / CLOCKS_PER_SEC);

    char model_path[512];
    char predict_path[512];
    FILE* config_file = fopen(argv[2], "r");

    while(fscanf(config_file, "%s %s", model_path, predict_path) != EOF)
    {
        fprintf(stderr, "model_path: %s predict_path: %s\n", model_path, predict_path);
        /*****Read model*****/
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
        }
        int tp_splits, choices;
        fscanf(modelfp, "%d %d", &tp_splits, &choices);
        assert(tp_splits == splits);
        assert(choices == 256);
        char cl;
        fscanf(modelfp, "%c", &cl);
        assert(cl == '\n');
        float* dot_map = (float*)malloc(sizeof(float) * tp_splits * choices);
        fread(dot_map, sizeof(float), tp_splits * choices, modelfp);
        fclose(modelfp);

        int first_label;
        sscanf(label_string, "label %d", &first_label);

        /*****Open predict file*****/
        FILE* predict_out = fopen(predict_path, "w");
        fprintf(predict_out, "%s\n", label_string);
        int nr_class = 2;
        //float sqrt_avg = sqrtf(avg);
        fprintf(stderr, "predicting\n");
        vector<double> label(files), s1(files), s2(files);
        #pragma omp parallel for schedule(dynamic)
        for(int ff = 0; ff < files; ff++)
        {
            float sum = 0;
            for(int j = 0; j < splits; j++)
            {
                sum += *(dot_map + j * choices + * (fs + (size_t)ff * (size_t)splits + j));
            }
            sum /= avg; //division for EK10 (average-normalized distance matrix)
            sum -= rho;
            if(noprobA == 1)    //no probA
            {
                label[ff] = 0;
                s1[ff] = sum;
                s2[ff] = -sum;
                continue;
            }

            double min_prob = 1e-7;
            double** pairwise_prob = Malloc(double*, nr_class);
            for(int i = 0; i < nr_class; i++)
            {
                pairwise_prob[i] = Malloc(double, nr_class);
            }
            int k = 0;
            for(int i = 0; i < nr_class; i++)
                for(int j = i + 1; j < nr_class; j++)
                {
                    pairwise_prob[i][j] = min(max(sigmoid_predict(sum, probA, probB), min_prob), 1 - min_prob);
                    pairwise_prob[j][i] = 1 - pairwise_prob[i][j];
                    k++;
                }
            //exit(-1);
            double* prob_estimates = (double*) malloc(nr_class * sizeof(double));
            multiclass_probability(nr_class, pairwise_prob, prob_estimates);

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
            for(int i = 0; i < nr_class; i++)
            {
                free(pairwise_prob[i]);
            }
            free(pairwise_prob);
            free(prob_estimates);
        }
        /*****Write out the predict result*****/
        fprintf(stderr, "prediction complete\n");
        for(int i = 0; i < files; i++)
        {
            fprintf(predict_out, "%g %g %g\n", label[i], s1[i], s2[i]);
        }
        fclose(predict_out);
        free(dot_map);
    }
    free(fs);
    fclose(config_file);
    return 0;
}
