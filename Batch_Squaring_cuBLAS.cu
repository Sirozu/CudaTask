#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include "math.h"
#include "cublas_v2.h" 
#include <ctime>
#pragma comment (lib, "cublas.lib")

#define ENTER_COUNT_MATRIX 196608//1966080 // Количество матриц
#define ENTER_DIM_MATRIX 4   // 11 > n > 1 // функциональная возможность вводить размерность

using namespace std;

void MMCPU(float* inMat, float* outMat, int countMat)
{
    int size = ENTER_DIM_MATRIX * ENTER_DIM_MATRIX;
    for (int addit = 0; addit < countMat; addit++)
    {

        for (int i = 0; i < ENTER_DIM_MATRIX; i++)
        {
            for (int j = 0; j < ENTER_DIM_MATRIX; j++)
            {
                float sum = 0;
                for (int k = 0; k < ENTER_DIM_MATRIX; k++)
                {
                    sum += inMat[addit * size + i * ENTER_DIM_MATRIX + k]
                        * inMat[addit * size + k * ENTER_DIM_MATRIX + j];
                }
                outMat[addit * size + i * ENTER_DIM_MATRIX + j] = sum;
            }
        }
    }

    //// print////////////////////////////////////////
    //for (int i = 0; i < countMat * size; i++) {
    //    if (i % ENTER_DIM_MATRIX == 0) {
    //        printf("\n");
    //    }

    //    if (i % size == 0) {
    //        printf("\n");
    //    }
    //    std::cout << outMat[i] << " ";
    //}
    //std::cout << "\n";
    //////////////////////////////////////////////////

}


int random(int a, int b)
{
    return a + rand() % (b - a);
}

int main(int argc, char** argv) {
    setlocale(LC_ALL, "Russian");
    srand(time(NULL));

    int countMatrix;
    int sizeMatrix;
    if (ENTER_COUNT_MATRIX != 0)
    {
        countMatrix = ENTER_COUNT_MATRIX;
    }
    else
    {
        countMatrix = 2;
    }


    if (ENTER_DIM_MATRIX != 0)
    {
        sizeMatrix = ENTER_DIM_MATRIX;
    }
    else
    {
        sizeMatrix = random(2, 10);
    }
    cout << std::endl;
    std::cout << "Количество матриц = " << countMatrix << "\n";
    std::cout << "Размерность матриц = " << sizeMatrix << "х" << sizeMatrix << "\n";
    std::cout << "Количество занимаемой памяти = "
        << (sizeMatrix * sizeMatrix * countMatrix * sizeof(float)) / 1048576. << " Mb" << "\n";

    int countElement = sizeMatrix * sizeMatrix * countMatrix;
    float* arrMatrix = new float[countElement];

    for (int i = 0; i < countElement; i++)
    {
        arrMatrix[i] = random(1, 10);
    }

    //int numBytes = countElement * sizeof ( float );
    float* a = arrMatrix;
    float* c = new float[countElement];

    for (int i = 0; i < countElement; i++)
    {
        c[i] = 0;
    }

    int N = sizeMatrix;

    //float *A_Device ,* B_Device,
    //float *C_Device;
    float* d_A, * d_C;

    // A_Device = (float*) malloc ( N * N * sizeof ( float ) );
    // B_Device = (float*) malloc ( N * N * sizeof ( float ) );
     //C_Device = (float*) malloc ( N * N * sizeof ( float ) );


    float stopGPU, sumCalcGPUTime = 0;

    //cublasInit();


    cudaMalloc((void**)&d_A, countElement * sizeof(float));
    cudaMalloc((void**)&d_C, countElement * sizeof(float));
    cudaMemcpy(d_A, a, countElement * sizeof(float), cudaMemcpyHostToDevice);
    // cublasSetMatrix ( N, N, sizeof(float), (void *) A_Device, N, (void *) d_A, N);
    // cublasSetMatrix ( N, N, sizeof(float), (void *) C_Device, N, (void *) d_C, N);


    // for (int k = 0; k < countMatrix; k++)
    // {
    //   float* A_Device = a + k * (N * N);
    //   float* C_Device = c + k * (N * N);

    cublasHandle_t handle;
    cublasCreate(&handle);

    cublasStatus_t status;

    float time_cuda_event;
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    clock_t startGPU;
    float elapsedTimeGPU;
    startGPU = clock();
    
    //cublasSgemmBatched(handle , transa, transb, N, N, N, 1.0f, d_A, N, d_A, N, 1.0f, d_C, N, (N*N));
    //status = cublasSgemmStridedBatched(handle, transa, transb, N, N, N, 1.0f, d_A, N, (N*N), d_A, N, (N*N), 1.0f, d_C, N, (N*N), countMatrix);
    float alpha = 1.0f;  float beta = 1.0f;
    cublasSgemmStridedBatched(handle,
        CUBLAS_OP_N,
        CUBLAS_OP_N,
        N, N, N,
        &alpha,
        (const float*)d_A, N,
        N * N,
        (const float*)d_A, N,
        N * N,
        &beta,
        d_C, N,
        (N * N),
        countMatrix);

    elapsedTimeGPU = (double)(clock() - startGPU) / CLOCKS_PER_SEC * 1000;

    (cudaEventRecord(stop, 0));
    (cudaEventSynchronize(stop));
    (cudaEventElapsedTime(&time_cuda_event, start, stop));
   /* printf("Time :  %3.1f ms \n", time_cuda_event);*/
    elapsedTimeGPU = time_cuda_event;
    //cublasSgemm ( 'n', 'n', N, N, N, 1.0f, d_A + k * (N * N), N, d_A + k * (N * N), N, 1.0f, d_C, N );
    cudaMemcpy(c, d_C, countElement * sizeof(float), cudaMemcpyDeviceToHost);

    cublasDestroy(handle);

   /* std::cout << status << "\n";*/
    //}

    //cudaMemcpy ( c, d_C, countElement* sizeof(float), cudaMemcpyDeviceToHost );

    //for (int i = 0; i < countElement; i++)
    //{
    //    std::cout << c[i] << " ";
    //}

    cudaFree(d_A);
    cudaFree(d_C);
    /* cublasShutdown();*/
    cout << std::endl;
    std::cout << "--------------------------------\n";
    cout << "GPU MM time = " << elapsedTimeGPU << " ms\n";
    cout << "CUDA memory throughput = " << countElement * sizeof(float) / elapsedTimeGPU / 1024 / 1024 / 1.024 << " Gb/s\n";
    // printf("Elapsed time = %.4f\n",time_seconds);
    // std::cout << C[0] << " "<< C[1] << " "<< C[2] << " "<< C[3] << "\n";

    std::cout << "--------------------------------\n";
    float* checkMass = new float[countElement];

    clock_t startCPU;
    float elapsedTimeCPU;
    startCPU = clock();

    MMCPU(arrMatrix, checkMass, countMatrix);

    elapsedTimeCPU = (double)(clock() - startCPU) / CLOCKS_PER_SEC * 1000;
    cout << "CPU MM time = " << elapsedTimeCPU << " ms\n";
    cout << "CPU memory throughput = " << countElement * sizeof(float) / elapsedTimeCPU / 1024 / 1024 / 1.024 << " Gb/s\n";
    std::cout << "--------------------------------\n";
    std::cout << "\n";

    bool flageq = true;
    for (int i = 0; i < countElement; i++)
    {
        //std::cout << c[i]  << " " << checkMass[i] << "\n";
        if (c[i] != checkMass[i])
        {
            flageq = false;
            break;
        }
    }

    std::cout << "Проверка " << (flageq ?
        "пройдена, матрицы равны" :
        "не пройдена, матрицы не равны") << "\n";

    std::cout << "GPU быстрее CPU в " << elapsedTimeCPU / elapsedTimeGPU << " раз\n";
    cout << std::endl;

    delete a;
    delete c;
    delete checkMass;

}
