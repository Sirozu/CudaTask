#include <iostream>
#include <stdio.h>
#include <time.h>

#define ENTER_COUNT_MATRIX 1966080 // Количество матриц
#define ENTER_DIM_MATRIX 4   // 11 > n > 1 // функциональная возможность вводить размерность

using namespace std;

#define CHECK(value) { \
cudaError_t _m_cudaStat = value; \
if (_m_cudaStat != cudaSuccess) { \
std::cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
<< " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
exit(1); \
} }

//1966080 * 4* 4* 4 = 125829120 / 18 гб/с = 7.6ms
//(2+24)×4×4×4=  = 64503316480 / 18гб/с = 26ms

__global__ void CudaMM(float *matrM, float *matrR, int N)
{
    if (((blockIdx.x * blockDim.x  + threadIdx.x) * (N * N)) >= ENTER_COUNT_MATRIX * ENTER_DIM_MATRIX * ENTER_DIM_MATRIX)
      return;

    float* a = matrM + (blockIdx.x * blockDim.x  + threadIdx.x) * (N * N);
    float* r = matrR + (blockIdx.x * blockDim.x  + threadIdx.x) * (N * N);

    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            float sum = 0;
            for (int k = 0; k < N; k++)
            {
                sum += a[i * N + k] * a[k * N + j];
            }
            r[i * N + j] = sum;
        }
    }

}


__global__ void matMult ( float *matrM, float *matrR, int N)
{
  if (((blockIdx.x * blockDim.x  + threadIdx.x) * (N * N)) >= ENTER_COUNT_MATRIX * ENTER_DIM_MATRIX * ENTER_DIM_MATRIX)
    return;

  extern __shared__ float a[];

  for (int i = 0; i < (N * N); i++)
  {
      int tmp = threadIdx.x + i * blockDim.x;
      a[tmp % (N*N) + tmp/(N*N)*(N*N + 1)] = matrM[(blockIdx.x * blockDim.x) * (N * N)  + threadIdx.x + i * blockDim.x];
      // добавить к первому индексу куда-то 1, а потом внизу добавить единичку к N*N
  }

  __syncthreads();

  for (int i = 0; i < N; i++)
  {
    for (int j = 0; j < N; j++)
    {
        float sum = 0;
        for (int k = 0; k < N; k++)
        {
          //threadIdx.x * (N*N) +
            sum += a[threadIdx.x * (N*N + 1)  + i * N + k]
                      * a[threadIdx.x * (N*N + 1)  + k * N + j];// что то еще с threadIdx.x  должно быть
        }
        //__syncthreads();
        matrR[(blockIdx.x * blockDim.x + threadIdx.x) * (N * N) + i * N + j] = sum;
    }
  }

}

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

    // print////////////////////////////////////////
    // for (int i = 0; i < countMat * size; i++) {
    //      if (i % ENTER_DIM_MATRIX == 0) {
    //          printf("\n");
    //      }
    //
    //      if (i % size == 0) {
    //          printf("\n");
    //      }
    //      std::cout << outMat[i] << " ";
    //  }
    //  std::cout << "\n";
    ////////////////////////////////////////////////

}


int random(int a, int b)
{
  return a + rand() % (b - a);
}

int main ( int argc, char * argv [] )
{
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
    << (sizeMatrix * sizeMatrix * countMatrix * sizeof ( float ))/1048576. << " Mb" << "\n";

  int countElement = sizeMatrix * sizeMatrix * countMatrix;
  float *arrMatrix = new float[countElement];

  for (int i = 0; i < countElement; i++)
  {
    arrMatrix[i] = random(1, 10);
  }

  int numBytes = countElement * sizeof ( float );
  float * a = arrMatrix;
  float * c = new float [countElement];
  //int N = sizeMatrix;

//print /////////////////////////////////////////////////////////////////////////
    // for (int i = 0; i < countElement; i++)
    // {
    //   if (i % (sizeMatrix * sizeMatrix) == 0)
    //     std::cout << "\n";
    //   std::cout << arrMatrix[i] << " ";
    // }
/////////////////////////////////////////////////////////////////////////////////

  int thread = (countMatrix > 256? 256: countMatrix);

  // allocate device memory
  float * adev = NULL;
  float * cdev = NULL;

  CHECK(cudaMalloc ( (void**)&adev, numBytes ));
  CHECK(cudaMalloc ( (void**)&cdev, numBytes ));

  // create cuda event handles
  cudaEvent_t start, stop;
  float gpuTime = 0.0f;

  cudaEventCreate ( &start );
  cudaEventCreate ( &stop );

  CHECK(cudaMemcpy ( adev, a, numBytes, cudaMemcpyHostToDevice ));
  cudaEventRecord ( start, 0 );

  //CudaMM<<<(countMatrix + 511)/512, 512>>> ( adev, cdev, sizeMatrix);

  matMult<<<(countMatrix + 255)/256, thread, (thread * sizeMatrix * sizeMatrix * 4) + thread * 4>>> (adev, cdev, sizeMatrix);

  cudaEventRecord(stop,0);
  cudaEventSynchronize(stop);
  CHECK(cudaGetLastError());

  CHECK(cudaMemcpy ( c, cdev, numBytes, cudaMemcpyDeviceToHost ));

  cudaEventElapsedTime ( &gpuTime, start, stop );

  // print the cpu and gpu times
  cout << std::endl;
  std::cout << "--------------------------------\n";
  cout << "GPU MM time = " << gpuTime << " ms\n";
  cout << "CUDA memory throughput = " << countElement*sizeof(float)/gpuTime/1024/1024/1.024 << " Gb/s\n";

  // release resources
  cudaEventDestroy ( start );
  cudaEventDestroy ( stop );
  CHECK(cudaFree ( adev ));
  CHECK(cudaFree ( cdev ));

  // print////////////////////////////////////////
  //  for (int i = 0; i < countElement; i++)
  //  {
  //    if (i % (sizeMatrix * sizeMatrix) == 0)
  //      std::cout << "\n\n";
  //    if (i % (sizeMatrix) == 0)
  //      std::cout << "\n";
   //
  //    std::cout << c[i] << " ";
  //  }
  //  std::cout << "\n";
  ////////////////////////////////////////


  std::cout << "--------------------------------\n";
  float* checkMass = new float[countElement];

  clock_t startCPU;
  float elapsedTimeCPU;
  startCPU = clock();

  MMCPU(arrMatrix, checkMass, countMatrix);

  elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC * 1000;
  cout << "CPU MM time = " << elapsedTimeCPU << " ms\n";
  cout << "CPU memory throughput = " << countElement*sizeof(float)/elapsedTimeCPU/1024/1024/1.024 << " Gb/s\n";
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

  std::cout << "GPU быстрее CPU в " << elapsedTimeCPU / gpuTime  << " раз\n";
  cout << std::endl;

  delete a;
  delete c;
  delete checkMass;

  return 0;
}
