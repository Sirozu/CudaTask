#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#include <CL/cl2.hpp>

#include <iostream>
#include <time.h>
#include <string.h>

#define ENTER_COUNT_MATRIX 1966080 // Количество матриц
#define ENTER_DIM_MATRIX 4   // 11 > n > 1 // функциональная возможность вводить размерность

using namespace cl;
using namespace std;

#define checkError(func) \
  if (errcode != CL_SUCCESS)\
  {\
    cout << "Error in " #func "\nError code = " << errcode << "\n";\
    exit(1);\
  }

#define checkErrorEx(command) \
  command; \
  checkError(command);

int random(int a, int b)
{
  return a + rand() % (b - a);
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

int main( int argc, char * argv [] )
{
  int device_index = 0;
  cl_int errcode;

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
  float* checkMass = new float[countElement];
  int thread = (countMatrix > 256? 256: countMatrix);

  //print /////////////////////////////////////////////////////////////////////////
      // for (int i = 0; i < countElement; i++)
      // {
      //   if (i % (sizeMatrix * sizeMatrix) == 0)
      //     std::cout << "\n";
      //   std::cout << arrMatrix[i] << " ";
      // }
  /////////////////////////////////////////////////////////////////////////////////


  clock_t startCPU = clock();
//#pragma omp parallel for
  MMCPU(arrMatrix, checkMass, countMatrix);
  double elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;

  //код kernel-функции
  string sourceString = "\n\
  __kernel void CudaMM(__global float *matrM, __global float *matrR, int N, int CountMat)\n\
  {\n\
      if ((get_global_id(0) * (N * N)) >= CountMat * N * N)\n\
        return;\n\
      int add = (get_global_id(0)) * (N * N);\n\
      for (int i = 0; i < N; i++)\n\
      {\n\
          for (int j = 0; j < N; j++)\n\
          {\n\
              float sum = 0;\n\
              for (int k = 0; k < N; k++)\n\
              {\n\
                  sum += matrM[add + i * N + k] * matrM[add + k * N + j];\n\
              }\n\
              matrR[add + i * N + j] = sum;\n\
          }\n\
      }\n\
  }";

  string sourceString1 = "\n\
  __kernel void CudaMM(__global float *matrM, __global float *matrR, int N, int CountMat)\n\
  {\n\
    if ((get_global_id(0) * (N * N)) >= CountMat * N * N)\n\
      return;\n\
    __local float a[4352];\n\
    for (int i = 0; i < (N * N); i++)\n\
    {\n\
        int tmp = get_local_id(0) + i * get_local_size(0);\n\
        a[tmp % (N*N) + tmp/(N*N)*(N*N + 1)] = matrM[(get_group_id(0) * get_local_size(0)) * (N * N)  + get_local_id(0) + i * get_local_size(0)];\n\
    }\n\
    barrier(CLK_LOCAL_MEM_FENCE);\n\
    for (int i = 0; i < N; i++)\n\
    {\n\
      for (int j = 0; j < N; j++)\n\
      {\n\
          float sum = 0;\n\
          for (int k = 0; k < N; k++)\n\
          {\n\
              sum += a[get_local_id(0) * (N*N + 1)  + i * N + k] * a[get_local_id(0) * (N*N + 1)  + k * N + j];\n\
          }\n\
          matrR[(get_global_id(0)) * (N * N) + i * N + j] = sum;\n\
      }\n\
    }\n\
  }";

  //получаем список доступных OpenCL-платформ (драйверов OpenCL)
  std::vector<Platform> platform;//массив в который будут записываться идентификаторы платформ
  checkErrorEx( errcode = Platform::get(&platform) );
  cout << "OpenCL platforms found: " << platform.size() << "\n";
  cout << "Platform[0] is : " << platform[0].getInfo<CL_PLATFORM_VENDOR>() << " ver. " << platform[0].getInfo<CL_PLATFORM_VERSION>() << "\n";

  //в полученном списке платформ находим устройство GPU (видеокарту)
  std::vector<Device> devices;
  platform[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
  cout << "GPGPU devices found: " << devices.size() << "\n";
  if (devices.size() == 0)
  {
      cout << "Warning: YOU DON'T HAVE GPGPU. Then CPU will be used instead.\n";
      checkErrorEx( errcode = platform[0].getDevices(CL_DEVICE_TYPE_CPU, &devices) );
      cout << "CPU devices found: " << devices.size() << "\n";
      if (devices.size() == 0) {cout << "Error: CPU devices not found\n"; exit(-1);}
  }
  cout << "Use device N " << device_index << ": " << devices[device_index].getInfo<CL_DEVICE_NAME>() << "\n";

  //создаем контекст на видеокарте
  checkErrorEx( Context context(devices, NULL, NULL, NULL, &errcode) );

  //создаем очередь задач для контекста
  checkErrorEx( CommandQueue queue(context, devices[device_index], CL_QUEUE_PROFILING_ENABLE, &errcode) );// третий параметр - свойства

  //создаем обьект-программу с заданным текстом программы
  //checkErrorEx( Program program = Program(context, sourceString, false/*build*/, &errcode) );///!!!!!!!!1
  checkErrorEx( Program program = Program(context, sourceString1, false/*build*/, &errcode) );

  //компилируем и линкуем программу для видеокарты
  errcode = program.build(devices, "-cl-fast-relaxed-math -cl-no-signed-zeros -cl-mad-enable");
  if (errcode != CL_SUCCESS)
  {
      cout << "There were error during build kernel code. Please, check program code. Errcode = " << errcode << "\n";
      cout << "BUILD LOG: " << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(devices[device_index]) << "\n";
      return 1;
  }
  //создаем буфферы в видеопамяти
  checkErrorEx( Buffer dev_a = Buffer( context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, numBytes,  a, &errcode ) );
  checkErrorEx( Buffer dev_c = Buffer( context, CL_MEM_READ_WRITE, numBytes,  NULL, &errcode ) );

  //создаем объект - точку входа GPU-программы
  auto sum = KernelFunctor<Buffer, Buffer, int, int>(program, "CudaMM");

  //создаем объект, соответствующий определенной конфигурации запуска kernel
  //EnqueueArgs enqueueArgs(queue, cl::NDRange(numBytes)/*globalSize*/, NullRange/*blockSize*/);///!!!!!!!!1
  EnqueueArgs enqueueArgs(queue, cl::NDRange(numBytes)/*globalSize*/, 256/*blockSize*/);

  //запускаем и ждем
  clock_t t0 = clock();
  Event event = sum(enqueueArgs, dev_a, dev_c, sizeMatrix, countMatrix);
  checkErrorEx( errcode = event.wait() );
  clock_t t1 = clock();

  //считаем время
  cl_ulong time_start, time_end;
  errcode = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_START, &time_start);
  errcode = event.getProfilingInfo<cl_ulong>(CL_PROFILING_COMMAND_END, &time_end);
  double elapsedTimeGPU;
  if (errcode == CL_PROFILING_INFO_NOT_AVAILABLE)
    elapsedTimeGPU = (double)(t1-t0)/CLOCKS_PER_SEC;
  else
  {
    checkError(event.getEventProfilingInfo);
    elapsedTimeGPU = (double)(time_end - time_start)/1e9;
  }

  checkErrorEx( errcode = queue.enqueueReadBuffer(dev_c, true, 0, numBytes, c, NULL, NULL) );

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

  std::cout << "GPU быстрее CPU в " << (elapsedTimeCPU / elapsedTimeGPU)  << " раз\n";
  cout << std::endl;

  cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
  cout << "CPU memory throughput = " << countElement*sizeof(float)/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";
  cout << "GPU sum time = " << elapsedTimeGPU*1000 << " ms\n";
  cout << "GPU memory throughput = " << countElement*sizeof(float)/elapsedTimeGPU/1024/1024/1024 << " Gb/s\n";
  return 0;
}
