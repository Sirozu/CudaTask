#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <iostream>

using namespace cv;
using namespace std;

#define CHECK(value) {                                          \
    cudaError_t _m_cudaStat = value;                                        \
    if (_m_cudaStat != cudaSuccess) {                                       \
        cout<< "Error:" << cudaGetErrorString(_m_cudaStat) \
            << " at line " << __LINE__ << " in file " << __FILE__ << "\n"; \
        exit(1);                                                            \
    } }

// размер массива ограничен максимальным размером пространства потоков
__global__ void removeColNum(uchar *c, int N, int colorNum)
{
    int d = 256/colorNum;
    int i = threadIdx.x + blockIdx.x*blockDim.x;
    if (i < N)
        c[i] = (c[i] / d)*d;
}

int main( int argc, char** argv )
{

  int N = 3*990*660;
  Mat image;
  image = imread("pic.jpg", CV_LOAD_IMAGE_COLOR);   // Read the file
  if(! image.data )                              // Check for invalid input
  {
      cout <<  "Could not open or find the image" << std::endl ;
      return -1;
  }

  cudaEvent_t startCUDA, stopCUDA;
  clock_t startCPU;
  float elapsedTimeCUDA, elapsedTimeCPU;

  cudaEventCreate(&startCUDA);
  cudaEventCreate(&stopCUDA);

  startCPU = clock();

//#pragma omp parallel for
int colorNum = 5;
int colorNum3 = colorNum * colorNum * colorNum;
int d = 256/colorNum;

for(int i = 0; i < image.rows; i++)
{
    //pointer to 1st pixel in row // 8 var
    //Уменьшить количество цветов изображения до n^3
    Vec3b* p = image.ptr<Vec3b>(i);
    for (int j = 0; j < image.cols; j++)
        for (int ch = 0; ch < 3; ch++)
          {
            p[j][ch] = (p[j][ch]/d)*d ;
          }
}


  elapsedTimeCPU = (double)(clock()-startCPU)/CLOCKS_PER_SEC;
  cout << "CPU sum time = " << elapsedTimeCPU*1000 << " ms\n";
  cout << "CPU memory throughput = " << 3*N*sizeof(float)/elapsedTimeCPU/1024/1024/1024 << " Gb/s\n";

  imwrite("picCPU.jpg",image);

    uchar* imageOnGPU;
    CHECK( cudaMalloc(&imageOnGPU, N ));


    CHECK( cudaMemcpy(imageOnGPU, image.data, N, cudaMemcpyHostToDevice) );

  cudaEventRecord(startCUDA,0);


   // размер массива ограничен максимальным размером пространства потоков
   removeColNum<<<(N+1023)/1024, 1024>>>(imageOnGPU, N, colorNum3);


   cudaEventRecord(stopCUDA,0);
   cudaEventSynchronize(stopCUDA);
   CHECK(cudaGetLastError());

   cudaEventElapsedTime(&elapsedTimeCUDA, startCUDA, stopCUDA);

   cout << "CUDA sum time = " << elapsedTimeCUDA << " ms\n";
   cout << "CUDA memory throughput = " << 3*N*sizeof(float)/elapsedTimeCUDA/1024/1024/1.024 << " Gb/s\n";

   CHECK( cudaMemcpy(image.data, imageOnGPU, N,cudaMemcpyDeviceToHost) );
   CHECK( cudaFree(imageOnGPU));



    imwrite("picGPU.jpg",image);

    //show image
    namedWindow( "Display window", WINDOW_AUTOSIZE );// Create a window for display.
    imshow( "Display window", image );                   // Show our image inside it.
    waitKey(0);                                          // Wait for a keystroke in the window
    return 0;
}
