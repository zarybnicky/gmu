#pragma comment( lib, "OpenCL" )

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <algorithm>

#include <CL/cl.hpp>
#include "oclHelper.h"

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif //WIN32

#define TEST_COUNT 5
#define TMP_BUFFER_SIZE 4096

#define SELECTED_DEVICE_TYPE CL_DEVICE_TYPE_CPU

#undef max


void matrix_mul(int *a, int *b, int *c, int a_w, int a_h, int b_w)
{
  int b_h = a_w;
  int c_w = b_w;
  int c_h = a_h;
  for(int y = 0; y < c_h; y++) {
    for(int x = 0; x < c_w; x++) {
      int result = 0;
      for(int i = 0; i < a_w; i++) {
        result += a[i + y * a_w] * b[x + i * b_w];
      }
      c[x + y * c_w] = result;
    }
  }
}

void set_matrix(int *a, int width, int height)
{
  for(int y = 0; y < height; y++) {
    for(int x = 0; x < width; x++) {
      a[y * width + x] = (x + y) % 256;
    }
  }
}

typedef struct {
  cl_int a_w;
  cl_int a_h;
  cl_int b_w;
  size_t local[2];
  const char *info;
} s_test;

int main(int argc, char* argv[])
{
  cl_int err_msg, err_msg2;
  s_test tests[TEST_COUNT] = {{16,512,512,{16,16}, " Work group: 16x16\n Matrix a: 16x512\n Matrix b: 512x16\n Matrix c: 512x512"},
                              {32,512,512,{16,16}, " Work group: 16x16\n Matrix a: 32x512\n Matrix b: 512x32\n Matrix c: 512x512"},
                              {512,512,512,{16,16}, " Work group: 16x16\n Matrix a: 512x512\n Matrix b: 512x512\n Matrix c: 512x512"},
                              {512,513,511,{16,16}, " Work group: 16x16\n Matrix a: 512x513\n Matrix b: 511x512\n Matrix c: 511x513"},
                              {1024,1024,1024,{16,16}, " Work group: 16x16\n Matrix a: 1024x1024\n Matrix b: 1024x1024\n Matrix c: 1024x1024"}};

  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> platform_devices;
  // Get Platforms count
  clPrintErrorExit(cl::Platform::get(&platforms), "cl::Platform::get");
  printf("Platforms:\n");
  for(unsigned int i = 0; i < platforms.size(); i++) {
    // Print platform name
    printf(" %d. platform name: %s.\n", i, platforms[i].getInfo<CL_PLATFORM_NAME>(&err_msg).c_str());
    clPrintErrorExit(err_msg,"cl::Platform::getInfo<CL_PLATFORM_NAME>");

    // Get platform devices count
    clPrintErrorExit(platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices), "getDevices");
    for(unsigned int j = 0; j < platform_devices.size(); j++) {
      // Get device name
      printf("  %d. device name: %s.\n", j, platform_devices[j].getInfo<CL_DEVICE_NAME>(&err_msg).c_str());
      clPrintErrorExit(err_msg, "cl::Device::getInfo<CL_DEVICE_NAME>");
    }
    platform_devices.clear();
  }

  cl::Device selected_device;
  bool device_found = false;
  for(unsigned int i = 0; i < platforms.size(); i++) {
    clPrintErrorExit(platforms[i].getDevices(SELECTED_DEVICE_TYPE, &platform_devices), "getDevices");
    if(platform_devices.size() != 0) {
      device_found = true;
      selected_device = platform_devices[0];
      break;
    }
  }
  if(!device_found) clPrintErrorExit(CL_DEVICE_NOT_FOUND, "GPU device");

  // check if device is correct
  if(selected_device.getInfo<CL_DEVICE_TYPE>() == SELECTED_DEVICE_TYPE) {
    printf("\nSelected device type: Correct\n");
  } else {
    printf("\nSelected device type: Incorrect\n");
  }
  printf("Selected device name: %s.\n", selected_device.getInfo<CL_DEVICE_NAME>().c_str());

  platforms.clear();

  cl::Context context(selected_device, NULL, NULL, NULL, &err_msg);
  clPrintErrorExit(err_msg,"cl::Context");

  cl::CommandQueue queue(context, selected_device, CL_QUEUE_PROFILING_ENABLE, &err_msg);
  clPrintErrorExit(err_msg,"cl::CommandQueue");

  char *program_source = readFile("matrix_mul.cl");
  cl::Program::Sources sources;
  sources.push_back(std::pair<const char *, size_t>(program_source, 0));

  cl::Program program(context, sources, &err_msg);
  //cl_program program2 = clCreateProgramWithSource(context, 1, (const char **)&program_source, NULL, &err_msg);
  clPrintErrorExit(err_msg,"clCreateProgramWithSource");

  // build program
  if((err_msg = program.build(std::vector<cl::Device>(1,selected_device), "", NULL, NULL)) == CL_BUILD_PROGRAM_FAILURE) {
    printf("Build log:\n %s", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device, &err_msg2).c_str());
    clPrintErrorExit(err_msg2, "cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG>");
  }
  clPrintErrorExit(err_msg, "clBuildProgram");

  // create kernel functors
  cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int&, cl_int&, cl_int&> matrix_mul_basic = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int&, cl_int&, cl_int&>(program, "matrix_mul_basic", &err_msg);
  cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int&, cl_int&, cl_int&, const cl::LocalSpaceArg&, const cl::LocalSpaceArg&> matrix_mul_local = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl::Buffer&, cl_int&, cl_int&, cl_int&, const cl::LocalSpaceArg&, const cl::LocalSpaceArg&>(program, "matrix_mul_local", &err_msg);

  for(int i = 0; i < TEST_COUNT; i++) {
    cl_int a_w = tests[i].a_w;
    cl_int a_h = tests[i].a_h;
    cl_int b_w = tests[i].b_w;

    cl_int b_h = a_w;
    cl_int c_w = b_w;
    cl_int c_h = a_h;

    size_t a_size = sizeof(cl_int) * a_w * a_h;
    size_t b_size = sizeof(cl_int) * b_w * b_h;
    size_t c_size = sizeof(cl_int) * c_w * c_h;

    cl_int *a_host = genRandomBuffer(a_w * a_h);
    cl_int *b_host = genRandomBuffer(b_w * b_h);
    cl_int *c_host = (cl_int *)malloc(c_size);
    cl_int *c_basic_host = (cl_int *)malloc(c_size);
    cl_int *c_local_host = (cl_int *)malloc(c_size);

    cl::Buffer a_dev(context, CL_MEM_READ_ONLY, a_size, NULL, &err_msg);
    clPrintErrorExit(err_msg,"clCreateBuffer: a_dev");
    cl::Buffer b_dev(context, CL_MEM_READ_ONLY, b_size, NULL, &err_msg);
    clPrintErrorExit(err_msg,"clCreateBuffer: b_dev");
    cl::Buffer c_basic_dev(context, CL_MEM_READ_WRITE, c_size, NULL, &err_msg);
    clPrintErrorExit(err_msg,"clCreateBuffer: c_basic_dev");
    cl::Buffer c_local_dev(context, CL_MEM_READ_WRITE, c_size, NULL, &err_msg);
    clPrintErrorExit(err_msg,"clCreateBuffer: c_local_dev");

    cl::UserEvent a_event(context, &err_msg);
    clPrintErrorExit(err_msg, "clCreateUserEvent a_event");
    cl::UserEvent b_event(context, &err_msg);
    clPrintErrorExit(err_msg, "clCreateUserEvent b_event");

    cl::UserEvent c_basic_event(context, &err_msg);
    clPrintErrorExit(err_msg, "clCreateUserEvent c_basic_event");

    cl::UserEvent c_local_event(context, &err_msg);
    clPrintErrorExit(err_msg, "clCreateUserEvent c_local_event");

    cl::NDRange local(tests[i].local[0], tests[i].local[1]);
    cl::NDRange global(alignTo(c_w, local[0]), alignTo(c_h, local[1]));

    size_t local_dim_size = std::max(tests[i].local[0], tests[i].local[1]);
    size_t local_size = sizeof(cl_int) * local_dim_size * local_dim_size;

    double cpu_start = getTime();
    matrix_mul(a_host, b_host, c_host, a_w, a_h, b_w);
    double cpu_end = getTime();

    clPrintErrorExit(queue.enqueueWriteBuffer(a_dev, CL_FALSE, 0, a_size, a_host, NULL, &a_event),"clEnqueueWriteBuffer: a_dev");

    clPrintErrorExit(queue.enqueueWriteBuffer(b_dev, CL_FALSE, 0, b_size, b_host, NULL, &b_event),"clEnqueueWriteBuffer: b_dev");

    cl::Event kernel_basic_event = matrix_mul_basic(cl::EnqueueArgs(queue, global, local), a_dev, b_dev, c_basic_dev, a_w, a_h, b_w);

    clPrintErrorExit(queue.enqueueReadBuffer(c_basic_dev, CL_FALSE, 0, c_size, c_basic_host, NULL, &c_basic_event),"clEnqueueWriteBuffer: c_basic_dev");

    // synchronize queue
    clPrintErrorExit(queue.finish(), "clFinish");

    cl::Event kernel_local_event = matrix_mul_local(cl::EnqueueArgs(queue, global, local), a_dev, b_dev, c_local_dev, a_w, a_h, b_w, cl::Local(local_size), cl::Local(local_size));

    clPrintErrorExit(queue.enqueueReadBuffer(c_local_dev, CL_FALSE, 0, c_size, c_local_host, NULL, &c_local_event),"clEnqueueWriteBuffer: c_local_dev");

    // synchronize queue
    clPrintErrorExit(queue.finish(), "clFinish");

    printf("\nTest %d:\n%s\n", i, tests[i].info);
    printf(" Global work dim: %lux%lu\n", global[0], global[1]);
    printf(" Basic kernel:\n");
    if(memcmp(c_basic_host, c_host, c_size) == 0) {
      printf("  Result: Correct\n");
      printf("  Timers: cpu:%.3fms ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
             (cpu_end - cpu_start) * 1000,
             (getEventTime(a_event) + getEventTime(b_event) + getEventTime(c_basic_event) + getEventTime(kernel_basic_event))*1000,
             (getEventTime(a_event) + getEventTime(b_event) + getEventTime(c_basic_event))*1000,
             getEventTime(kernel_basic_event)*1000);
    } else {
      printf("  Result: Incorrect\n");
    }
    printf(" Local kernel:\n");
    if (memcmp(c_local_host, c_host, c_size) == 0) {
      printf("  Result: Correct\n");
      printf("  Timers: cpu:%.3fms ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
             (cpu_end - cpu_start) * 1000,
             (getEventTime(a_event) + getEventTime(b_event) + getEventTime(c_local_event) + getEventTime(kernel_local_event))*1000,
             (getEventTime(a_event) + getEventTime(b_event) + getEventTime(c_local_event))*1000,
             getEventTime(kernel_local_event)*1000);
    } else {
      printf("  Result: Incorrect\n");
    }

    // for (int i = 0; i < a_h; i++) {
    //   for (int j = 0; j < a_w; j++) {
    //     printf("%d ", a_host[i * a_h + j]);
    //   }
    //   printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < b_h; i++) {
    //   for (int j = 0; j < b_w; j++) {
    //     printf("%d ", b_host[i * b_h + j]);
    //   }
    //   printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < c_h; i++) {
    //   for (int j = 0; j < c_w; j++) {
    //     printf("%d ", c_basic_host[i * c_h + j]);
    //   }
    //   printf("\n");
    // }
    // printf("\n");
    // for (int i = 0; i < c_h; i++) {
    //   for (int j = 0; j < c_w; j++) {
    //     printf("%d ", c_local_host[i * c_h + j]);
    //   }
    //   printf("\n");
    // }
    // printf("\n");

    // deallocate host data
    free(a_host);
    free(b_host);
    free(c_host);
    free(c_basic_host);
    free(c_local_host);
    // break;
  }

  return 0;
}
