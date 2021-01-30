#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <string.h>
#include "oclHelper.h"

#define __CL_ENABLE_EXCEPTIONS
#include <CL/cl.hpp>

#define VECTOR_SIZE (1024*1024)
#define ALPHA 5

void vector_saxpy(cl_int *y, const cl_int *x, cl_int alpha, int vector_size)
{
  for (int i = 0; i < vector_size; i++)
    {
      y[i] += x[i] * alpha;
    }
}

int main(int argc, char* argv[])
{
  cl_int err_msg, err_msg2;

  // Create host buffers
  cl_int *x_data = genRandomBuffer(VECTOR_SIZE);
  cl_int *y_data = genRandomBuffer(VECTOR_SIZE);
  cl_int *y_host_data = (cl_int *)malloc(sizeof(cl_int) * VECTOR_SIZE);
  cl_int *y_device_data = (cl_int *)malloc(sizeof(cl_int) * VECTOR_SIZE);
  cl_int *y_device_sep_data = (cl_int *)malloc(sizeof(cl_int) * VECTOR_SIZE);
  memcpy(y_host_data, y_data, sizeof(cl_int) * VECTOR_SIZE);

  std::vector<cl::Platform> platforms;
  std::vector<cl::Device> platform_devices;
  // Get Platforms count
  clPrintErrorExit(cl::Platform::get(&platforms), "cl::Platform::get");
  printf("Platforms:\n");
  for (int i = 0; i < platforms.size(); i++) {
    // Print platform name
    printf(" %d. platform name: %s.\n", i, platforms[i].getInfo<CL_PLATFORM_NAME>(&err_msg).c_str());
    clPrintErrorExit(err_msg, "cl::Platform::getInfo<CL_PLATFORM_NAME>");

    // Get platform devices count
    clPrintErrorExit(platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices), "getDevices");
    if (platform_devices.size() == 0) continue;

    for (int j = 0; j < platform_devices.size(); j++) {
      // Get device name
      printf("  %d. device name: %s.\n", j, platform_devices[j].getInfo<CL_DEVICE_NAME>(&err_msg).c_str());
      clPrintErrorExit(err_msg, "cl::Device::getInfo<CL_DEVICE_NAME>");
    }
    platform_devices.clear();
  }

  //===========================================================================================
  /* ======================================================
  * TODO 1. Cast
  * ziskat gpu device
  * =======================================================
  */

  cl::Device gpu_device = platform_devices[0];

  // check if device is correct
  if (gpu_device.getInfo<CL_DEVICE_TYPE>(&err_msg) == CL_DEVICE_TYPE_GPU) {
    printf("\nSelected device type: Correct\n");
  } else {
    printf("\nSelected device type: Incorrect\n");
  }
  clPrintErrorExit(err_msg, "cl::Device::getInfo<CL_DEVICE_TYPE>");
  printf("Selected device name: %s.\n", gpu_device.getInfo<CL_DEVICE_NAME>(&err_msg).c_str());
  clPrintErrorExit(err_msg, "cl::Device::getInfo<CL_DEVICE_NAME>");
  platforms.clear();

  //===========================================================================================
  /* ======================================================
  * TODO 2. Cast
  * vytvorit context a queue se zapnutym profilovanim
  * =======================================================
  */
  cl::Context context = cl::Context(gpu_device);
  cl::CommandQueue queue = cl::CommandQueue(context, CL_QUEUE_PROFILING_ENABLE);

  char * program_source = readFile("saxpy.cl");
  cl::Program::Sources sources;
  sources.push_back(std::pair<const char *, size_t>(program_source, strlen(program_source)));

  // get program
  cl::Program program(context, sources);
  clPrintErrorExit(err_msg, "clCreateProgramWithSource");

  // build program
  if ((err_msg = program.build(std::vector<cl::Device>(1, gpu_device), "", NULL, NULL)) == CL_BUILD_PROGRAM_FAILURE) {
    printf("Build log:\n %s", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(gpu_device, &err_msg2).c_str());
    clPrintErrorExit(err_msg2, "cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG>");
  }
  clPrintErrorExit(err_msg, "clBuildProgram");

  // create kernel_saxpy
  cl::Kernel kernel_saxpy(program, "vector_saxpy", &err_msg);
  clPrintErrorExit(err_msg, "cl::Kernel");
  cl::Kernel kernel_mul(program, "vector_mul", &err_msg);
  clPrintErrorExit(err_msg, "cl::Kernel");
  cl::Kernel kernel_add(program, "vector_add", &err_msg);
  clPrintErrorExit(err_msg, "cl::Kernel");

  //===========================================================================================
  /* ======================================================
  * TODO 3. Cast
  * vytvorit buffery
  * =======================================================
  */

  cl::Buffer x_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * VECTOR_SIZE);
  cl::Buffer y_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * VECTOR_SIZE);
  cl::Buffer x_sep_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * VECTOR_SIZE);
  cl::Buffer y_sep_buffer(context, CL_MEM_READ_WRITE, sizeof(cl_int) * VECTOR_SIZE);

  cl_int vector_size = VECTOR_SIZE;
  cl_int alpha = ALPHA;

  //===========================================================================================
  /* ======================================================
  * TODO 4. Cast
  * nastavit parametry spusteni kernelu
  * =======================================================
  */
  kernel_saxpy.setArg(0, y_buffer);
  kernel_saxpy.setArg(1, x_buffer);
  kernel_saxpy.setArg(2, ALPHA);
  kernel_saxpy.setArg(3, VECTOR_SIZE);
  kernel_mul.setArg(0, x_sep_buffer);
  kernel_mul.setArg(1, ALPHA);
  kernel_mul.setArg(2, VECTOR_SIZE);
  kernel_add.setArg(0, y_sep_buffer);
  kernel_add.setArg(1, x_sep_buffer);
  kernel_add.setArg(2, VECTOR_SIZE);

  // Eventy pro kopirovani na zarizeni u saxpy
  cl::UserEvent x_write_buffer_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent x_write_buffer_event");
  cl::UserEvent y_write_buffer_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent y_write_buffer_event");
  // Event pro kernel_saxpy
  cl::UserEvent saxpy_ndrange_kernel_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent saxpy_ndrange_kernel_event");
  // Event pro kopirovani ze zarizeni u saxpy
  cl::UserEvent y_read_buffer_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent y_read_buffer_event");

  // Eventy pro kopirovani na zarizeni u saxpy
  cl::UserEvent x_sep_write_buffer_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent x_sep_write_buffer_event");
  cl::UserEvent y_sep_write_buffer_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent y_sep_write_buffer_event");
  // Event pro kernels
  cl::UserEvent mul_sep_ndrange_kernel_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent mul_sep_ndrange_kernel_event");
  cl::UserEvent add_sep_ndrange_kernel_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent add_sep_ndrange_kernel_event");
  // Event pro kopirovani ze zarizeni u saxpy
  cl::UserEvent y_sep_read_buffer_event(context, &err_msg);
  clPrintErrorExit(err_msg, "clCreateUserEvent y_sep_read_buffer_event");

  //===========================================================================================
  /* ======================================================
  * TODO 5. Cast
  * nastavit velikost skupiny 256, kopirovat data na gpu, spusteni vyhodnoceni pomoci saxpy kernelu, kopirovani dat zpet
  * pro zarovnání muzete pouzit funkci alignTo(co, na_nasobek_ceho)
  * jako vystupni event kopirovani nastavte prepripravene eventy x_write_buffer_event y_write_buffer_event y_read_buffer_event
  * vystupni event kernelu saxpy_ndrange_kernel_event
  * =======================================================
  */

  double gpu_start = getTime();

  cl::NDRange local(256);
  cl::NDRange global(alignTo(VECTOR_SIZE, 256));

  queue.enqueueWriteBuffer(x_buffer, CL_FALSE, 0, sizeof(cl_int) * VECTOR_SIZE, x_data, NULL, &x_write_buffer_event);
  queue.enqueueWriteBuffer(y_buffer, CL_FALSE, 0, sizeof(cl_int) * VECTOR_SIZE, y_data, NULL, &y_write_buffer_event);
  queue.enqueueNDRangeKernel(kernel_saxpy, cl::NullRange, global, local, NULL, &saxpy_ndrange_kernel_event);
  queue.enqueueReadBuffer(y_buffer, CL_FALSE, 0, sizeof(cl_int) * VECTOR_SIZE, y_device_data, NULL, &y_read_buffer_event);

  // synchronize queue
  clPrintErrorExit(queue.finish(), "clFinish");

  double gpu_end = getTime();

  //===========================================================================================
  /* ======================================================
  * TODO 5. Cast
  * nastavit velikost skupiny 256, kopirovat data na gpu, spusteni vyhodnoceni pomoci mul a add kernelu, kopirovani dat zpet
  * pro zarovnání muzete pouzit funkci alignTo(co, na_nasobek_ceho)
  * jako vystupni event kopirovani nastavte prepripravene eventy x_sep_write_buffer_event y_sep_write_buffer_event y_sep_read_buffer_event
  * vystupni event kernelu mul_sep_ndrange_kernel_event a add_sep_ndrange_kernel_event
  * =======================================================
  */

  double gpu_sep_start = getTime();

  cl::NDRange local_sep(256);
  cl::NDRange global_sep(alignTo(VECTOR_SIZE, 256));

  queue.enqueueWriteBuffer(x_sep_buffer, CL_FALSE, 0, sizeof(cl_int) * VECTOR_SIZE, x_data, NULL, &x_sep_write_buffer_event);
  queue.enqueueWriteBuffer(y_sep_buffer, CL_FALSE, 0, sizeof(cl_int) * VECTOR_SIZE, y_data, NULL, &y_sep_write_buffer_event);
  queue.enqueueNDRangeKernel(kernel_mul, cl::NullRange, global_sep, local_sep, NULL, &mul_sep_ndrange_kernel_event);
  queue.enqueueNDRangeKernel(kernel_add, cl::NullRange, global_sep, local_sep, NULL, &add_sep_ndrange_kernel_event);
  queue.enqueueReadBuffer(y_sep_buffer, CL_FALSE, 0, sizeof(cl_int) * VECTOR_SIZE, y_device_sep_data, NULL, &y_sep_read_buffer_event);

  // synchronize queue
  clPrintErrorExit(queue.finish(), "clFinish");

  double gpu_sep_end = getTime();

  // compute results on host
  double cpu_start = getTime();
  vector_saxpy(y_host_data, x_data, alpha, vector_size);
  double cpu_end = getTime();

  // check data
  if (memcmp(y_device_data, y_host_data, VECTOR_SIZE * sizeof(cl_int)) == 0) {
    printf("\nResult saxpy: Correct\n");
  } else {
    printf("\nResult saxpy: Incorrect\n");
    // print results
    printf("\nExample saxpy results:\n");
    for (int i = 0; i < 20; i++) {
      printf(" [%d] %d * %d + %d = %d(gpu) %d(cpu)\n", i, x_data[i], ALPHA, y_data[i], y_device_data[i], y_host_data[i]);
    }
  }

  if (memcmp(y_device_sep_data, y_host_data, VECTOR_SIZE * sizeof(cl_int)) == 0) {
    printf("\nResult separable: Correct\n");
  } else {
    printf("\nResult separable: Incorrect\n");
    // print results
    printf("\nExample separable results:\n");
    for (int i = 0; i < 20; i++) {
      printf(" [%d] %d * %d + %d = %d(gpu) %d(cpu)\n", i, x_data[i], ALPHA, y_data[i], y_device_sep_data[i], y_host_data[i]);
    }
  }

  // print performance info
  printf("\nHost timers:\n");
  printf(" OpenCL processing time: %fs\n", gpu_end - gpu_start);
  printf(" CPU    processing time: %fs\n", cpu_end - cpu_start);
  printf("\nDevice timers:\n");
  printf(" OpenCL copy time saxpy: %fs\n", getEventTime(x_write_buffer_event) + getEventTime(y_write_buffer_event) + getEventTime(y_read_buffer_event));
  printf(" OpenCL processing time: saxpy %fs\n", getEventTime(saxpy_ndrange_kernel_event));
  printf(" OpenCL copy time separable: %fs\n", getEventTime(x_sep_write_buffer_event) + getEventTime(y_sep_write_buffer_event) + getEventTime(y_sep_read_buffer_event));
  printf(" OpenCL processing time: separable %fs\n", getEventTime(mul_sep_ndrange_kernel_event) + getEventTime(add_sep_ndrange_kernel_event));

  // deallocate host data
  free(x_data);
  free(y_data);
  free(y_host_data);
  free(y_device_data);

  return 0;
}

