#pragma comment( lib, "OpenCL" )

#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include "oclHelper.h"

#include <CL/cl.hpp>

#ifdef _WIN32
#include <windows.h>
#else
#include <sys/time.h>
#endif //WIN32

#define SELECTED_DEVICE_TYPE CL_DEVICE_TYPE_CPU

#define TEST_COUNT 3

int array_sum(int *a, int array_size)
{
    int result = 0;
    for (int i = 0; i < array_size; i++)
    {
        result += a[i];
    }
    return result;
}

typedef struct
{
    cl_int a_w;
    size_t local;
    const char *info;
}s_test;

int main(int argc, char* argv[])
{
    cl_int err_msg, err_msg2;
    s_test tests[TEST_COUNT] = { { 524288, 256, " Work group: 256\n Array a: 524288" },
    { 524288, 128, " Work group: 128\n Array a: 524288" },
    { 524289, 256, " Work group: 256\n Array a: 524289" } };

    std::vector<cl::Platform> platforms;
    std::vector<cl::Device> platform_devices;
    // Get Platforms count
    clPrintErrorExit(cl::Platform::get(&platforms), "cl::Platform::get");
    printf("Platforms:\n");
    for (unsigned int i = 0; i < platforms.size(); i++)
    {
        // Print platform name
        printf(" %d. platform name: %s.\n", i, platforms[i].getInfo<CL_PLATFORM_NAME>(&err_msg).c_str());
        clPrintErrorExit(err_msg, "cl::Platform::getInfo<CL_PLATFORM_NAME>");

        // Get platform devices count
        clPrintErrorExit(platforms[i].getDevices(CL_DEVICE_TYPE_ALL, &platform_devices), "getDevices");
        for (unsigned int j = 0; j < platform_devices.size(); j++)
        {
            // Get device name
            printf("  %d. device name: %s.\n", j, platform_devices[j].getInfo<CL_DEVICE_NAME>(&err_msg).c_str());
            clPrintErrorExit(err_msg, "cl::Device::getInfo<CL_DEVICE_NAME>");
        }
        platform_devices.clear();
    }

    cl::Device selected_device;
    bool device_found = false;
    for (unsigned int i = 0; i < platforms.size(); i++)
    {
        clPrintErrorExit(platforms[i].getDevices(SELECTED_DEVICE_TYPE, &platform_devices), "getDevices");
        if (platform_devices.size() != 0)
        {
            device_found = true;
            selected_device = platform_devices[0];
            break;
        }
    }
    if (!device_found) clPrintErrorExit(CL_DEVICE_NOT_FOUND, "GPU device");

    // check if device is correct
    if (selected_device.getInfo<CL_DEVICE_TYPE>() == SELECTED_DEVICE_TYPE)
    {
        printf("\nSelected device type: Correct\n");
    }
    else
    {
        printf("\nSelected device type: Incorrect\n");
    }
    printf("Selected device name: %s.\n", selected_device.getInfo<CL_DEVICE_NAME>().c_str());

    platforms.clear();

    cl::Context context(selected_device, NULL, NULL, NULL, &err_msg);
    clPrintErrorExit(err_msg, "cl::Context");

    cl::CommandQueue queue(context, selected_device, CL_QUEUE_PROFILING_ENABLE, &err_msg);
    clPrintErrorExit(err_msg, "cl::CommandQueue");

    char *program_source = readFile("array_reduce.cl");
    cl::Program::Sources sources;
    sources.push_back(std::pair<const char *, size_t>(program_source, 0));

    cl::Program program(context, sources, &err_msg);
    clPrintErrorExit(err_msg, "clCreateProgramWithSource");

    // build program
    if ((err_msg = program.build(std::vector<cl::Device>(1, selected_device), "", NULL, NULL)) == CL_BUILD_PROGRAM_FAILURE)
    {
        printf("Build log:\n %s", program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(selected_device, &err_msg2).c_str());
        clPrintErrorExit(err_msg2, "cl::Program::getBuildInfo<CL_PROGRAM_BUILD_LOG>");
    }
    clPrintErrorExit(err_msg, "clBuildProgram");

    cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&> global_atomic_reduce_sum = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&>(program, "global_atomic_reduce_sum", &err_msg);
    cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&> local_atomic_reduce_sum = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&>(program, "local_atomic_reduce_sum", &err_msg);
    cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&, const cl::LocalSpaceArg&> local_reduce_sum = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&, const cl::LocalSpaceArg&>(program, "local_reduce_sum", &err_msg);
    cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&, const cl::LocalSpaceArg&> local_naive_reduce_sum = cl::make_kernel<cl::Buffer&, cl::Buffer&, cl_int&, const cl::LocalSpaceArg&>(program, "local_naive_reduce_sum", &err_msg);

    for (int i = 0; i < TEST_COUNT; i++)
    {
        cl_int a_w = tests[i].a_w;
        cl_int result;
        cl_int result_global_atomic;
        cl_int result_local_atomic;
        cl_int result_local;
        cl_int result_local_naive;
        cl_int result_init = 0;

        size_t a_size = sizeof(cl_int) * a_w;
        size_t result_size = sizeof(cl_int);

        cl_int *a_host = genRandomBuffer(a_w);

        cl::Buffer a_dev(context, CL_MEM_READ_ONLY, a_size, NULL, &err_msg);
        clPrintErrorExit(err_msg, "clCreateBuffer: a_dev");
        cl::Buffer result_global_atomic_dev(context, CL_MEM_READ_WRITE, result_size, NULL, &err_msg);
        clPrintErrorExit(err_msg, "clCreateBuffer: result_global_atomic_dev");
        cl::Buffer result_local_atomic_dev(context, CL_MEM_READ_WRITE, result_size, NULL, &err_msg);
        clPrintErrorExit(err_msg, "clCreateBuffer: result_local_atomic_dev");
        cl::Buffer result_local_dev(context, CL_MEM_READ_WRITE, result_size, NULL, &err_msg);
        clPrintErrorExit(err_msg, "clCreateBuffer: result_local_dev");
        cl::Buffer result_local_naive_dev(context, CL_MEM_READ_WRITE, result_size, NULL, &err_msg);
        clPrintErrorExit(err_msg, "clCreateBuffer: result_local_naive_dev");

        cl::UserEvent a_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent a_event");

        cl::UserEvent in_counter_global_atomic_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent in_counter_global_atomic_event");
        cl::UserEvent out_counter_global_atomic_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent out_counter_global_atomic_event");

        cl::UserEvent in_counter_local_atomic_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent in_counter_local_atomic_event");
        cl::UserEvent out_counter_local_atomic_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent out_counter_local_atomic_event");

        cl::UserEvent in_counter_local_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent in_counter_local_event");
        cl::UserEvent out_counter_local_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent out_counter_local_event");

        cl::UserEvent in_counter_local_naive_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent in_counter_local_naive_event");
        cl::UserEvent out_counter_local_naive_event(context, &err_msg);
        clPrintErrorExit(err_msg, "clCreateUserEvent out_counter_local_naive_event");

        cl::NDRange local(tests[i].local);
        cl::NDRange global(alignTo(a_w, local[0]));

        double cpu_start = getTime();
        result = array_sum(a_host, a_w);
        double cpu_end = getTime();

        clPrintErrorExit(queue.enqueueWriteBuffer(a_dev, CL_FALSE, 0, a_size, a_host, NULL, &a_event), "clEnqueueWriteBuffer: a_dev");

        clPrintErrorExit(queue.enqueueWriteBuffer(result_global_atomic_dev, CL_FALSE, 0, result_size, &result_init, NULL, &in_counter_global_atomic_event), "clEnqueueWriteBuffer: result_global_atomic_dev");

        cl::Event kernel_global_atomic_event = global_atomic_reduce_sum(cl::EnqueueArgs(queue, global, local), a_dev, result_global_atomic_dev, a_w);

        clPrintErrorExit(queue.enqueueReadBuffer(result_global_atomic_dev, CL_FALSE, 0, result_size, &result_global_atomic, NULL, &out_counter_global_atomic_event), "clEnqueueReadBuffer: result_global_atomic_dev");

        // synchronize queue
        clPrintErrorExit(queue.finish(), "clFinish");

        clPrintErrorExit(queue.enqueueWriteBuffer(result_local_atomic_dev, CL_FALSE, 0, result_size, &result_init, NULL, &in_counter_local_atomic_event), "clEnqueueWriteBuffer: result_local_atomic_dev");

        cl::Event kernel_local_atomic_event = local_atomic_reduce_sum(cl::EnqueueArgs(queue, global, local), a_dev, result_local_atomic_dev, a_w);

        clPrintErrorExit(queue.enqueueReadBuffer(result_local_atomic_dev, CL_FALSE, 0, result_size, &result_local_atomic, NULL, &out_counter_local_atomic_event), "clEnqueueReadBuffer: result_local_atomic_dev");

        // synchronize queue
        clPrintErrorExit(queue.finish(), "clFinish");

        clPrintErrorExit(queue.enqueueWriteBuffer(result_local_dev, CL_FALSE, 0, result_size, &result_init, NULL, &in_counter_local_event), "clEnqueueWriteBuffer: result_local_dev");

        cl::Event kernel_local_event = local_reduce_sum(cl::EnqueueArgs(queue, global, local), a_dev, result_local_dev, a_w, cl::Local(sizeof(cl_int) * local[0]));

        clPrintErrorExit(queue.enqueueReadBuffer(result_local_dev, CL_FALSE, 0, result_size, &result_local, NULL, &out_counter_local_event), "clEnqueueReadBuffer: result_local_dev");

        // synchronize queue
        clPrintErrorExit(queue.finish(), "clFinish");

        clPrintErrorExit(queue.enqueueWriteBuffer(result_local_naive_dev, CL_FALSE, 0, result_size, &result_init, NULL, &in_counter_local_naive_event), "clEnqueueWriteBuffer: result_local_dev");

        cl::Event kernel_local_naive_event = local_naive_reduce_sum(cl::EnqueueArgs(queue, global, local), a_dev, result_local_naive_dev, a_w, cl::Local(sizeof(cl_int) * local[0]));

        clPrintErrorExit(queue.enqueueReadBuffer(result_local_naive_dev, CL_FALSE, 0, result_size, &result_local_naive, NULL, &out_counter_local_naive_event), "clEnqueueReadBuffer: result_local_dev");

        // synchronize queue
        clPrintErrorExit(queue.finish(), "clFinish");

        printf("\nTest %d:\n%s\n", i, tests[i].info);
        printf(" Global work dim: %d\n", global[0]);
        printf(" Result: %d\n", result);
        printf(" Global basic kernel:\n");
        if (result_global_atomic == result)
        {
            printf("  Result: %d Correct\n", result_global_atomic);
            printf("  Timers: cpu:%.3fms ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
                (cpu_end - cpu_start) * 1000,
                (getEventTime(a_event) + getEventTime(in_counter_global_atomic_event) + getEventTime(out_counter_global_atomic_event) + getEventTime(kernel_global_atomic_event))*1000,
                (getEventTime(a_event) + getEventTime(in_counter_global_atomic_event) + getEventTime(out_counter_global_atomic_event))*1000,
                getEventTime(kernel_global_atomic_event)*1000);
        }
        else
        {
            printf("  Result: %d Incorrect\n", result_global_atomic);
        }
        printf(" Local atomic kernel:\n");
        if (result_local_atomic == result)
        {
            printf("  Result: %d Correct\n", result_local_atomic);
            printf("  Timers: cpu:%.3fms ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
                (cpu_end - cpu_start) * 1000,
                (getEventTime(a_event) + getEventTime(in_counter_local_atomic_event) + getEventTime(out_counter_local_atomic_event) + getEventTime(kernel_local_atomic_event))*1000,
                (getEventTime(a_event) + getEventTime(in_counter_local_atomic_event) + getEventTime(out_counter_local_atomic_event))*1000,
                getEventTime(kernel_local_atomic_event)*1000);
        }
        else
        {
            printf("  Result: %d Incorrect\n", result_local_atomic);
        }
        printf(" Local kernel:\n");
        if (result_local == result)
        {
            printf("  Result: %d Correct\n", result_local);
            printf("  Timers: cpu:%.3fms ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
                (cpu_end - cpu_start) * 1000,
                (getEventTime(a_event) + getEventTime(in_counter_local_event) + getEventTime(out_counter_local_event) + getEventTime(kernel_local_event))*1000,
                (getEventTime(a_event) + getEventTime(in_counter_local_event) + getEventTime(out_counter_local_event))*1000,
                getEventTime(kernel_local_event)*1000);
        }
        else
        {
            printf("  Result: %d Incorrect\n", result_local);
        }
        printf(" Local naive kernel:\n");
        if (result_local_naive == result)
        {
            printf("  Result: %d Correct\n", result_local_naive);
            printf("  Timers: cpu:%.3fms ocl:%.3fms ocl_copy:%.3fms ocl_kernel:%.3fms\n",
                (cpu_end - cpu_start) * 1000,
                (getEventTime(a_event) + getEventTime(in_counter_local_naive_event) + getEventTime(out_counter_local_naive_event) + getEventTime(kernel_local_naive_event)) * 1000,
                (getEventTime(a_event) + getEventTime(in_counter_local_naive_event) + getEventTime(out_counter_local_naive_event)) * 1000,
                getEventTime(kernel_local_naive_event) * 1000);
        }
        else
        {
            printf("  Result: %d Incorrect\n", result_local_naive);
        }

        // deallocate host data
        free(a_host);
    }

    getchar();

    return 0;
}

