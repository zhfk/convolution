/*
 * main.c
 *
 *  Created on: 2016年10月12日
 *      Author: zhfk
 */
//For clarity,error checking has been omitted.
//#pragma warning( disable : 4996 )

#include "tool.h"
using namespace std;
int main(int argc, char* argv[])
{
	double start_time,end_time;
	cl_uint numOfDevice;
	cl_event events[2];
	cl_program program;
	cl_int    status;
	/**Step 1: Getting platforms and choose an available one(first).*/
	cl_platform_id platform;
	cout<<"program running---->"<<endl;
	getPlatform(platform);
	/**Step 2:Query the platform and choose the first GPU device if has one.*/
	cl_device_id *devices = getCl_device_id(platform,numOfDevice);

	/**Step 3: Create context.*/
	cl_context context = clCreateContext(NULL, 1, devices, NULL, NULL, NULL);

	/**Step 4: Creating command queue associate with the context.*/
	cl_command_queue commandQueue = clCreateCommandQueue(context, *devices, 0, NULL);

	//此处添加读取二进制 kernel文件
	  // Create the program for all device. Use the first device as the
	  // representative device (assuming all device are of the same type).

	const char *cl_kernel_file="CL_kernel";
	string binary_file = getBoardBinaryFile(cl_kernel_file, devices[0]);
	printf("%-15s ===> %s \n","Using AOCX",binary_file.c_str());
	program= createProgramFromBinary(context, binary_file.c_str(), devices, numOfDevice);


	 // debug("Kernel execute scale calculate Matrix Multiple [%d x %d]",SIZE,SIZE);
	  // Build the program that was just created.
	status = clBuildProgram(program, 0, devices, "", NULL, NULL);
	// Shows the log
	ShowBuildLog(program, devices);

	/**Step 7: Initial input,output for the host and create memory objects for the kernel*/
	size_t m=28,n=28;
	size_t km=5,kn=5;
	size_t kns = n-kn+1;
	size_t kms = m-km+1;
	size_t input_size = m*n;
	size_t kenel_size = km*kn;
	size_t output_szie = kns*kms;

	float *input = new float[input_size];
	float *cal_kernel = new float[kenel_size];
	float *output = new float[output_szie];

		for (size_t i = 0; i < m; i++)
			for (size_t j = 0; j < n; j++)
				input[i*n+j]=i+j;

		for (size_t i = 0; i < km; i++)
			for (size_t j = 0; j < kn; j++)
				cal_kernel[i*kn + j] = i+j;

	debug_msg(INFO,"Kernel execute start...");
	start_time=getCurrentTimestamp();
	printf("%-15s ===> %.3f %s \n","start_time",start_time*1e3," Ms");
	//cout << "clCreateBuffer---------->" << endl << endl;
	cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR , input_size*sizeof(float), (void*)input, NULL);
	cl_mem kernelBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY|CL_MEM_COPY_HOST_PTR , kenel_size*sizeof(float), (void*)cal_kernel, NULL);
	cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, output_szie*sizeof(float), NULL, NULL);

	const char* KernelFunction="convolution";
	cl_kernel CL_kernel = clCreateKernel(program, KernelFunction, NULL);

	/**Step 9: Sets Kernel arguments.*/
	//cout << "clSetKernelArg---------->" << endl << endl;
	//设置Kernel参数
	cclSetKernelArg(CL_kernel, 0, sizeof(cl_mem),  (void *)&inputBuffer);
	clSetKernelArg(CL_kernel, 1, sizeof(cl_mem),  (void *)&kernelBuffer);
	clSetKernelArg(CL_kernel, 2, sizeof(cl_int), (void *)&m);
	clSetKernelArg(CL_kernel, 3, sizeof(cl_int), (void *)&km);
	clSetKernelArg(CL_kernel, 4, sizeof(cl_mem),  (void *)&outputBuffer);
	clSetKernelArg(CL_kernel, 5, sizeof(cl_int), (void *)&kms);


	/**Step 10: Running the kernel.*/
	size_t globalThreads[] = {m, n};
	//size_t localThreads[] = {32, 32}; // localx*localy应该是64的倍数
	//printf("global_work_size =(%d,%d), local_work_size=(16, 16)\n",W,H);
	printf("%-15s ===> <%d,%d> \n","globalworksize",globalThreads[0],globalThreads[1]);
	//printf("%-15s ===> <%d,%d> \n","localworksize",localThreads[0],localThreads[1]);
	//	const size_t local_ws = 512;    // Number of work-items per work-group
	//	cl_event enentPoint;
	//cout << "clEnqueueNDRangeKernel---------->" << endl << endl;
	status = clEnqueueNDRangeKernel(commandQueue, CL_kernel, MatrixDim, NULL, globalThreads, NULL, 0, NULL, &events[0]);
	status = clWaitForEvents(1,&events[0]);
	if (status != CL_SUCCESS)
	{
		//cout <<"Error: Waiting for kernel run to finish.(clWaitForEvents)"<<endl;
		debug_msg(status, "Waiting for kernel run to finish.(clWaitForEvents)");
		return 0;
	}
	status = clReleaseEvent(events[0]);
	//计算kerenl执行时间
	//cl_ulong startTime, endTime;
	//clGetEventProfilingInfo(events[0], CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &startTime, NULL);
	//clGetEventProfilingInfo(events[0],  CL_PROFILING_COMMAND_END,sizeof(cl_ulong), &endTime, NULL);
	//cl_ulong kernelExecTimeNs = endTime-startTime;
	//printf("kernal exec time :%8.6f ms\n ", kernelExecTimeNs*1e-6 );
	//printf("%-15s ===> %.3f %s \n ", "ker_exec_time", kernelExecTimeNs*1e-6,"Ms");
	//将结果拷贝到主机端
	end_time = getCurrentTimestamp();
	//cout << "end_time :" << end_time*1e3<<" Ms" << endl;
	printf("%-15s ===> %.3f %s \n","end_time",end_time*1e3," Ms");
	//cout << "took time :" << ((end_time - start_time) * 1e3) << " Ms"<< endl;
	printf("%-15s ===> %.3f %s \n","ker_took_time",((end_time - start_time) * 1e3)," Ms");
	debug_msg(INFO,"Kernel execute finish !");

	/**Step 11: Read the cout put back to host memory.*/
	//cout << "clEnqueueReadBuffer---------->" << endl << endl;
	//out_image=(unsigned char*)malloc(mem_size);
	//status = clEnqueueReadBuffer(commandQueue, outputBuffer, CL_TRUE, 0, mem_size, out_image, 0, NULL, &events[1]);
	clEnqueueReadBuffer(*clcpp._queue, outputBuffer, CL_TRUE, 0,output_szie * sizeof(int), output, 0, NULL,  &events[1]);
		status = clWaitForEvents(1, &events[1]);
		if (status != CL_SUCCESS)
		{
			//cout <<"Error: Waiting for read buffer call to finish. (clWaitForEvents)"<<endl;
			debug_msg(status, "Waiting for clEnqueueMapBuffer call to finish. (clWaitForEvents)");
			return ;
		}
		status = clReleaseEvent(events[1]);
	const char *outfileName="FGPA_convolution_lenna.jpg";

	debug_msg(INFO,"image %s saved in current directory",outfileName);
	/**Step 12: Clean the resources.*/
	//cout << "clRelease---------->" << endl << endl;
	status = clReleaseKernel(CL_kernel);//*Release kernel.
	status = clReleaseProgram(program);    //Release the program object.
	status = clReleaseMemObject(inputBuffer_a);//Release mem object.
	status = clReleaseMemObject(kernelBuffer);//Release mem object.
	status = clReleaseMemObject(outputBuffer_c);
	status = clReleaseCommandQueue(commandQueue);//Release  Command queue.
	status = clReleaseContext(context);//Release context.

	if (devices)
	{
		free(devices);
		devices = NULL;
	}
	cout << "program over--------->" << endl << endl;
	return 0;
}

