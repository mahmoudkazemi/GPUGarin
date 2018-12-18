#include <stdio.h>
#include <time.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <string.h>
#include <iostream>
#include <math.h>
#include <assert.h>
#include <fstream>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
//#include "helper_cuda.h"
#include "cuComplex.h"
//#include <math_functions.h>
#include <cufft.h>
#include "gputimer.h"
#include "gpuerrors.h"
#include "reconst.h"

#define PI 3.141592653589793115997963468544185161590576171875f

// ===========================> Functions Prototype <===============================
void fill(cuFloatComplex* data, int size);
void gpuKernel(cuFloatComplex* in, cuFloatComplex* out);
void printmat(cuFloatComplex* data,int row,int col, char* title);
int read_data(cuFloatComplex* data,char* address);
void write_data(cuFloatComplex* data,char* address, int length);

cufftHandle handle;
int rank = 1;                           // --- 1D FFTs
int n[] = { Nz_padded };                 // --- Size of the Fourier transform
int istride = 1, ostride = 1;           // --- Distance between two successive input/output elements
int idist = Nz_padded, odist = (Nz_padded); // --- Distance between batches
int inembed[] = { 0 };                  // --- Input size with pitch (ignored for 1D transforms)
int onembed[] = { 0 };                  // --- Output size with pitch (ignored for 1D transforms)
int batch = Nf;                      // --- Number of batched executions

int main(int argc, char** argv) {
	struct cudaDeviceProp p;
	cudaGetDeviceProperties(&p, 0);
	printf("Device Name: %s\n", p.name);
	
	cuFloatComplex* data;
	data = (cuFloatComplex*)malloc(Nf*Nz_padded*Nphi * sizeof(cuFloatComplex));
	//fill(data, Nf*Nz*Nphi);
	
	int readDT = read_data(data, "./data.dat");
	printf("No. of Data Read: %d\n", readDT);
	
	cuFloatComplex* out;
	out = (cuFloatComplex*)malloc(Nz_padded*Ny*Nx * sizeof(cuFloatComplex));

	cufftPlanMany(&handle, rank, n, 
		inembed, istride, idist,
		onembed, ostride, odist, CUFFT_C2C, batch);
	
	//time measurement for GPU calculations
	clock_t t1 = clock();
	gpuKernel (data, out);
	clock_t t2 = clock();
	
	//printmat(temph,row+row/2,n,"temph");
	//printmat(out,m,n,"output Matrix");
	
	printf("Execution Time: %ld\n", (t2-t1)/1000);
	
	write_data(out, "./out.dat", Nz_padded*Ny*Nx);
	
	free(out);
	free(data);
	return 0;
}

int read_data(cuFloatComplex* data,char* address)
{
	unsigned int num=0;
	std::ifstream infile(address);
	std::string line;
	char cstr[100];
	char* token;
	unsigned int numline = 0;
	while (std::getline(infile, line))
	{
		if(line.size()>1)
		{
			strcpy(cstr, line.c_str());
			num = 0;
			token = strtok(cstr, ",");
			while (token) {
				assert(num < 2);
				if (num == 0)
				{
					data[numline].x = atof(token);
				} else
				{
					data[numline].y = atof(token);
				}
				token = strtok(NULL, ",");
				num++;
			}
			++numline;
		}
	}
	return numline;
	//printf("Hello, %d!\n",numline);
}

void write_data(cuFloatComplex* data,char* address, int length)
{
	FILE * fp;
	fp = fopen(address, "w");
	if (fp == NULL)
	exit(EXIT_FAILURE);
	for(int i=0; i<length; i++)
	{
		fprintf(fp, "%.6E,%.6E\n", data[i].x, data[i].y);
	}
	fclose(fp);
}

void fill(cuFloatComplex* data, int size) {
	for (int i=0; i<size; ++i)
	{
		data[i].x = (float) (rand() % 10- 5);
		data[i].y = (float) (rand() % 10- 5);
	}
}

void printmat(cuFloatComplex* data,int row,int col, char* title) {
	printf("%s\n",title);
	for (int i=0; i<row; ++i)
	{
		for (int j=0; j<col; ++j)
		{
			printf("%g+%gi\t",mem2d(data,col,i,j).x,mem2d(data,col,i,j).y);
		}
		printf("\n");
	}
	printf("......................\n");
}

void gpuKernel(cuFloatComplex* data, cuFloatComplex* out) {
	cuFloatComplex* voxel1d;
	cuFloatComplex* colData;
	cuFloatComplex* colDataFFT;
	cuFloatComplex Ant_position;
	cuFloatComplex* inp;
	//cuFloatComplex* dtpointer;
	//cuFloatComplex inp [Nz_padded*Nf];
	//memset (inp, 0, Nz_padded*Nf*sizeof(cuFloatComplex));
	//for (int i_f=0; i_f<Nf; i_f++)
	//{
	//	for (int i_z_Ant=0; i_z_Ant<Nz_padded; i_z_Ant++)
	//	{
	//		mem2d(inp,Nz_padded,i_f,i_z_Ant) = make_cuComplex((float) (rand() % 10- 5), (float) (rand() % 10- 5));
	//	}
	//}
	
	HANDLE_ERROR(cudaMalloc((void**)&voxel1d, Nz_padded*Ny*Nx * sizeof(cuFloatComplex)));
	HANDLE_ERROR(cudaMemset(voxel1d, 0, Nz_padded*Ny*Nx*sizeof(cuFloatComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&colData, Nf*Nz_padded* sizeof(cuFloatComplex)));
	HANDLE_ERROR(cudaMalloc((void**)&colDataFFT, Nf*Nz_padded* sizeof(cuFloatComplex)));
	
	dim3 dimBlock(Nx);
	dim3 dimGrid(Ny, Nz_padded);
	
	GpuTimer timer;
	timer.Start();
	//printf("%zd\n",sizeof(cuFloatComplex));
	for (int i_phi_Ant=0; i_phi_Ant<Nphi; i_phi_Ant++)
	{
		//printf("if:%d\n",i_phi_Ant);
		Ant_position = make_cuFloatComplex(R_ant*cos( i_phi_Ant * PI / 180.0 )+(Lx/2), R_ant*sin( i_phi_Ant * PI / 180.0 )+(Ly/2));
		//Ant_position = make_cuFloatComplex(R_ant*0.7, R_ant*0.7);
		inp = data + i_phi_Ant*Nz_padded*Nf;
		//HANDLE_ERROR(cudaMemcpyToSymbol(colData, inp, sizeof(cuFloatComplex)*Nz*Nf));
		HANDLE_ERROR(cudaMemcpy(colData, inp, Nz_padded*Nf*sizeof(cuFloatComplex), cudaMemcpyHostToDevice));
		cufftExecC2C(handle,  colData, colDataFFT, CUFFT_INVERSE);
		SARbp<<< dimGrid,dimBlock >>>(voxel1d, colDataFFT, Ant_position);
	}
	timer.Stop();
	float gpu_kernel_time = timer.Elapsed();
	printf("GPU Time:%f\n", gpu_kernel_time);

	HANDLE_ERROR(cudaMemcpy(out, voxel1d, Nz_padded*Ny*Nx * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));
	
	HANDLE_ERROR(cudaFree(voxel1d));
}

//////////////////////////CUDA\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

#define tx threadIdx.x
#define ty threadIdx.y
#define tz threadIdx.z

#define bx blockIdx.x
#define by blockIdx.y
#define bz blockIdx.z

#define X_idx tx
#define Y_idx bx
#define Z_idx by


#define bdx blockDim.x
#define bdy blockDim.y
#define bdz blockDim.z

#define gdx gridDim.x
#define gdy gridDim.y
#define gdz gridDim.z

#define NumThreads 1024
#define DegToRad (PI/180)
#define C 299792458
#define k_prefix (PI*2/C)
#define delta_k (PI*2*delta_f/C)
#define k_0 (k_prefix*(fstart))
#define Nz_padded_inv (1.0f/(Nz_padded-1))
#define k_z_max (PI/dz)
#define init_f_prefix (C/(PI*delta_f*4))
#define f_norm (fstart/delta_f)

#if Prec_SAR==1	
__global__ void SARbp(cuFloatComplex* voxel1d, cuFloatComplex* colData, cuFloatComplex Ant_position)
{	
	const float kz_arg_sqrt = fabsf(k_z_max*(Nz_padded_inv+(2*Nz_padded_inv)*Z_idx-(Z_idx/Nz)*(2*Nz_padded_inv)*(Nz_padded)));
	const float kz_arg = kz_arg_sqrt*kz_arg_sqrt;
	const float distXY = ((Ant_position.x-X_idx*dx)*(Ant_position.x-X_idx*dx) + (Ant_position.y-Y_idx*dy)*(Ant_position.y-Y_idx*dy));
	float exp_arg;
	
	cuFloatComplex voxel, data;
	cuFloatComplex exp_coef;
	
	unsigned long int index = tx + bx*bdx + by*gdx*bdx;
	voxel = make_cuComplex(0,0);

	unsigned int i_f_start = ceilf(init_f_prefix*kz_arg_sqrt - f_norm);
	float f = fstart + i_f_start*delta_f;
	for (int i_f=i_f_start; i_f<Nf; i_f++)
	{
		exp_arg = sqrtf(distXY*(k_prefix*k_prefix*f*f*4 - kz_arg));
		__sincosf(exp_arg, &(exp_coef.y), &(exp_coef.x));
		data = mem2d(colData,Nz_padded,i_f,Z_idx);
		voxel = make_cuComplex(voxel.x + data.x*exp_coef.x - data.y*exp_coef.y, voxel.y + data.x*exp_coef.y + data.y*exp_coef.x);
		f = f+delta_f;
	}
	//voxel1d[index] = voxel;
	atomicAdd(&voxel1d[index].x, voxel.x);
	atomicAdd(&voxel1d[index].y, voxel.y);
}
#else
__global__ void SARbp(cuFloatComplex* voxel1d, cuFloatComplex* colData, cuFloatComplex Ant_position)
{	
	const float kz_arg = (Nz_padded_inv+(2*Nz_padded_inv)*Z_idx-(Z_idx/Nz)*(2*Nz_padded_inv)*(Nz_padded));
	const float dist_kz = sqrt(((Ant_position.x-X_idx*dx)*(Ant_position.x-X_idx*dx) + (Ant_position.y-Y_idx*dy)*(Ant_position.y-Y_idx*dy)) * (4-kz_arg*kz_arg));
	float exp_arg, delta_exp_arg;
	

	cuFloatComplex exp_coef, delta_exp_coef;
	cuFloatComplex voxel, data;
	
	unsigned long int index = tx + bx*bdx + by*gdx*bdx;
	voxel = make_cuComplex(0,0);

	exp_arg = k_0*dist_kz;
	delta_exp_arg = delta_k*dist_kz;
	__sincosf(exp_arg, &(exp_coef.y), &(exp_coef.x));
	__sincosf(delta_exp_arg, &(delta_exp_coef.y), &(delta_exp_coef.x));
	for (int i_f=0; i_f<Nf; i_f++)
	{
		data = mem2d(colData,Nz_padded,i_f,Z_idx);
		voxel = make_cuComplex(voxel.x + data.x*exp_coef.x - data.y*exp_coef.y, voxel.y + data.x*exp_coef.y + data.y*exp_coef.x);
		exp_coef = cuCmulf(exp_coef, delta_exp_coef);
	}
	//voxel1d[index] = voxel;
	atomicAdd(&voxel1d[index].x, voxel.x);
	atomicAdd(&voxel1d[index].y, voxel.y);
}
#endif		
