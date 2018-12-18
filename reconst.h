#ifndef GPU_H
#define GPU_H
#include "cuComplex.h"
#define Nf 13
#define fstart 33.5e9f
#define delta_f 250e6f
#define Nphi 91
//#define Nphi 1
#define Nz 81
#define Nz_padded (2*Nz)
#define Nx 201
#define Ny 201
#define dz 0.004f
#define dx 0.005f
#define dy 0.005f
#define Lz (Nz-1)*dz
#define Ly (Ny-1)*dy
#define Lx (Nx-1)*dx
#define R_ant 0.5f
#define Prec_SAR 1


#define mem2d(data,q,y,x)   data[((y)*(q))+(x)]
__global__ void SARbp(cuFloatComplex* voxeleld, cuFloatComplex* colData, cuFloatComplex Ant_position);
#endif
