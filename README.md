# Parallel Histogram
Calculates the distribution of color value from data of an image

The distribution is calculated per exact color value.
Color with B,G,R value *(0,0,0)* is different with *(0,0,1)*.
If the image has 2 pixel with color *(0,0,0)*, then the value of color *(0,0,0)* is 2.

## Serial
Calculate distribution of colors in serial.

## Parallel
### OpenMP

Calculate distribution in parallel with `omp.h` (OpenMP library in C/C++)

`omp_get_wtime` - to calculate execution time

`#pragma omp parallel for` - to parallelize the for loop

`#pragma omp atomic` - to let the processor know that this operation is atomic (only one thread can get access to its content at a time)

### OpenMPI

Calculate distribution in parallel with `mpi.h` (OpenMPI library in C/C++)

`MPI_Wtime` - to calculate execution time

`MPI_Bcast` - to broadcast the size of image

`MPI_Scatterv` - to give each processor its scattered data to be processed, each processor get relatively equal amount of data

`MPI_Reduce` - to collect calculation from each processor and accumulate them

Added three calculation in OpenMPI: 
- distribution calculation of blue color value, 
- distribution calculation of green color value, 
- distribution calculation of red color value
