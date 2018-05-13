/**
 *  Program MPI
 *  Perhitungan histogram secara paralel
 *  Menggunakan MPI_Bcast, MPI_Scatterv, MPI_Reduce
 *    
 *  Nama: Steven Albert
 *  NIM: 00000011011
 */

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

// Defined constants
#define TOTAL_R 256
#define TOTAL_G 256
#define TOTAL_B 256
#define TOTAL_COLOR (TOTAL_R * TOTAL_G * TOTAL_B)

typedef int bgrColor;
typedef struct bgrImage {
	int 		_rows; // Number of rows in the image
	int 		_cols; // Number of cols in the image
	bgrColor	*_colors; // Color values in the image
} bgrImage;
typedef struct histogram {
	int			*_n; // Total color {i} is distributed
	int 		_size; // Size of histogram / total colors
} histogram;

// Color functions
int blue(bgrColor c);
int green(bgrColor c);
int red(bgrColor c);
bgrColor create_color(int b, int g, int r);

// Image functions
bgrColor color(const bgrImage img, int row, int col);
histogram bgr_histogram(const bgrColor *colors, const int n);
histogram b_histogram(const bgrColor *colors, const int n);
histogram g_histogram(const bgrColor *colors, const int n);
histogram r_histogram(const bgrColor *colors, const int n);
void free_image(bgrImage *img);
void free_histogram(histogram *hist);

// Output function
void write_to_file_fc(histogram histogram_result, const char *name);
void write_to_file_c(histogram histogram_result, const char *name, const char c);

int main(int argc, char **argv) {
	MPI_Init(NULL, NULL);

	int world_size, world_rank;
	MPI_Comm_size(MPI_COMM_WORLD, &world_size);
	MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

	// Read input from file only from process with rank 0
	bgrColor *image = NULL;
	int total_pixels;
	int *size_per_proc, *displs, remaining_pixels, sum = 0;
	double time;

	if(world_rank == 0) {
		if(argc != 2) {
			// Check application argument
			fprintf(stderr, "Usage %s file_path\n", argv[0]);
			MPI_Abort(MPI_COMM_WORLD, 1);
		}
		// Start timer for read input file
		time = -MPI_Wtime();
		// Start read input file
		FILE *f = fopen(argv[1], "r");
		assert(f != NULL);
		int r, g, b, n, m;
		fscanf(f, "%d %d", &n, &m);
		total_pixels = n * m / 3;
		image = (bgrColor *)malloc(total_pixels * sizeof(bgrColor));
		assert(image != NULL);
		for(int i=0; i<total_pixels; i++) {
			fscanf(f, "%d %d %d", &b, &g, &r);
			image[i] = create_color(b, g, r);
		}
		fclose(f);

		// Finish timer for read input file
		time += MPI_Wtime();
		printf("## START ##   Number of procs: %d\n", world_size);
		printf("(Master) Read data in %lf second(s)\n", time);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	// Start timer for send and receive all information from input
	time = -MPI_Wtime();
	// Start broadcast the data for processing
	size_per_proc = (int*)malloc(sizeof(int) * world_size);
	displs = (int*)malloc(sizeof(int) * world_size);
	MPI_Bcast(&total_pixels, 1, MPI_INT, 0, MPI_COMM_WORLD);
	// Calculate number of pixels each processor get
	remaining_pixels = total_pixels % world_size;
	for(int i=0; i<world_size; i++) {
		size_per_proc[i] = total_pixels / world_size;
		if(remaining_pixels > 0) {
			size_per_proc[i]++;
			remaining_pixels--;
		}

		displs[i] = sum;
		sum += size_per_proc[i];
	}

	// Scatter image to each process (MPI_Scatterv)
	bgrColor *sub_image = NULL;
	sub_image = (bgrColor*)malloc(sizeof(bgrColor) * size_per_proc[world_rank]);
	assert(sub_image != NULL);

	MPI_Scatterv(image, size_per_proc, displs, MPI_INT, sub_image, size_per_proc[world_rank], MPI_INT, 0, MPI_COMM_WORLD);

	MPI_Barrier(MPI_COMM_WORLD);
	// Finish timer for send and receive all information from input
	time += MPI_Wtime();
	if(world_rank == 0) printf("(Master) Send to all processes in %lf second(s)\n", time);
	else printf("(%d/%d) Receive all information needed in %lf second(s)\n", world_rank, world_size, time);

	MPI_Barrier(MPI_COMM_WORLD);
	// Start timer for calculating histogram of sub-image in each processor
	time = -MPI_Wtime();
	// Start calculating histogram of sub-image
	histogram sub_histogram = b_histogram(sub_image, size_per_proc[world_rank]); /* Uncomment to get blue histogram */
//	histogram sub_histogram = g_histogram(sub_image, size_per_proc[world_rank]); /* Uncomment to get green histogram */
//	histogram sub_histogram = r_histogram(sub_image, size_per_proc[world_rank]); /* Uncomment to get red histogram */
//	histogram sub_histogram = bgr_histogram(sub_image, size_per_proc[world_rank]); /* Uncomment to get color histogram */
	// Finish timer for calculating histogram of sub-image in each proessor
	time += MPI_Wtime();
	printf("(%d/%d) Calculate sub histogram in %lf second(s)\n", world_rank, world_size, time);

	MPI_Barrier(MPI_COMM_WORLD);
	// Start timer for collecting and combining all histograms of sub-image
	time = -MPI_Wtime();

	int *hist_n_procs = NULL;
	if(world_rank == 0) {
		hist_n_procs = (int*)malloc(sizeof(int) * TOTAL_B); /* Uncomment to get blue histogram */
//		hist_n_procs = (int*)malloc(sizeof(int) * TOTAL_G); /* Uncomment to get green histogram */
//		hist_n_procs = (int*)malloc(sizeof(int) * TOTAL_R); /* Uncomment to get red histogram */
//		hist_n_procs = (int*)malloc(sizeof(int) * TOTAL_COLOR); /* Uncomment to get color histogram */
		assert(hist_n_procs != NULL);
	}
	// Start collecting and summing histograms of sub-image
	MPI_Reduce(sub_histogram._n, hist_n_procs, TOTAL_B, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); /* Uncomment to get blue histogram */
//	MPI_Reduce(sub_histogram._n, hist_n_procs, TOTAL_G, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); /* Uncomment to get green histogram */
//	MPI_Reduce(sub_histogram._n, hist_n_procs, TOTAL_R, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); /* Uncomment to get red histogram */
//	MPI_Reduce(sub_histogram._n, hist_n_procs, TOTAL_COLOR, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD); /* Uncomment to get color histogram */

	MPI_Barrier(MPI_COMM_WORLD);
	// Finish timer for collecting and combining all histograms of sub-image
	time += MPI_Wtime();

	if(world_rank == 0) {
		printf("(Master) Reduce done in %lf second(s)\n", time);
	}

	MPI_Barrier(MPI_COMM_WORLD);
	if(world_rank == 0) {
		// Start timer for writing histogram result to a file
		time = -MPI_Wtime();
		histogram result = {hist_n_procs, TOTAL_B}; /* Uncomment to get blue histogram */
//		histogram result = {hist_n_procs, TOTAL_G}; /* Uncomment to get green histogram */
//		histogram result = {hist_n_procs, TOTAL_R}; /* Uncomment to get red histogram */
//		histogram result = {hist_n_procs, TOTAL_COLOR}; /* Uncomment to get color histogram */
		char filename[50];
		strcpy(filename, "result/");
		strcat(filename, strrchr(argv[1], '/'));
		strcat(filename, "-b-histogram-mpi.csv");
		printf("(Master) Write to file, filename: %s\n", filename); 
		write_to_file_c(result, filename, 'B'); /* Uncomment to get blue histogram */
//		write_to_file_c(result, filename, 'G'); /* Uncomment to get green histogram */
//		write_to_file_c(result, filename, 'R'); /* Uncomment to get red histogram */
//		write_to_file_fc(result, filename); /* Uncomment to get color histogram */
		// Finish timer for writing histogram result to a file
		time += MPI_Wtime();
		printf("(Master) Write histogram in %lf second(s)\n", time);
	}
	MPI_Barrier(MPI_COMM_WORLD);

	free(sub_image);
	free_histogram(&sub_histogram);
	if(world_rank == 0) {
		free(hist_n_procs);
		free(image);
	}
	MPI_Finalize();

	return 0;
}

int blue(bgrColor c) {
	return c >> 16;
}

int green(bgrColor c) {
	return ((c & 65280 /*0...01111111100000000*/) >> 8);
}

int red(bgrColor c) {
	return c & 255;
}

bgrColor create_color(int b, int g, int r) {
	return (b << 16) ^ (g << 8) ^ r;
}

bgrColor color(const bgrImage img, int row, int col) {
	return img._colors[row * img._cols + col];
}

histogram bgr_histogram(const bgrColor *colors, const int n) {
	int *histo = (int *)malloc(TOTAL_COLOR * sizeof(int));
	for(int i=0; i<TOTAL_COLOR; i++) histo[i] = 0;
	
	for(int i=0; i<n; i++) {
		histo[colors[i]]++;
	}

	histogram _histogram = {histo, TOTAL_COLOR};
	return _histogram; 
}

histogram b_histogram(const bgrColor *colors, const int n){
	int *histo = (int *)malloc(TOTAL_B * sizeof(int));
	for(int i=0; i<TOTAL_B; i++) histo[i] = 0;

	for(int i=0; i<n; i++) {
		histo[blue(colors[i])]++;
	}
	
	histogram _histogram = {histo, TOTAL_B};
	return _histogram; 
}

histogram g_histogram(const bgrColor *colors, const int n){
	int *histo = (int *)malloc(TOTAL_G * sizeof(int));
	for(int i=0; i<TOTAL_G; i++) histo[i] = 0;
	
	for(int i=0; i<n; i++) {
		histo[green(colors[i])]++;
	}

	histogram _histogram = {histo, TOTAL_G};
	return _histogram; 
}

histogram r_histogram(const bgrColor *colors, const int n){
	int *histo = (int *)malloc(TOTAL_R * sizeof(int));
	for(int i=0; i<TOTAL_R; i++) histo[i] = 0;
	
	for(int i=0; i<n; i++) {
		histo[red(colors[i])]++;
	}

	histogram _histogram = {histo, TOTAL_R};
	return _histogram; 
}

void free_image(bgrImage *img) {
	free((*img)._colors);
}

void free_histogram(histogram *hist) {
	free((*hist)._n);
}

void write_to_file_fc(histogram histogram_result, const char *name) {
	FILE *f = fopen(name, "w");
	assert(f != NULL);
	int size = histogram_result._size;
	fprintf(f, "B,G,R,n\n");
	for(int col=0; col < size; col++) {
		if(histogram_result._n[col] == 0) continue;
		fprintf(f, "%d,%d,%d,%d\n", blue(col), green(col), red(col), histogram_result._n[col]);
	}
	fclose(f);
}

void write_to_file_c(histogram histogram_result, const char *name, const char c) {
	FILE *f = fopen(name, "w");
	assert(f != NULL);
	int size = histogram_result._size;
	fprintf(f, "%c,n\n", c);
	for(int col=0; col < size; col++) {
		if(histogram_result._n[col] == 0) continue;
		fprintf(f, "%d,%d\n", col, histogram_result._n[col]);
	}
	fclose(f);
}

