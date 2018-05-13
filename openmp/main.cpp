#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <utility>
#include <vector>

// Color Structure
struct BGRColor {
	static const int TOTAL_POSSIBLE_COLORS = (1 << 24); 	// Constant value of total possible BGR color
	int colorValue;						// Color value saved as an integer
	// CONSTRUCTOR
	BGRColor(int color) { colorValue = color; }
	BGRColor(int blue, int green, int red) { colorValue = (blue << 16) ^ (green << 8) ^ red; }
	~BGRColor() {}									// DESTRUCTOR
	int getValue() { return colorValue; }						// Get color value
	int getBlue() { return (colorValue >> 16); }					// Get blue color value
	int getGreen() { return ((colorValue >> 8) ^ (getBlue() << 8)); }		// Get green color value
	int getRed() { return (colorValue ^ (getBlue() << 16) ^ (getGreen() << 8)); }	// Get red color value
};

// Image Structure
struct BGRImage {
	long long nRows; 	// Number of rows in the image
	long long nCols; 	// Number of cols in the image
	BGRColor *colors; 	// Color values in the image
	// CONSTRUCTOR
	BGRImage(int *bgrColors, long long cols, long long rows) {
		nRows = rows;
		nCols = cols;
		long long size = nRows * nCols;
		colors = (BGRColor *) malloc (size * sizeof(BGRColor));
		for(long long i = 0; i < size; i++) { colors[i] = BGRColor(bgrColors[3 * i], bgrColors[3 * i + 1], bgrColors[3 * i + 2]); }
	}
	~BGRImage() { free(colors); }			// DESTRUCTOR
	static const int INVALID_BLOCK_SIZE = -1; 	// BLOCK SIZE cannot fully divide nRows and nCols
	static const int INVALID_ROW_NUMBER = -2; 	// ROW NUMBER is not in range of BLOCK SIZE
	static const int INVALID_COL_NUMBER = -3;	// COL NUMBER is not in range of BLOCK SIZE
	static const int INVALID_BLOCK_NUMBER = -4;	// BLOCK NUMBER is not in range of TOTAL BLOCK
	// Print information
	void print() { printf("Number of rows = %lld | Number of columns = %lld\n", nRows, nCols); }
	// _block_size = n --> size of block is n x n.
	// _block --> block number
	// _row --> row number relative to its block
	// _col --> col number relative to its block
	int getPixel(int _block_size, int _block, int _col, int _row) {
		if(nCols % _block_size || nRows % _block_size) return INVALID_BLOCK_SIZE;
		if(_row < 0 || _row >= _block_size) return INVALID_ROW_NUMBER;
		if(_col < 0 || _col >= _block_size) return INVALID_COL_NUMBER;
		int numOfBlockInRow = nCols / _block_size;
		int numOfBlockInCol = nRows / _block_size;
		if(_block < 0 || _block >= numOfBlockInRow * numOfBlockInCol) return INVALID_BLOCK_NUMBER;

		int blockColOffset = (_block % numOfBlockInCol) * _block_size;
		int blockRowOffset = (_block / numOfBlockInCol) * _block_size;

		return (blockRowOffset + _row) * nCols + (blockColOffset + _col);
	}
	BGRColor getColor(int pixel) { return colors[pixel]; }			// Get BGRColor of the pixel
	// Calculate Histogram in parallel
	std::vector< std::pair<BGRColor, int> > *calculateHistogramParallel(int n_threads) {
		// Create histogram array
		int *histogramArray = (int *) malloc (BGRColor::TOTAL_POSSIBLE_COLORS * sizeof(int));
		long long i, j;
		#pragma omp parallel for num_threads(n_threads) private(i)
		for(i = 0; i < BGRColor::TOTAL_POSSIBLE_COLORS; i++) { histogramArray[i] = 0; }
		// Add colors to histograms
		long long _size = nRows * nCols;
		#pragma omp parallel for num_threads(n_threads) private(i)
		for(i = 0; i < _size; i++) { 
			#pragma omp atomic
			histogramArray[getColor(i).getValue()]++;
		}
		// Populate the return value
		std::vector< std::pair<BGRColor, int> > *histogram = new std::vector< std::pair<BGRColor, int> >();
		for(i = 0; i < BGRColor::TOTAL_POSSIBLE_COLORS; i++) 
		{ if(histogramArray[i]) histogram->push_back(std::make_pair(BGRColor(i), histogramArray[i])); }
		// Free the histogram array
		free(histogramArray);
		// Return
		return histogram;
	}
	// Calculate Block Histogram in parallel
	std::vector< std::pair<BGRColor, int> > *calculateBlockHistogramParallel(int n_threads, int b_size, int _block) {
		// Check block size
		if(nRows % b_size != 0 || nCols % b_size != 0) return new std::vector< std::pair<BGRColor, int> >();
		// Create histogram array
		int *histogramArray = (int *) malloc (BGRColor::TOTAL_POSSIBLE_COLORS * sizeof(int));
		long long i, j;
		#pragma omp parallel for num_threads(n_threads) private(i)
		for(i = 0; i < BGRColor::TOTAL_POSSIBLE_COLORS; i++) { histogramArray[i] = 0; }
		// Add colors to histogram
		// i = row, j = col
		#pragma omp parallel for num_threads(n_threads) private(i, j)
		for(i = 0; i < b_size; i++) { 
			for(j = 0; j < b_size; j++) {
				#pragma omp atomic
				histogramArray[getColor(getPixel(b_size, _block, i, j)).getValue()]++; 
			}
		}
		// Populate the return value
		std::vector< std::pair<BGRColor, int> > *histogram = new std::vector< std::pair<BGRColor, int> >();
		for(i = 0; i < BGRColor::TOTAL_POSSIBLE_COLORS; i++) 
		{ if(histogramArray[i]) histogram->push_back(std::make_pair(BGRColor(i), histogramArray[i])); }
		// Free the histogram array
		free(histogramArray);
		// Return
		return histogram;
	}
};

// Function declarations
void printTime(double start, double end);
void writeToFile(std::vector< std::pair<BGRColor, int> > &hist, const char *name);
// Constants
const int NUM_THREADS = 8;

// Main function
int main(int argc, char *argv[]) {

	// File descriptor for read and write
	FILE *colorDataFd;
	// x - image width * 3 (BGR format)
	// y - image height
	// total_input - x * y | colorData - read input array
	long long x, y, total_input;
	int *colorData;
	// startTime, endTime - get time at runtime
	double startTime, endTime;
	// Image variables
	BGRImage *image;

	///////////
	// INPUT //
	///////////

	// Read file
	printf("File input: %s\n", argv[1]);
	colorDataFd = fopen(argv[1], "r");	
	// Get the total input
	fscanf(colorDataFd, "%lld %lld", &x, &y);
	// Get the colors (B, G, R)
	total_input = x * y;
	colorData = (int *) malloc (total_input * sizeof(int));
	for(long long i=0; i<total_input; i++) fscanf(colorDataFd, "%d", &colorData[i]);
	// Close file
	fclose(colorDataFd);
	// Construct BGRImage
	image = new BGRImage(colorData, x / 3, y);
	// Free color data input
	free(colorData);
	// Print image information
	image->print();

	/////////////////////////
	// PARALLELIZE PROCESS //
	/////////////////////////
	printf("## PARALLELIZE PROCESS  --- ");
	// Get start time
	startTime = omp_get_wtime();
	// Calculate Histogram parallelize
	// Histogram --> calculateHistogramParallel(n_thread)
	// Block Histogram --> calculateHistogramParallel(n_thread, b_size, _block)
	std::vector< std::pair<BGRColor, int> > *parallelHistogram = image->calculateHistogramParallel(NUM_THREADS);
	//std::vector< std::pair<BGRColor, int> > *parallelHistogram = image->calculateBlockHistogramParallel(NUM_THREADS, 8, 0);
	// Get end time
	endTime = omp_get_wtime();

	////////////////////////
	// PARALLELIZE OUTPUT //
	////////////////////////

	// Print in command line
	printTime(startTime, endTime);
	// Write result to file
	writeToFile(*parallelHistogram, "histogram-parallel");
	// Delete vector
	delete parallelHistogram;
	// Delete image
	delete image;

	return 0;
}

// Print Elapsed Time
void printTime(double start, double end) { printf("Elapsed time: %lf\n", end - start); }

// Write histogram result to file named *name*
void writeToFile(std::vector< std::pair<BGRColor, int> > &hist, const char *name) {
	// Open file
	FILE *colorDataFd = fopen(name, "w");
	fprintf(colorDataFd, "B,G,R,N\n");
	std::vector< std::pair<BGRColor, int> >::iterator it;
	int b, g, r, count;
	for(it = hist.begin(); it != hist.end(); it++) {
		b = it->first.getBlue();
		g = it->first.getGreen();
		r = it->first.getRed();
		count = it->second;
		fprintf(colorDataFd, "%d,%d,%d,%d\n", b, g, r, count);
	}
	// Close written file
	fclose(colorDataFd);
}

