#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <math.h>

#define WIDTH 1920        
#define HEIGHT 1080      
#define MAX_ITER 5000    

global void computeMandelbrot(double xmin, double xmax, double ymin, double ymax, int width, int height, unsigned char *output) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height) {
        double real = xmin + (xmax - xmin) * x / width;
        double imag = ymin + (ymax - ymin) * y / height;

        double z_real = 0.0, z_imag = 0.0;
        int n = 0;

        while (z_real * z_real + z_imag * z_imag <= 4.0 && n < MAX_ITER) {
            double temp_real = z_real * z_real - z_imag * z_imag + real;
            z_imag = 2.0 * z_real * z_imag + imag;
            z_real = temp_real;
            n++;
        }

        double log_zn = logf(z_real * z_real + z_imag * z_imag) / 2.0f; 
        double nu = logf(log_zn / logf(2.0f)) / logf(2.0f);            
        double iter_smooth = n + 1 - nu;

        int color = (int)(255.0 * iter_smooth / MAX_ITER);
        int idx = 3 * (y * width + x); // R, G, B for each pixel
        output[idx] = color;          // Red
        output[idx + 1] = (color * 5) % 255; // Green
        output[idx + 2] = (color * 10) % 255; // Blue
    }
}

void savePPM(const char *filename, unsigned char *data, int width, int height) {
    FILE *fp = fopen(filename, "wb");
    fprintf(fp, "P6\n%d %d\n255\n", width, height); 
    fwrite(data, sizeof(unsigned char), width * height * 3, fp);
    fclose(fp);
}

int main() {
    const int numFrames = 600;     // Total number of frames are set to 600, more can be made for a longer animation
    const double zoomFactor = 1.02; // Zoom speed
    const char *outputDir = "frames";

    // Starting region (Seahorse Valley)
    double centerX = -0.743643887037151;
    double centerY = 0.13182590420533;
    double scale = 4.0; 

    system("mkdir -p frames");

    unsigned char *h_output = (unsigned char *)malloc(WIDTH * HEIGHT * 3);
    unsigned char *d_output;
    cudaMalloc((void **)&d_output, WIDTH * HEIGHT * 3);

    dim3 threadsPerBlock(16, 16);
    dim3 numBlocks((WIDTH + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (HEIGHT + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Generating Mandelbrot zoom animation...\n");

    for (int frame = 0; frame < numFrames; frame++) {
        double xmin = centerX - scale / 2;
        double xmax = centerX + scale / 2;
        double ymin = centerY - scale / 2;
        double ymax = centerY + scale / 2;

        computeMandelbrot<<<numBlocks, threadsPerBlock>>>(xmin, xmax, ymin, ymax, WIDTH, HEIGHT, d_output);
        cudaDeviceSynchronize();

        cudaMemcpy(h_output, d_output, WIDTH * HEIGHT * 3, cudaMemcpyDeviceToHost);

        char filename[256];
        sprintf(filename, "%s/frame_%04d.ppm", outputDir, frame);
        savePPM(filename, h_output, WIDTH, HEIGHT);

        scale /= zoomFactor;

        printf("Frame %d/%d saved to %s\n", frame + 1, numFrames, filename);
    }

    free(h_output);
    cudaFree(d_output);

    printf("All frames generated. Use ffmpeg to create a video.\n");
    return 0;
}