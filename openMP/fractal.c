#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 50000

//mapping the number of iterations to a color
void get_color(int n, int *r, int *g, int *b) {
    if (n == MAX_ITER) {
        *r = *g = *b = 0; // black chosen for points inside the set
    } else {
        *r = (n % 256);
        *g = (n * 5) % 256;
        *b = (n * 10) % 256;
    }
}
//mandelbrot set mathematical function
int mandelbrot(double real, double imag) {
    double z_real = 0, z_imag = 0;
    int n = 0;
    while ((z_real * z_real + z_imag * z_imag <= 4) && (n < MAX_ITER)) {
        double temp_real = z_real * z_real - z_imag * z_imag + real;
        z_imag = 2 * z_real * z_imag + imag;
        z_real = temp_real;
        n++;
    }
    return n;
}
//julia set mathematical function
int julia(double real, double imag, double c_real, double c_imag) {
    double z_real = real, z_imag = imag;
    int n = 0;
    while ((z_real * z_real + z_imag * z_imag <= 4) && (n < MAX_ITER)) {
        double temp_real = z_real * z_real - z_imag * z_imag + c_real;
        z_imag = 2 * z_real * z_imag + c_imag;
        z_real = temp_real;
        n++;
    }
    return n;
}

void save_pgm(int image[HEIGHT][WIDTH], const char *filename) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "P2\n%d %d\n%d\n", WIDTH, HEIGHT, MAX_ITER);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(fp, "%d ", image[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

void save_ppm(int image[HEIGHT][WIDTH][3], const char *filename) {
    FILE *fp = fopen(filename, "w");
    fprintf(fp, "P3\n%d %d\n255\n", WIDTH, HEIGHT); // PPM header
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(fp, "%d %d %d ", image[i][j][0], image[i][j][1], image[i][j][2]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
}

int main() {
    int (*mandelbrot_img)[WIDTH] = malloc(HEIGHT * sizeof(*mandelbrot_img));  
    int (*julia_img)[WIDTH] = malloc(HEIGHT * sizeof(*julia_img));  
    int (*mandelbrot_img_colored)[WIDTH][3] = malloc(HEIGHT * sizeof(*mandelbrot_img_colored));  
    int (*julia_img_colored)[WIDTH][3] = malloc(HEIGHT * sizeof(*julia_img_colored));  

    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

    printf("Generating Mandelbrot set with parallelization...\n");
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            double real = x_min + j * (x_max - x_min) / WIDTH;
            double imag = y_min + i * (y_max - y_min) / HEIGHT;
            int n = mandelbrot(real, imag);
            mandelbrot_img[i][j] = n;
            get_color(n, &mandelbrot_img_colored[i][j][0], &mandelbrot_img_colored[i][j][1], &mandelbrot_img_colored[i][j][2]);
        }
    }

    save_pgm(mandelbrot_img, "mandelbrot.pgm");
    save_ppm(mandelbrot_img_colored, "mandelbrot_color.ppm");
    printf("Mandelbrot set generated and saved as mandelbrot.pgm and mandelbrot_color.ppm\n");

    double c_real = -0.7, c_imag = 0.27015;  // using julia constant
    printf("Generating Julia set with parallelizatin...\n");
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            double real = x_min + j * (x_max - x_min) / WIDTH;
            double imag = y_min + i * (y_max - y_min) / HEIGHT;
            int n = julia(real, imag, c_real, c_imag);
            julia_img[i][j] = n;
            get_color(n, &julia_img_colored[i][j][0], &julia_img_colored[i][j][1], &julia_img_colored[i][j][2]);
        }
    }

    save_pgm(julia_img, "julia.pgm");
    save_ppm(julia_img_colored, "julia_color.ppm");
    printf("Julia set generated and saved as julia.pgm and julia_color.ppm\n");

    free(mandelbrot_img);
    free(julia_img);
    free(mandelbrot_img_colored);
    free(julia_img_colored);

    return 0;
}
