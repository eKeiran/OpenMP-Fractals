
#include <stdio.h>
#include <stdlib.h>

#define WIDTH 800
#define HEIGHT 800
#define MAX_ITER 1000

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

//Julia set function (f(z) = z^2 + c)
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
int main() {
       int mandelbrot_img[HEIGHT][WIDTH];
    int julia_img[HEIGHT][WIDTH];

    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

    // Generate the Mandelbrot set
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            double real = x_min + j * (x_max - x_min) / WIDTH;
            double imag = y_min + i * (y_max - y_min) / HEIGHT;
            mandelbrot_img[i][j] = mandelbrot(real, imag);
        }
    }

    save_pgm(mandelbrot_img, "mandelbrot.pgm");
    printf("Mandelbrot set generated and saved as mandelbrot.pgm\n");

    // Generate the Julia set
    double c_real = -0.7, c_imag = 0.27015;  // Julia constant

    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            double real = x_min + j * (x_max - x_min) / WIDTH;
            double imag = y_min + i * (y_max - y_min) / HEIGHT;
            julia_img[i][j] = julia(real, imag, c_real, c_imag);
        }
    }

    save_pgm(julia_img, "julia.pgm");
    printf("Julia set generated and saved as julia.pgm\n");

    return 0;
}
