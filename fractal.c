
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

int main() {
    int image[HEIGHT][WIDTH];

    double x_min = -2.0, x_max = 1.0;
    double y_min = -1.5, y_max = 1.5;

    // Generate the Mandelbrot set
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            double real = x_min + j * (x_max - x_min) / WIDTH;
            double imag = y_min + i * (y_max - y_min) / HEIGHT;
            image[i][j] = mandelbrot(real, imag);
        }
    }

    FILE *fp = fopen("mandelbrot.pgm", "w");
    fprintf(fp, "P2\n%d %d\n%d\n", WIDTH, HEIGHT, MAX_ITER);
    for (int i = 0; i < HEIGHT; i++) {
        for (int j = 0; j < WIDTH; j++) {
            fprintf(fp, "%d ", image[i][j]);
        }
        fprintf(fp, "\n");
    }
    fclose(fp);
    return 0;
}
