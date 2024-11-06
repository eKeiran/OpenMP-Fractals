# Fractal Visualization with Mandelbrot and Julia Sets (a micro-project for High Performance Computing class)

## This project generates **Mandelbrot** and **Julia** fractals, showcasing self-similarity through iterative processes utilizing OpenMP parallelization.

### What are Fractals?
Fractals are complex shapes that exhibit self-similarity at every scale. The **Mandelbrot set** and **Julia set** are examples of fractals, created by iterating functions over complex numbers.

### Mandelbrot & Julia Sets
- **Mandelbrot Set**: Iterates `z = z^2 + c` to generate intricate patterns for complex numbers `c`.
- **Julia Set**: Similar to the Mandelbrot set but with a fixed constant `c`, producing different patterns depending on `c`.

### Performance with OpenMP
Using **OpenMP**, we parallelize fractal generation, reducing computation time:
- **Serial**: ~21 seconds
- **Parallel (OpenMP)**: ~7 seconds (3x faster)

### Scaling with HPC

This project benefits from High Performance Computing (HPC) techniques, enabling the fractal generation process to scale as the size of the image and the complexity of the fractal increases. With OpenMP parallelization, the computation load is distributed across multiple cores, reducing execution time and enabling faster generation of high-resolution fractals.


## Instructions

1. **Clone the Repository**:
   Clone the project repository to your local machine using the following command:
   ```bash
   git clone https://github.com/eKeiran/OpenMP-Fractals
   cd OpenMP-Fractals
   ```

2. **Install Dependencies**:
   Ensure you have a C compiler like GCC and OpenMP installed.

3. **Compile the Code**:
   Use the following command to compile the C code with OpenMP support:
   ```bash
   gcc -fopenmp -o fractal fractal.c
   ```

4. **Run the Code**:
   After compilation, run the fractal generation program:
   ```bash
   ./fractal
   ```

5. **Optional - Measure Execution Time**:
   You can check the time it takes to run the program using the `time` command:
   ```bash
   time ./fractal
   ```

6. **View the Output**:
   The program generates two colored fractal images and two grayscale ones which can all be viewed through GIMP or feh:
   - `mandelbrot_color.ppm`
   - `julia_color.ppm`
   - `mandelbrot.pgm`
   - `julia.pgm`



