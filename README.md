# Fractal Visualization with Mandelbrot and Julia Sets (a micro-project for High Performance Computing class)

## This project generates **Mandelbrot** and **Julia** fractals, showcasing self-similarity through iterative processes by OpenMP parallelization for the first assessment. 
![image](https://github.com/user-attachments/assets/40da792a-778a-4465-940d-06708ca6be73)
![image](https://github.com/user-attachments/assets/9d5b9664-9849-48eb-aa6f-33101ae79fb5)
## Then, it works on zooming into fractals by using CUDA as well as creating an animation of the Mandelbrot set for the second assessment. 
![mandelbrot_zoom](https://github.com/user-attachments/assets/69ccd097-1524-4707-802b-839f6ed5190c)


### What are Fractals?
Fractals are complex shapes that exhibit self-similarity at every scale. The **Mandelbrot set** and **Julia set** are examples of fractals, created by iterating functions over complex numbers.

### Mandelbrot & Julia Sets
- **Mandelbrot Set**: Iterates `z = z^2 + c` to generate intricate patterns for complex numbers `c`.
- **Julia Set**: Similar to the Mandelbrot set but with a fixed constant `c`, producing different patterns depending on `c`.

### I. OpenMP folder 
Using **OpenMP**, we parallelize fractal generation, reducing computation time:
### - **Serial**: ~21 seconds
![image](https://github.com/user-attachments/assets/15e04be1-c4a1-4106-b6d3-ef6d02b31eb1)

### - **Parallel (OpenMP)**: ~7 seconds (3x faster)
![image](https://github.com/user-attachments/assets/1d01ca58-e7c5-4aab-9283-ba78a0af862e)

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
### II. Cuda Folder: Mandelbrot Zoom Animation

The CUDA code utilizes GPU parallelization to compute the Mandelbrot fractal for multiple frames, creating a smooth zoom effect.

1. **Initial Parameters**: The zoom starts from the **Seahorse Valley** of the Mandelbrot set.
   - **Starting region**: Center at `(-0.743643887037151, 0.13182590420533)` 
   - **Scale**: Initial zoom level is set to `4.0`.

2. **Parallel Computation**: Using CUDA's `computeMandelbrot` kernel, each pixel in the image is computed in parallel across the GPU. This involves iterating the Mandelbrot function for each pixel's complex number and determining the number of iterations before divergence.

3. **Zooming Effect**: The program zooms into the fractal by decreasing the scale by a factor of `1.02` for each frame.

4. **Output**: Each frame is saved as a `.ppm` file, and after all frames are generated, they can be converted into a video using `ffmpeg`.


### Running the CUDA Code
1. **Install Dependencies**: Ensure you have `CUDA` and `nvcc` installed on your machine.
2. **Compile the CUDA Code**:
   ```bash
   nvcc -o mandelbrot_zoom mandelbrot_zoom.cu
   ```
3. **Run the Program**:
   ```bash
   ./mandelbrot_zoom
   ```

4. **Create the Video**:
   After generating all the `.ppm` files, use the following `ffmpeg` command to create a video:
   ```bash
   ffmpeg -framerate 60 -i frame_%04d.ppm -c:v libx264 -crf 18 -pix_fmt yuv420p mandelbrot_zoom.mp4
   ```
Now, you can play the video with whatever you prefer. Enjoy :D



