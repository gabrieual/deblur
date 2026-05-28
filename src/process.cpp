#include <cmath>
#include <vector>
#include <cstdint>
#include "process.hpp"
#include "fftw3.h"

#define LAMBDA 0.05
#define SIGMA 2

std::vector<double> pad_kernel(
    std::vector<double> &kernel, 
    int k_height,
    int k_width, 
    int height, 
    int width){

    std::vector<double> p_kernel(height * width, 0.0);
    
    int cx = k_width / 2;
    int cy = k_height / 2;

    for (int y = 0; y < k_height; y++){
        for (int x = 0; x < k_width; x++){
            int diffx = x - cx;
            int diffy = y - cy;

            int new_x = (diffx + width) % width;
            int new_y = (diffy + height) % height;
        
            int coords_pkernel = new_y*width + new_x;
            int coords_kernel  = y*k_width + x;

            p_kernel[coords_pkernel] = kernel[coords_kernel];
        }
    }

    return p_kernel;
}

void weiner_filter(
    fftw_complex *out_clean,
    fftw_complex *out_blurred,
    fftw_complex *out_kernel,
    double lambda,
    int height,
    int out_width
    ){
    
    for (int i = 0; i < height * out_width; i++){
        double blur_real = out_blurred[i][0];
        double blur_imag = out_blurred[i][1];

        double kern_real = out_kernel[i][0];
        double kern_imag = out_kernel[i][1];

        double kern_mag_sq = kern_real*kern_real + kern_imag*kern_imag;

        double denominator = kern_mag_sq + lambda;

        out_clean[i][0] = (blur_real*kern_real + blur_imag*kern_imag) / denominator;
        out_clean[i][1] = (blur_imag*kern_real - blur_real*kern_imag) / denominator;
    }

}

std::vector<uint8_t>* deconv(
        std::vector<uint8_t> &image, 
        std::vector<double> &kernel,
        int height, 
        int width,
        double lambda
    ){

    // output dimension (r2c otmization)
    int out_width = (width/2) + 1;
    
    double *in_blurred, *in_kernel, *in_clean; 
    fftw_complex *out_blurred, *out_clean, *out_kernel ; 

    in_blurred  = fftw_alloc_real(height * width);
    in_clean    = fftw_alloc_real(height * width); 
    in_kernel   = fftw_alloc_real(height * width);

    out_blurred = fftw_alloc_complex(height * out_width);
    out_clean   = fftw_alloc_complex(height * out_width);
    out_kernel  = fftw_alloc_complex(height * out_width);

    fftw_plan plan_blurred = fftw_plan_dft_r2c_2d(height, width, in_blurred, out_blurred, FFTW_ESTIMATE);
    fftw_plan plan_kernel  = fftw_plan_dft_r2c_2d(height, width, in_kernel, out_kernel, FFTW_ESTIMATE);
    fftw_plan plan_inverse = fftw_plan_dft_c2r_2d(height, width, out_clean, in_clean, FFTW_ESTIMATE);
    
    for (int i = 0; i < height * width; ++i) {
        in_blurred[i] = static_cast<double>(image[i*3]);
        in_kernel[i]  = kernel[i];
    }

    fftw_execute(plan_blurred);
    fftw_execute(plan_kernel);
    
    weiner_filter(out_clean, out_blurred, out_kernel, lambda, height, out_width);

    fftw_execute(plan_inverse);

    std::vector<uint8_t> *new_image = new std::vector<uint8_t>(image.size());

    double N = width * height;

    for (int i = 0; i < height * width; i++){
        double pixel = in_clean[i] / N;
        
        if (pixel > 255) pixel = 255;
        if (pixel < 0) pixel = 0;
    
        uint8_t pixel_val = static_cast<uint8_t>(pixel);

        (*new_image)[i*3 + 1] = pixel_val;
        (*new_image)[i*3 + 2] = pixel_val;
        (*new_image)[i*3 + 3] = pixel_val;
    }

    fftw_destroy_plan(plan_blurred);
    fftw_destroy_plan(plan_kernel);
    fftw_destroy_plan(plan_inverse);

    fftw_free(in_blurred);
    fftw_free(in_clean);
    fftw_free(in_kernel);

    
    fftw_free(out_blurred);
    fftw_free(out_clean);
    fftw_free(out_kernel);

    return new_image; 
}

std::vector<double> gaussian_kernel(double sigma, int &out_size){
    // kernel radius is based in Radius = ceil ( 3 x \sigma) 
    // 99.7% of the energy of the blur is at 3 sigmas from the center
    int kernel_radius = std::ceil(3*sigma);
    
    out_size = 2 * kernel_radius + 1;

    std::vector<double> kernel(out_size * out_size);

    // G(x, y) = (1 / 2·pi·sigma²)·e^(-(x²+y²) / 2·sigma²)

    double exp_denom = 2.0 * sigma * sigma; 
    double sum = 0.0; // used for normalization
    for (int y = -kernel_radius; y <= kernel_radius; y++){
        for (int x = -kernel_radius; x <= kernel_radius; x++){
            int idx = (y + kernel_radius) * out_size + (x + kernel_radius);
            // we can ignore the constant cause normalization
            double v = std::exp(-(x*x + y*y) / exp_denom);
            kernel[idx] = v;
            sum += kernel[idx];
        }
    }

    if ((sum > 1 && sum - 1 > 1e-15) || (sum < 1 && 1 - sum > 1e-15)){
        for (int i = 0; i < out_size * out_size; i++)
            kernel[i] /= sum; 
    }

    return kernel;
}

std::vector<uint8_t>* deblur(
    std::vector<uint8_t> &image,
    int height,
    int width
){

    int out_size;

    std::vector<double> kernel = gaussian_kernel(SIGMA, out_size);

    std::vector<double> p_kernel = pad_kernel(kernel, out_size, out_size, height, width);

    return deconv(image, p_kernel, height, width, LAMBDA);
}