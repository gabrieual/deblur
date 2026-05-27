#include <cmath>
#include <cstddef>
#include <vector>
#include <cstdint>
#include "process.hpp"
#include "fftw3.h"

std::vector<double> pad_kernel(
    std::vector<double> &kernel, 
    size_t k_height,
    size_t k_width, 
    size_t height, 
    size_t width){

    std::vector<double> p_kernel(height * width, 0.0);
    
    size_t cx = k_width / 2;
    size_t cy = k_height / 2;

    for (int y = 0; y < k_height; y++){
        for (int x = 0; x < k_width; x++){
            int diffx = x - cx;
            int diffy = y - cy;

            int new_x = (diffx + width) % width;
            int new_y = (diffy + height) % height;
        
            size_t coords_pkernel = new_y*width + new_x;
            size_t coords_kernel  = y*k_width + x;

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
    size_t height,
    size_t out_width
){
    
    for (size_t i = 0; i < height * out_width; i++){
        double blur_real = out_blurred[i][0];
        double blur_imag = out_blurred[i][1];

        double kern_real = out_kernel[i][0];
        double kern_imag = out_kernel[i][1];

        double kern_mag_sq = kern_real*kern_real + kern_imag*kern_imag;

        int denominator = kern_mag_sq + lambda;

        out_clean[i][0] = (blur_real*kern_real + blur_imag*blur_real) / denominator;
        out_clean[i][1] = (blur_imag*kern_real - blur_real*kern_imag) / denominator;
    }

}

std::vector<uint8_t>* deconv(
        std::vector<uint8_t> &image, 
        std::vector<double> &kernel,
        size_t height, 
        size_t width){

    // output dimension (r2c otmization)
    size_t out_width = (width/2) + 1;
    
    double *in_blurred, *in_kernel, *in_clean; 
    fftw_complex *out_blurred, *out_clean, *out_kernel ; 

    in_blurred  = fftw_alloc_real(sizeof(double) * height * width);
    in_clean    = fftw_alloc_real(sizeof(double) * height * width); 
    in_kernel   = fftw_alloc_real(sizeof(double) * height * width);

    out_blurred = fftw_alloc_complex(sizeof(fftw_complex) * height * out_width);
    out_clean   = fftw_alloc_complex(sizeof(fftw_complex) * height * out_width);
    out_kernel  = fftw_alloc_complex(sizeof(fftw_complex) * height * out_width);

    for (size_t i = 0; i < image.size(); i++) {
        in_blurred[i] = static_cast<double>(image[i]);
        in_kernel[i]  = kernel[i];
    }

    fftw_plan plan_blurred = fftw_plan_dft_r2c_2d(height, width, in_blurred, out_blurred, FFTW_ESTIMATE);
    fftw_plan plan_kernel  = fftw_plan_dft_r2c_2d(height, width, in_kernel, out_kernel, FFTW_ESTIMATE);
    
    
    fftw_execute(plan_blurred);
    fftw_execute(plan_kernel);
    
    weiner_filter(out_clean, out_blurred, out_kernel, 0.5, height, out_width);
    
    fftw_plan plan_inverse = fftw_plan_dft_c2r_2d(height, width, out_clean, in_clean, FFTW_ESTIMATE);

    fftw_execute(plan_inverse);

    std::vector<uint8_t> *new_image = new std::vector<uint8_t>(image.size());

    double N = width * height;

    for (size_t i = 0; i < image.size(); i++){
        double pixel = in_clean[i] / N;
        
        if (pixel > 255) pixel = 255;
        if (pixel < 0) pixel = 0;
    
        (*new_image)[i] = (uint8_t)(pixel);
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
