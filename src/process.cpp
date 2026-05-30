#include <cmath>
#include <vector>
#include <cstdint>
#include "process.hpp"
#include "fftw3.h"

#define LAMBDA 0.02

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

double estimate_sigma(fftw_complex *out_blurred, int height, int width){
    int out_width = (width / 2) + 1;
    
    // accumulators for linear regression
    double sum_x = 0.0;
    double sum_y = 0.0;
    double sum_xy = 0.0;
    double sum_xx = 0.0;
    long long n = 0;

    double f_min = 0.05; 
    double f_max = 0.15; // 0.15 because the image is very noisy
    double f_min_sq = f_min * f_min;
    double f_max_sq = f_max * f_max;

    for (int y = 0; y < height; ++y) {
        
        double fy = (y <= height / 2) ? static_cast<double>(y) : static_cast<double>(y - height);
        double fy_norm = fy / height; 

        for (int x = 0; x < out_width; ++x) {
            
            double fx_norm = static_cast<double>(x) / width; 
            
            double f_sq = (fx_norm * fx_norm) + (fy_norm * fy_norm);

            if (f_sq >= f_min_sq && f_sq <= f_max_sq) {
                int idx = y * out_width + x;
                
                double real = out_blurred[idx][0];
                double imag = out_blurred[idx][1];
                
                // power spectro
                double power = (real * real) + (imag * imag);
                
                if (power > 1e-10) {
                    
                    double ln_power = std::log(power);

                    sum_x += f_sq;
                    sum_y += ln_power;
                    sum_xy += (f_sq * ln_power);
                    sum_xx += (f_sq * f_sq);
                    n++;
                }
            }
        }
    }

    // if the filter ignored almost everything there is no enough data
    if (n < 10) return 0.0; 

    // slope of the line
    double denominator = (n * sum_xx) - (sum_x * sum_x);
    
    if (denominator == 0.0) return 0.0;

    double m = ((n * sum_xy) - (sum_x * sum_y)) / denominator;

    // if the slope is positive the image is not blurry or has too much noisy
    if (m >= 0.0) {
        return 0.0; 
    }

    // ln(P) = -4 * pi^2 * sigma^2 * f^2
    // m = -4 * pi^2 * sigma^2  =>  sigma = sqrt(-m) / (2 * pi)
    const double PI = 3.14159265358979323846;
    double sigma = std::sqrt(-m) / (2.0 * PI);

    return sigma;

}

std::vector<uint8_t>* deconv(
        std::vector<uint8_t> &image, 
        int height, 
        int width,
        double lambda = LAMBDA
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
    }

    fftw_execute(plan_blurred);
    
    // estimate sigma and create kernel pipeline
    double sigma = estimate_sigma(out_blurred, height, width);
    
    int kernel_side;
    std::vector<double> kernel = gaussian_kernel(sigma, kernel_side);
    kernel = pad_kernel(kernel, kernel_side, kernel_side, height, width);

    for (int i = 0; i < height * width; ++i) {
        in_kernel[i] = static_cast<double>(kernel[i]);
    }
    
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

std::vector<uint8_t>* deblur(
    std::vector<uint8_t> &image,
    int height,
    int width
){
    return deconv(image, height, width, LAMBDA);
}