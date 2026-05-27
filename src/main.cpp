#include <cstddef>
#include <cstdint>
#include <ios>
#include <iostream>
#include <vector>
#include "process.hpp"
 
// using the attribute to handle with the header misalign
struct __attribute__((packed)) Header{ 
    uint16_t bfType;
    uint32_t bfSize;
    uint16_t bfReserved1;
    uint16_t bfReserved2;
    uint32_t bfOffBits;
};

struct FileHeader{
    uint32_t biSize;
    uint32_t biWidth;
    uint32_t biHeight;
    uint16_t biPlanes;
    uint16_t biBitCount;
    uint32_t biCompression;
    uint32_t biSizeImage;
    uint32_t biXPelsPerMeter;
    uint32_t biClrUsed;
    uint32_t biClrImportant;
};

int main(){
    std::ios_base::sync_with_stdio(false);
    
    Header header;
    FileHeader file_header;
    
    std::cin.read(reinterpret_cast<char*>(&header), sizeof(header));
    std::cin.read(reinterpret_cast<char*>(&file_header), sizeof(file_header));
    
    size_t image_size = file_header.biHeight * file_header.biWidth * 3;
    std::vector<uint8_t> image(image_size);
    std::cin.read(reinterpret_cast<char*>(image.data()), image_size);
    
    std::vector<uint8_t> *new_image;

    new_image = deblur(image, file_header.biHeight, file_header.biWidth);


    std::cout.write(reinterpret_cast<char*>(&header), sizeof(header));
    std::cout.write(reinterpret_cast<char*>(&file_header), sizeof(file_header));
    std::cout.write(reinterpret_cast<char*>(new_image->data()), image_size);

    return 0;
}