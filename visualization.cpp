#include "visualization.h"
#include "Image.h"
#include "bmp_writer.h"

void visualize(std::vector<double> pixels, std::string output) {
    const int h = 28;
    const int w = 28;

    if (w * h < pixels.size()) {
        std::cerr << "Size missmatch error. Expected size is 784, actual: " << pixels.size() << std::endl;
    }

    Image image(w, h);

    for (int idx = 0; idx < w * h; ++idx) {
        double color = pixels[idx];
        int rgb_idx = 3 * idx;
        image.pixels[rgb_idx] = image.pixels[rgb_idx + 1] = image.pixels[rgb_idx + 2] = static_cast<uint8_t>(color);
    }
    BmpWriter writer;
    writer.write(image, output);
}