#include "Image.h"
using namespace std;
#include <stdexcept>

Image::Image() {
    width = 0;
    height = 0;
    pixels = vector<uint8_t>(0);
}

Image::Image(int w, int h) {
    width = w;
    height = h;
    pixels = vector<uint8_t>(w * h * 3);
}

void Image::resize(int scale, int unscale) {
    int new_width = (scale * width / unscale);
    int new_height = (scale * height / unscale);

    vector<uint8_t> new_pixels(new_width * new_height * 3);

    for (int y = 0; y < new_height; ++y) {
        for (int x = 0; x < new_width; ++x) {
            int x_initial = min(static_cast<int>(unscale * x / scale), width - 1);
            int y_initial = min(static_cast<int>(unscale * y / scale), height - 1);

            int sourse_ind = 3 * (y_initial * width + x_initial);
            int new_ind = 3 * (y * new_width + x);

            new_pixels[new_ind + 0] = pixels[sourse_ind + 0];
            new_pixels[new_ind + 1] = pixels[sourse_ind + 1];
            new_pixels[new_ind + 2] = pixels[sourse_ind + 2];
        }
    }

    width = new_width;
    height = new_height;
    pixels = move(new_pixels);
}

void Image::to_grayscale() {
    for (size_t i = 0; i < pixels.size(); i += 3) {
        uint8_t in_gray_scale = static_cast<uint8_t>(0.2126 * pixels[i] + 0.7152 * pixels[i + 1] + 0.0722 * pixels[i + 2]);
        pixels[i] = pixels[i + 1] = pixels[i + 2] = in_gray_scale;
    }
}
float Image::compare_with(Image im2)
{
    if (width != im2.width || height != im2.height || pixels.size() != im2.pixels.size()) {
        throw runtime_error("Image sizes does not add up");
    }

    float mse = 0.0f;
    for (size_t i = 0; i < pixels.size(); ++i) {
        float diff = (static_cast<float>(pixels[i]) - static_cast<float>(im2.pixels[i])) / 255.0f;
        mse += diff * diff;
    }

    mse /= pixels.size();

    return mse;
}
;