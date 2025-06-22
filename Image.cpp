#include "Image.h"
using namespace std;
#include <stdexcept>

Image::Image() {
    width = 0;
    height = 0;
    pixels = vector<uint8_t>(0);
}

Image::Image(const int w, const int h) {
    width = w;
    height = h;
    pixels = vector<uint8_t>(w * h * 3);
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