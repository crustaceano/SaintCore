#pragma once
#include <vector>
#include <cstdint>
using namespace std;

class Image {
public:
    int width;
    int height;
    vector<uint8_t> pixels;

    Image();
    Image(int w, int h);
    float compare_with(Image image_for_compare);
};