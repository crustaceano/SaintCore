#pragma once
#include <string>
#include "image.h"
using namespace std;

class ImageWriter {
public:
    virtual void write(Image image, string out_name) = 0;
    virtual ~ImageWriter() = default;
};
