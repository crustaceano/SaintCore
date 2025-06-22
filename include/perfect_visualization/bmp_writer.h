#pragma once
#include "image_writer.h"
using namespace std;

class BmpWriter : public ImageWriter {
public:
    void write(Image image, string out_name) override;
};
