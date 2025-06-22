#include "bmp_writer.h"
#include <fstream>
#include <stdexcept>
#include <vector>
using namespace std;

void BmpWriter::write(Image image, string out_name) {
    ofstream out;
    out.open(out_name, ios::binary);
    if (!out) throw runtime_error("Unexisted output file: " + out_name);

    int w = image.width, h = image.height;

    int pad_size = (4 - (w * 3) % 4) % 4;
    int row_size = w * 3 + pad_size;
    int data_size = row_size * h;

    char header[54] = {
        'B','M',
        0,0,0,0,
        0,0,0,0,
        54,0,0,0,
        40,0,0,0,
        0,0,0,0,
        0,0,0,0,
        1,0,
        24,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0,
        0,0,0,0
    };

    int file_size = data_size + 54;

    memcpy(header + 2, &file_size, 4);
    memcpy(header + 18, &w, 4);
    memcpy(header + 22, &h, 4);
    memcpy(header + 34, &data_size, 4);

    out.write(header, 54);

    vector<uint8_t> row(row_size);
    for (int y = 0; y < h; ++y) {
        int y_initial = h - 1 - y;

        for (int x = 0; x < w; ++x) {
            int rgb_ind = 3 * (y_initial * w + x);
            int bmp_ind = x * 3;

            row[bmp_ind + 0] = image.pixels[rgb_ind + 2];
            row[bmp_ind + 1] = image.pixels[rgb_ind + 1];
            row[bmp_ind + 2] = image.pixels[rgb_ind + 0];
        }

        out.write(reinterpret_cast<char*>(row.data()), row_size);
    }
    out.close();
}