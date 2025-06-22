#include <gtest/gtest.h>

#include <include/tensor.h>

#include "exceptions.h"

TEST(Tensor, Base) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {1, 2, 3}, {4, 5, 6} }));
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(1, 1), 5);
    EXPECT_THROW(Tensor({{1, 2}, {3}}), BaseException);
}


TEST(Tensor, Index) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(0, 0), 1);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(0, 1), 2);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(0, 2), 3);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(1, 0), 4);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(1, 1), 5);
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }).at(1, 2), 6);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(-1, 0), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(-1, -1), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(0, -1), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(-100, -100), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(-100, 0), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(0, -100), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(100, 0), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(0, 100), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(100, 100), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(2, 0), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(0, 3), BaseException);
    EXPECT_THROW(Tensor({{1, 2, 3}, {4, 5, 6}}).at(2, 3), BaseException);
}


TEST(Tensor, Equal) {
    using namespace SaintCore;
    floatT eps = 5e-6;
    floatT s1 = 1 + eps, s2 = 2 + eps, s3 = 3 + eps, s4 = 4 + eps, s5 = 5 + eps, s6 = 6 + eps;
    EXPECT_NE(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {s1, s2, s3}, {s4, s5, s6} }));
    eps = 5e-7;
    s1 = 1 + eps, s2 = 2 + eps, s3 = 3 + eps, s4 = 4 + eps, s5 = 5 + eps, s6 = 6 + eps;
    EXPECT_EQ(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({ {s1, s2, s3}, {s4, s5, s6} }));
    EXPECT_NE(Tensor({ {1, 2, 3}, {4, 5, 6} }), Tensor({{1, 2, 3}}));
}


TEST(Tensor, Sum) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({{1, 2}}) + Tensor({{3, 4}}), Tensor({{4, 6}}));
    EXPECT_EQ(Tensor({{1, 2}}) - Tensor({{3, 10}}), Tensor({{-2, -8}}));
    EXPECT_EQ(Tensor({{1, 2, 3}, {4, 5, 6}}) + Tensor({{7, 8, 9}, {10, 11, 12}}),
        Tensor({{8, 10, 12}, {14, 16, 18}}));
    EXPECT_EQ(Tensor({{1, 2, 3}, {4, 5, 6}}) - Tensor({{7, 8, 9}, {10, 11, 12}}),
        Tensor({{-6, -6, -6}, {-6, -6, -6}}));
    EXPECT_THROW(Tensor({{1, 2}}) + Tensor({{3, 4, 5}}), BaseException);
}


TEST(Tensor, ElementMul) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({{1, 2}}) % Tensor({{3, 4}}), Tensor({{3, 8}}));
    EXPECT_EQ(Tensor({{1, 2, 3}, {4, 5, 6}}) % Tensor({{7, 8, 9}, {10, 11, 12}}),
        Tensor({{7, 16, 27}, {40, 55, 72}}));
    EXPECT_THROW(Tensor({{1, 2}}) % Tensor({{3, 4, 5}}), BaseException);
}


TEST(Tensor, Transpose) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({{1, 2}}).transposed(), Tensor({{1,2}}).transposed());
    EXPECT_EQ(Tensor({{1, 2, 3}, {4, 5, 6}}).transposed(), Tensor({{1, 4}, {2, 5}, {3, 6}}));
}


TEST(Tensor, MatMul) {
    using namespace SaintCore;
    EXPECT_EQ(Tensor({{1, 2}}) * Tensor({{1, 2}}).transposed(), Tensor({std::vector<floatT>{5}}));
    EXPECT_EQ(Tensor({{1, 2, 3}, {4, 5, 6}}) * Tensor({{1, 2}, {3, 4}, {5, 6}}),
        Tensor({{22, 28}, {49, 64}}));
    EXPECT_EQ(Tensor({{1, 2, 3}}) * Tensor({{4}, {5}, {6}}),
        Tensor({std::vector<floatT>{32}}));
    EXPECT_THROW(Tensor({{1, 2, 3, 4}, {5, 6, 7, 8}}) * Tensor({{1, 2}, {3, 4}, {5, 6}}), BaseException);
}
