#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>
#include <string>

namespace SaintCore {
    class BaseException : public std::exception {
        std::string msg;
    public:
        explicit BaseException(const std::string& message) : msg(message) {}

        const char* what() const noexcept override {
            return msg.c_str();
        }
    };

    class SizeMismatchException : public BaseException {
    public:
        explicit SizeMismatchException(const std::string& message)
            : BaseException(message) {}
    };
}

#endif // EXCEPTIONS_H