#ifndef EXCEPTIONS_H
#define EXCEPTIONS_H

#include <exception>

namespace SaintCore {
    class BaseException : public std::exception {
        std::string msg;
    public:
        explicit BaseException(const std::string& message) : msg(message) {}

        const char* what() const noexcept override {
            return msg.c_str();
        }
    };
}

#endif // EXCEPTIONS_H