#ifndef EXCEPTIONS_H_
#define EXCEPTIONS_H_

#include <exception>
#include <string>

namespace cudapp {

struct base_exception : public std::exception {
	base_exception(const std::string &msg) : msg(msg) {}
	base_exception(const char* msg) : msg(std::string(msg)) {}
	~base_exception() throw() {}

	const char* what() const throw() { return (std::string("--(!) cudapp Error: ") + msg).c_str(); }

private:
	std::string msg;

};

struct bad_alloc : public base_exception {
	bad_alloc(const char* msg) : base_exception(msg) {}
};

struct bad_init : public base_exception {
	bad_init(const char* msg) : base_exception(msg) {}
};

struct invalid_argument : public base_exception {
	invalid_argument(const char* msg) : base_exception(msg) {}
};

struct cuda_error : public base_exception {
	cuda_error(const cudaError_t &error, std::string msg = "Cuda Error: ") : base_exception(msg+std::string(cudaGetErrorString(error))) {}
};


}

#endif /* EXCEPTIONS_H_ */
