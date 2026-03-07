/**
 * CAPD Logger stubs.
 *
 * CAPD was built with HAVE_LOG4CXX, so libcapd.so expects Logger member
 * functions that normally live in log4cxx. Since we don't link log4cxx,
 * we provide no-op implementations here using the actual CAPD Logger class.
 */
#include "capd/auxil/Logger.h"

namespace capd { namespace auxil {

// Constructor — declared in Logger.h but not defined (normally in log4cxx backend)
Logger::Logger(const std::string& /*fileName*/, const std::string& /*buildDir*/, bool /*global*/) {}

// These are declared (not inline) in Logger.h when HAVE_LOG4CXX is defined
bool Logger::isDebugEnabled() const { return false; }
bool Logger::isTraceEnabled() const { return false; }
void Logger::forcedLogDebug(const std::string&, const char*, const char*, int) {}
void Logger::forcedLogTrace(const std::string&, const char*, const char*, int) {}

}} // namespace capd::auxil

// Global CAPD_LOGGER — required by CAPD library internals
capd::auxil::Logger CAPD_LOGGER::getCAPDLogger(const char* file, const char* buildDir) {
    static capd::auxil::Logger logger(file, buildDir ? buildDir : "", true);
    return logger;
}
