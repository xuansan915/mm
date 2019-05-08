#ifndef LOGGER_H
#define LOGGER_H

/*
 *\logger.h
 *\brief 日记模块
 */

#include <iostream>
#include <iomanip>
#include <fstream>
#include <string>
#include <cstdlib>
#include <stdint.h>

///
/// \brief 日志文件的类型
///
typedef enum log_rank
{
    INFO,
    WARNING,
    ERROR,
    FATAL
} log_rank_t;

///
/// \brief 初始化日志文件
/// \param info_log_filename 信息文件的名字
/// \param warn_log_filename 警告文件的名字
/// \param error_log_filename 错误文件的名字
void initLogger(const std::string&info_log_filename);

///
/// \brief 日志系统类
///
class Logger
{
    friend void initLogger(const std::string& info_log_filename);

public:
    //构造函数
    Logger(log_rank_t log_rank) : m_log_rank(log_rank) {};

    ~Logger();
    ///
    /// \brief 写入日志信息之前先写入的源代码文件名, 行号, 函数名
    /// \param log_rank 日志的等级
    /// \param line 日志发生的行号
    /// \param function 日志发生的函数
    static std::ostream& start(log_rank_t log_rank,
                               const int32_t line,
                               const std::string& functions);

private:
    ///
    /// \brief 根据等级获取相应的日志输出流
    ///
    static std::ostream& getStream(log_rank_t log_rank);

    static std::ofstream m_info_log_file;                   ///< 信息日子的输出流
    static std::ofstream m_warn_log_file;                  ///< 警告信息的输出流
    static std::ofstream m_error_log_file;                  ///< 错误信息的输出流
    log_rank_t m_log_rank;                             ///< 日志的信息的等级
};


///
/// \brief 根据不同等级进行用不同的输出流进行读写
///
#define LOGS(msg)   \
Logger(INFO).start(INFO, __LINE__,msg)

#endif // LOGGER_H
