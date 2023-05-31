#include <chrono>
#include <cstdio>
#include <ratio>
#include <string>
#include <iostream>


class Timer {
public:
    using s  = std::ratio<1, 1>;
    using ms = std::ratio<1, 1000>;
    using us = std::ratio<1, 1000000>;
    using ns = std::ratio<1, 1000000000>;

public:
    Timer(){};

public:
    void start() {mStart = std::chrono::high_resolution_clock::now();}
    void stop()  {mStop  = std::chrono::high_resolution_clock::now();}

    template <typename span>
    void duration(std::string msg);

private:
    std::chrono::time_point<std::chrono::high_resolution_clock> mStart;
    std::chrono::time_point<std::chrono::high_resolution_clock> mStop;
};

template <typename span>
void Timer::duration(std::string msg){
    std::string str;
    char fMsg[100];
    std::sprintf(fMsg, "%-40s", msg.c_str());

    if(std::is_same<span, s>::value) { str = " s"; }
    else if(std::is_same<span, ms>::value) { str = " ms"; }
    else if(std::is_same<span, us>::value) { str = " us"; }
    else if(std::is_same<span, ns>::value) { str = " ns"; }

    std::chrono::duration<double, span> time = mStop - mStart;
    std::cout << fMsg << " uses " << time.count() << str << std::endl;
}

