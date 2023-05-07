#include <iostream>
#include <memory>

#include "model.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    Model model;
    if(!model.build()){
        cout << "fail in building model" << endl;
    }
    return 0;
}
