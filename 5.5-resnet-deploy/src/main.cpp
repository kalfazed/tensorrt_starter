#include <iostream>
#include <memory>

#include "model.hpp"

using namespace std;

int main(int argc, char const *argv[])
{
    Model model("models/shufflenetV2.onnx");
    string imagePath = "data/tiny-cat.jpg";

    if(!model.build()){
        cout << "fail in building model" << endl;
        return 0;
    }
    if(!model.infer(imagePath)){
        cout << "fail in infering model" << endl;
        return 0;
    }
    return 0;
}
