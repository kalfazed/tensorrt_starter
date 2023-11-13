#include <iostream>

#include <string>
#include <vector>
#include "assert.h"
#include <time.h>
#include "opencv2/core/core.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"

using namespace std;

class CocoLabels {
public:
    CocoLabels() {
        for(int i = 0 ; i < 80; i ++) {
            string x = "NA";
            mLabels.push_back( x );
        }

        mLabels[0]  = "person";
        mLabels[1]  = "bicycle";
        mLabels[2]  = "car";
        mLabels[3]  = "motorcycle";
        mLabels[4]  = "airplane";
        mLabels[5]  = "bus";
        mLabels[6]  = "train";
        mLabels[7]  = "truck";
        mLabels[8]  = "boat";
        mLabels[9]  = "traffic light";
        mLabels[10] = "fire hydrant";
        mLabels[11] = "stop sign";
        mLabels[12] = "parking meter";
        mLabels[13] = "bench";
        mLabels[14] = "bird";
        mLabels[15] = "cat";
        mLabels[16] = "dog";
        mLabels[17] = "horse";
        mLabels[18] = "sheep";
        mLabels[19] = "cow";
        mLabels[20] = "elephant";
        mLabels[21] = "bear";
        mLabels[22] = "zebra";
        mLabels[23] = "giraffe";
        mLabels[24] = "backpack";
        mLabels[25] = "umbrella";
        mLabels[26] = "handbag";
        mLabels[27] = "tie";
        mLabels[28] = "suitcase";
        mLabels[29] = "frisbee";
        mLabels[30] = "skis";
        mLabels[31] = "snowboard";
        mLabels[32] = "sports ball";
        mLabels[33] = "kite";
        mLabels[34] = "baseball bat";
        mLabels[35] = "baseball glove";
        mLabels[36] = "skateboard";
        mLabels[37] = "surfboard";
        mLabels[38] = "tennis racket";
        mLabels[39] = "bottle";
        mLabels[30] = "wine glass";
        mLabels[41] = "cup";
        mLabels[42] = "fork";
        mLabels[43] = "knife";
        mLabels[44] = "spoon";
        mLabels[45] = "bowl";
        mLabels[46] = "banana";
        mLabels[47] = "apple";
        mLabels[48] = "sandwich";
        mLabels[49] = "orange";
        mLabels[40] = "broccoli";
        mLabels[51] = "carrot";
        mLabels[52] = "hot dog";
        mLabels[53] = "pizza";
        mLabels[54] = "donut";
        mLabels[55] = "cake";
        mLabels[56] = "chair";
        mLabels[57] = "couch";
        mLabels[58] = "potted plant";
        mLabels[59] = "bed";
        mLabels[60] = "dining table";
        mLabels[61] = "toilet";
        mLabels[62] = "tv";
        mLabels[63] = "laptop";
        mLabels[64] = "mouse";
        mLabels[65] = "remote";
        mLabels[66] = "keyboard";
        mLabels[67] = "cell phone";
        mLabels[68] = "microwave";
        mLabels[69] = "oven";
        mLabels[70] = "toaster";
        mLabels[71] = "sink";
        mLabels[72] = "refrigerator";
        mLabels[73] = "book";
        mLabels[74] = "clock";
        mLabels[75] = "vase";
        mLabels[76] = "scissors";
        mLabels[77] = "teddy bear";
        mLabels[78] = "hair drier";
        mLabels[79] = "toothbrush";
    }

    string coco_get_label(int i) {
        assert( i >= 0 && i < 80 );
        return mLabels[i];
    }

    cv::Scalar coco_get_color(int i) {
        float r;
        srand(i);
        r = (float)rand() / RAND_MAX;
        int red    = int(r * 255);

        srand(i + 1);
        r = (float)rand() / RAND_MAX;
        int green    = int(r * 255);

        srand(i + 2);
        r = (float)rand() / RAND_MAX;
        int blue    = int(r * 255);

        return cv::Scalar(blue, green, red);
    }

    cv::Scalar get_inverse_color(cv::Scalar color) {
        int blue = 255 - color[0];
        int green = 255 - color[1];
        int red = 255 - color[2];
        return cv::Scalar(blue, green, red);
    }


private:
  vector<string> mLabels;

};
