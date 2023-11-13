#include <stdlib.h>

void im2colOnHost(
    float *filter, float* input, float *output,
    float *M, float *N, float *P,
    int IC, int IH, int IW,
    int KH, int KW,
    int OC, int OH, int OW)
{
    //感兴趣的人可以实现一下
}

void ExplicitGEMMConvOnHost(
    float *filter, float *input, float *output,
    int IC, int IH, int IW,
    int KH, int KW,
    int OC, int OH, int OW)
{
    float* M = (float*)malloc(OC * IC * KH * KW * sizeof(float));
    float* N = (float*)malloc(OH * OW * IC * KH * KW * sizeof(float));
    float* P = (float*)malloc(OH * OH * OW * sizeof(float));

    im2colOnHost(
        filter, input, output, 
        M, N, P, 
        IC, IH, IW, 
        KH, KW, 
        OC, OH, OW);

    for (int i = 0; i < OC; i++) {
        for (int j = 0; j < OH * OW; j++) {
            float sum = 0;
            for (int k = 0; k < IC * KH * KW; k++) {
                float a = M[i * IC * KH * KW + k];
                float b = N[k * OH * OW + j];
                sum += a * b;
            }
            P[i * OC + j] = sum;
        }
    }
}


void ImplicitGEMMConvOnHost(
    float *filter, float *input, float *output,
    int IC, int IH, int IW,
    int KH, int KW,
    int OC, int OH, int OW)
{
    for (int i = 0; i < OC; i++) {
        for (int j = 0; j < OH * OW; j++) {
            int oh = j / OW;
            int ow = j % OW;
            int oc = i;
            int output_index = oc * OH * OW + oh * OW + ow;
            float sum = 0;
            for (int k = 0; k < IC * KH * KW; k++) {
                int ic = k / (KH * KW);
                int kh = (k % (KH * KW)) / KW;
                int kw = (k % (KH * KW)) % KW;
                int ih = oh + kh;
                int iw = ow + ow;

                int filter_index = oc * IC * KH * KW + 
                    ic * KH * KW + kh * KW + kw;
                int input_index = ic * IH * IW + ih * IW + iw;

                sum += filter[filter_index] * input[input_index];
            }
            output[output_index] = sum;
        }
    }
}



//Implicit GEMM Convolution

for (int i = 0; i < K; i++) {
    for (int j = 0; j < N*Oh*Ow; j++) {
        int on = j/(Oh*Ow);                    //N维度坐标
        int oh = (j%(Oh*Ow))/Ow;               //Oh维度坐标
        int ow = (j%(Oh*Ow))%Ow;               //Ow维度坐标
        output[on][i][oh][ow] =0;
        for (int k = 0; k < C*R*S; k++) {
            int ic = k/(R*S);                  //C维度坐标
            int ir = k%(R*S)/S;                //R维度坐标
            int is = k%(R*S)%S;                //S维度坐标
            int ih = oh*STRIDE + ir;           //H维度坐标
            int iw = ow*STRIDE + is;           //W维度坐标
            output[on][i][oh][ow] += filter[i][ic][ir][is] * input[on][ic][ih][iw];
        }
    }
}


