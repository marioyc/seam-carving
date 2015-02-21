#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <cstring>

using namespace cv;
using namespace std;

Mat calculate_energy(Mat I){
    Mat Ix,Iy;

    Sobel(I,Ix,CV_32F,1,0);
    convertScaleAbs(Ix,Ix);

    Sobel(I,Iy,CV_32F,0,1);
    convertScaleAbs(Iy,Iy);

    Mat energy;
    addWeighted(Ix, 0.5, Iy, 0.5, 0, energy, CV_8U);

    return energy;
}

Mat calculate_energy2(Mat I){
    int Y = I.rows,X = I.cols;
    Mat energy = Mat(Y, X, CV_32S);

    for(int x = 0;x < X;++x){
        for(int y = 0;y < Y;++y){
            int val = 0;

            if(x > 0 && x + 1 < X)
                val += abs((int)I.at<uchar>(y,x + 1) - (int)I.at<uchar>(y,x - 1));
            else if(x > 0)
                val += 2 * abs((int)I.at<uchar>(y,x) - (int)I.at<uchar>(y,x - 1));
            else
                val += 2 * abs((int)I.at<uchar>(y,x + 1) - (int)I.at<uchar>(y,x));

            if(y > 0 && y + 1 < Y)
                val += abs((int)I.at<uchar>(y + 1,x) - (int)I.at<uchar>(y - 1,x));
            else if(y > 0)
                val += 2 * abs((int)I.at<uchar>(y,x) - (int)I.at<uchar>(y - 1,x));
            else
                val += 2 * abs((int)I.at<uchar>(y + 1,x) - (int)I.at<uchar>(y,x));

            energy.at<int>(y,x) = val;
        }
    }

    return energy;
}

int query(int bit[], int idx){
    int ret = 0;

    for(int x = idx;x > 0;x -= x & -x)
        ret += bit[x];

    return ret;
}

void update(int bit[], int maxX, int idx){
    for(int x = idx;x < maxX;x += x & -x)
        ++bit[x];
}

int main(){
    Mat_<Vec3b> I = imread("../bench.jpg");

    imshow("seam-carving",I);
    waitKey(0);

    int Y = I.rows,X = I.cols;
    int Y0 = Y,X0 = X;

    Mat gray,energy;
    
    //cvtColor(I,gray,CV_BGR2GRAY);
    //energy = calculate_energy(gray);
    //imshow("seam-carving",energy);
    //waitKey(0);

    unsigned int dpH[X][Y],dpV[X][Y];
    int dirH[X][Y],dirV[X][Y];
    vector<int> posH[X],posV[Y];
    int bitH[Y + 1],bitV[X + 1];

    Mat seams = I.clone();

    // Horizontal seams

    for(int it = 0;it < 50;++it){
        cvtColor(I,gray,CV_BGR2GRAY);
        energy = calculate_energy2(gray);

        for(int y = 0;y < Y;++y)
            dpH[0][y] = energy.at<uchar>(y,0);

        for(int x = 1;x < X;++x){
            for(int y = 0;y < Y;++y){
                uint val = energy.at<uchar>(y,x);
                dpH[x][y] = -1;

                if(y > 0 && (dpH[x][y] == -1 || val + dpH[x - 1][y - 1] < dpH[x][y])){
                    dpH[x][y] = val + dpH[x - 1][y - 1];
                    dirH[x][y] = -1;
                }

                if(dpH[x][y] == -1 || val + dpH[x - 1][y] < dpH[x][y]){
                    dpH[x][y] = val + dpH[x - 1][y];
                    dirH[x][y] = 0;
                }

                if(y + 1 < Y && (dpH[x][y] == -1 || val + dpH[x - 1][y + 1] < dpH[x][y])){
                    dpH[x][y] = val + dpH[x - 1][y + 1];
                    dirH[x][y] = 1;
                }
            }
        }

        unsigned int bestH = dpH[X - 1][0];
        int cury = 0;

        for(int y = 0;y < Y;++y){
            if(dpH[X - 1][y] < bestH){
                bestH = dpH[X - 1][y];
                cury = y;
            }
        }

        //cout << "cury = " << cury << endl;
        //cout << "bestH = " << bestH << endl;

        Mat_<Vec3b> tmp(Y - 1,X);

        for(int x = X - 1,cont = 0;x >= 0;--x,++cont){
            posH[x].push_back(cury);

            for(int i = 0;i < Y;++i){
                if(i < cury) tmp.at<Vec3b>(i,x) = I.at<Vec3b>(i,x);
                else if(i > cury) tmp.at<Vec3b>(i - 1,x) = I.at<Vec3b>(i,x);
            }

            if(x > 0)
                cury = cury + dirH[x][cury];
        }

        I = tmp;
        --Y;
    }

    //imwrite("seam_horizontal2.jpg", I);

    for(int x = 0;x < X;++x){
        memset(bitH,0,sizeof bitH);

        for(int i = 0;i < posH[x].size();++i){
            int y = posH[x][i];
            update(bitH,Y0 + 1,y + 1);
            y += query(bitH,y + 1) - 1;

            seams.at<Vec3b>(y,x) = Vec3b(0,0,255);
        }
    }

    // Vertical seams

    for(int it = 0;it < 100;++it){
        cvtColor(I,gray,CV_BGR2GRAY);
        energy = calculate_energy2(gray);

        for(int x = 0;x < X;++x)
            dpV[x][0] = energy.at<uint>(0,x);

        for(int y = 1;y < Y;++y){
            for(int x = 0;x < X;++x){
                uint val = energy.at<uint>(y,x);
                dpV[x][y] = -1;

                if(x > 0 && (dpV[x][y] == -1 || val + dpV[x - 1][y - 1] < dpV[x][y])){
                    dpV[x][y] = val + dpV[x - 1][y - 1];
                    dirV[x][y] = -1;
                }

                if(dpV[x][y] == -1 || val + dpV[x][y - 1] < dpV[x][y]){
                    dpV[x][y] = val + dpV[x][y - 1];
                    dirV[x][y] = 0;
                }

                if(x + 1 < X && (dpV[x][y] == -1 || val + dpV[x + 1][y - 1] < dpV[x][y])){
                    dpV[x][y] = val + dpV[x + 1][y - 1];
                    dirV[x][y] = 1;
                }
            }
        }

        unsigned int bestV = dpV[0][Y - 1];
        int curx = 0;

        for(int x = 0;x < X;++x){
            if(dpV[x][Y - 1] < bestV){
                bestV = dpV[x][Y - 1];
                curx = x;
            }
        }

        Mat_<Vec3b> tmp(Y,X - 1);

        for(int y = Y - 1;y >= 0;--y){
            posV[y].push_back(curx);

            for(int i = 0;i < X;++i){
                if(i < curx) tmp.at<Vec3b>(y,i) = I.at<Vec3b>(y,i);
                else if(i > curx) tmp.at<Vec3b>(y,i - 1) = I.at<Vec3b>(y,i);
            }

            if(y > 0)
                curx = curx + dirV[curx][y];
        }

        I = tmp;
        --X;
    }

    imshow("seams",seams);
    imshow("seam-carving-out",I);
    waitKey(0);

    return 0;
}