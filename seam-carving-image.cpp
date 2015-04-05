#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

#include <iostream>
#include <fstream>
#include <cstring>

using namespace cv;
using namespace std;

int get(Mat I, int x, int y){
    return (int)I.at<uchar>(y,x);
}

Mat calculate_energy(Mat I){
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

#define MAXR 1000
#define MAXC 1000

Mat gray,energy;
int dpH[MAXR][MAXC],dpV[MAXR][MAXC];
int dirH[MAXR][MAXC],dirV[MAXR][MAXC];

void reduce(Mat &I, int YF, int XF, bool forward=false){
    cout << "REDUCE" << endl;
    int Y0 = I.rows,X0 = I.cols;
    int Y = Y0,X = X0;
    
    pair<int, int> pos[X][Y];

    for(int i = 0;i < X;++i)
        for(int j = 0;j < Y;++j)
            pos[i][j] = make_pair(i,j);

    Mat seams = I.clone();

    // Horizontal seams

    for(int it = 0;it < Y0 - YF;++it){
        cvtColor(I,gray,CV_BGR2GRAY);
        energy = calculate_energy(gray);

        for(int y = 0;y < Y;++y)
            dpH[0][y] = energy.at<int>(y,0);

        for(int x = 1;x < X;++x){
            for(int y = 0;y < Y;++y){
                uint val = energy.at<int>(y,x);
                dpH[x][y] = -1;

                int cost1 = 0,cost2 = 0,cost3 = 0;

                if(!forward){
                    cost1 = val;
                    cost2 = val;
                    cost3 = val;
                }else{
                    if(y > 0 && y + 1 < Y){
                        cost1 = abs(get(gray,x,y - 1) - get(gray,x,y + 1));
                    }else if(y == 0){
                        cost1 = abs(get(gray,x,y) - get(gray,x,y + 1));
                    }else{
                        cost1 = abs(get(gray,x,y - 1) - get(gray,x,y));
                    }
                    
                    cost1 = cost1 + val;
                    cost2 = cost1;
                    cost3 = cost1;

                    if(y > 0)
                        cost1 += abs(get(gray,x,y - 1) - get(gray,x - 1,y));

                    if(y + 1 < Y)
                        cost3 += abs(get(gray,x,y + 1) - get(gray,x - 1,y + 1));
                }

                if(y > 0 && (dpH[x][y] == -1 || cost1 + dpH[x - 1][y - 1] < dpH[x][y])){
                    dpH[x][y] = cost1 + dpH[x - 1][y - 1];
                    dirH[x][y] = -1;
                }

                if(dpH[x][y] == -1 || cost2 + dpH[x - 1][y] < dpH[x][y]){
                    dpH[x][y] = cost2 + dpH[x - 1][y];
                    dirH[x][y] = 0;
                }

                if(y + 1 < Y && (dpH[x][y] == -1 || cost3 + dpH[x - 1][y + 1] < dpH[x][y])){
                    dpH[x][y] = cost3 + dpH[x - 1][y + 1];
                    dirH[x][y] = 1;
                }
            }
        }

        int bestH = dpH[X - 1][0];
        int cury = 0;

        for(int y = 0;y < Y;++y){
            if(dpH[X - 1][y] < bestH){
                bestH = dpH[X - 1][y];
                cury = y;
            }
        }

        Mat_<Vec3b> tmp(Y - 1,X);

        for(int x = X - 1,cont = 0;x >= 0;--x,++cont){
            for(int i = 0;i < Y;++i){
                if(i < cury){
                    tmp.at<Vec3b>(i,x) = I.at<Vec3b>(i,x);
                }else if(i > cury){
                    tmp.at<Vec3b>(i - 1,x) = I.at<Vec3b>(i,x);
                    pos[x][i - 1] = pos[x][i];
                }else{
                    seams.at<Vec3b>(pos[x][i].second, pos[x][i].first) = Vec3b(0,0,255);
                }
            }

            if(x > 0)
                cury = cury + dirH[x][cury];
        }

        I = tmp;
        --Y;
    }

    // Vertical seams

    for(int it = 0;it < X0 - XF;++it){
        cvtColor(I,gray,CV_BGR2GRAY);
        energy = calculate_energy(gray);

        for(int x = 0;x < X;++x)
            dpV[x][0] = energy.at<int>(0,x); 

        for(int y = 1;y < Y;++y){
            for(int x = 0;x < X;++x){
                int val = energy.at<int>(y,x);
                dpV[x][y] = -1;

                int cost1 = 0,cost2 = 0,cost3 = 0;

                if(!forward){
                    cost1 = val;
                    cost2 = val;
                    cost3 = val;
                }else{
                    if(x > 0 && x + 1 < X){
                        cost1 = abs(get(gray,x - 1,y) - get(gray,x + 1,y));
                    }else if(x == 0){
                        cost1 = abs(get(gray,x,y) - get(gray,x + 1,y));
                    }else{
                        cost1 = abs(get(gray,x - 1,y) - get(gray,x,y));
                    }

                    cost2 = cost1;
                    cost3 = cost1;

                    if(x > 0)
                        cost1 += abs(get(gray,x - 1,y) - get(gray,x,y - 1));

                    if(x + 1 < X)
                        cost3 += abs(get(gray,x + 1,y) - get(gray,x,y - 1));
                }

                if(x > 0 && (dpV[x][y] == -1 || cost1 + dpV[x - 1][y - 1] < dpV[x][y])){
                    dpV[x][y] = cost1 + dpV[x - 1][y - 1];
                    dirV[x][y] = -1;
                }

                if(dpV[x][y] == -1 || cost2 + dpV[x][y - 1] < dpV[x][y]){
                    dpV[x][y] = cost2 + dpV[x][y - 1];
                    dirV[x][y] = 0;
                }

                if(x + 1 < X && (dpV[x][y] == -1 || cost3 + dpV[x + 1][y - 1] < dpV[x][y])){
                    dpV[x][y] = cost3 + dpV[x + 1][y - 1];
                    dirV[x][y] = 1;
                }
            }
        }

        int bestV = dpV[0][Y - 1];
        int curx = 0;

        for(int x = 0;x < X;++x){
            if(dpV[x][Y - 1] < bestV){
                bestV = dpV[x][Y - 1];
                curx = x;
            }
        }

        Mat_<Vec3b> tmp(Y,X - 1);

        for(int y = Y - 1;y >= 0;--y){
            for(int i = 0;i < X;++i){
                if(i < curx){
                    tmp.at<Vec3b>(y,i) = I.at<Vec3b>(y,i);
                }else if(i > curx){
                    tmp.at<Vec3b>(y,i - 1) = I.at<Vec3b>(y,i);
                    pos[i - 1][y] = pos[i][y];
                }else{
                    seams.at<Vec3b>(pos[i][y].second, pos[i][y].first) = Vec3b(0,0,255);
                }
            }

            if(y > 0)
                curx = curx + dirV[curx][y];
        }

        I = tmp;
        --X;
    }

    string w1 = "seams",w2 = "seam-carving-out";

    if(forward){
        w1 += "-forward";
        w2 += "-forward";
    }

    imshow(w1,seams);

    if(forward)
        imwrite("seams-forward.jpg", seams);
    else
        imwrite("seams.jpg", seams);

    imshow(w2,I);

    if(forward)
        imwrite("result-forward.jpg",I);
    else
        imwrite("result.jpg",I);

    waitKey(0);
}

void remove_horizontal(Mat &I, int YF){
    int Y0 = I.rows;
    int Y = Y0,X = I.cols;

    for(int it = 0;it < Y0 - YF;++it){
        cvtColor(I,gray,CV_BGR2GRAY);
        energy = calculate_energy(gray);

        for(int y = 0;y < Y;++y)
            dpH[0][y] = energy.at<int>(y,0);

        for(int x = 1;x < X;++x){
            for(int y = 0;y < Y;++y){
                uint val = energy.at<int>(y,x);
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

        int bestH = dpH[X - 1][0];
        int cury = 0;

        for(int y = 0;y < Y;++y){
            if(dpH[X - 1][y] < bestH){
                bestH = dpH[X - 1][y];
                cury = y;
            }
        }

        Mat_<Vec3b> tmp(Y - 1,X);

        for(int x = X - 1,cont = 0;x >= 0;--x,++cont){
            for(int i = 0;i < Y;++i){
                if(i < cury){
                    tmp.at<Vec3b>(i,x) = I.at<Vec3b>(i,x);
                }else if(i > cury){
                    tmp.at<Vec3b>(i - 1,x) = I.at<Vec3b>(i,x);
                }
            }

            if(x > 0)
                cury = cury + dirH[x][cury];
        }

        I = tmp;
        --Y;
    }
}

void remove_vertical(Mat &I, int XF){
    int X0 = I.cols;
    int X = X0,Y = I.rows;

    for(int it = 0;it < X0 - XF;++it){
        cvtColor(I,gray,CV_BGR2GRAY);
        energy = calculate_energy(gray);

        for(int x = 0;x < X;++x)
            dpV[x][0] = energy.at<int>(0,x);

        for(int y = 1;y < Y;++y){
            for(int x = 0;x < X;++x){
                int val = energy.at<int>(y,x);
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

        int bestV = dpV[0][Y - 1];
        int curx = 0;

        for(int x = 0;x < X;++x){
            if(dpV[x][Y - 1] < bestV){
                bestV = dpV[x][Y - 1];
                curx = x;
            }
        }

        Mat_<Vec3b> tmp(Y,X - 1);

        for(int y = Y - 1;y >= 0;--y){
            for(int i = 0;i < X;++i){
                if(i < curx){
                    tmp.at<Vec3b>(y,i) = I.at<Vec3b>(y,i);
                }else if(i > curx){
                    tmp.at<Vec3b>(y,i - 1) = I.at<Vec3b>(y,i);
                }
            }

            if(y > 0)
                curx = curx + dirV[curx][y];
        }

        I = tmp;
        --X;
    }
}

Vec3b average(Vec3b x, Vec3b y){
    Vec3b ret;

    for(int i = 0;i < 3;++i)
        ret.val[i] = (x.val[i] + y.val[i]) / 2;
    
    return ret;
}

void add_horizontal(Mat &I, int YF){
    Mat I0 = I;
    int Y0 = I.rows;
    int Y = Y0,X = I.cols;
    bool mark[Y][X];
    int pos[X][Y];

    memset(mark,false,sizeof mark);

    for(int i = 0;i < X;++i)
        for(int j = 0;j < Y;++j)
            pos[i][j] = j;

    for(int it = 0;it < YF - Y0;++it){
        cvtColor(I,gray,CV_BGR2GRAY);
        energy = calculate_energy(gray);

        for(int y = 0;y < Y;++y)
            dpH[0][y] = energy.at<int>(y,0);

        for(int x = 1;x < X;++x){
            for(int y = 0;y < Y;++y){
                uint val = energy.at<int>(y,x);
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

        int bestH = dpH[X - 1][0];
        int cury = 0;

        for(int y = 0;y < Y;++y){
            if(dpH[X - 1][y] < bestH){
                bestH = dpH[X - 1][y];
                cury = y;
            }
        }

        Mat_<Vec3b> tmp(Y - 1,X);

        for(int x = X - 1,cont = 0;x >= 0;--x,++cont){
            for(int i = 0;i < Y;++i){
                if(i < cury){
                    tmp.at<Vec3b>(i,x) = I.at<Vec3b>(i,x);
                }else if(i > cury){
                    tmp.at<Vec3b>(i - 1,x) = I.at<Vec3b>(i,x);
                    pos[x][i - 1] = pos[x][i];
                }else{
                    mark[ pos[x][i] ][x] = true;
                }
            }

            if(x > 0)
                cury = cury + dirH[x][cury];
        }

        I = tmp;
        --Y;
    }

    Mat_<Vec3b> tmp(YF,X);

    for(int j = 0;j < X;++j){
        int cont = 0;

        for(int i = 0;i < Y0;++i){
            if(mark[i][j]){
                Vec3b aux;

                if(i == 0) aux = average(I0.at<Vec3b>(i,j),I0.at<Vec3b>(i + 1,j));
                else if(i == Y0 - 1) aux = average(I0.at<Vec3b>(i,j),I0.at<Vec3b>(i - 1,j));
                else aux = average(I0.at<Vec3b>(i - 1,j),I0.at<Vec3b>(i + 1,j));

                tmp.at<Vec3b>(cont,j) = aux; cont++;
                tmp.at<Vec3b>(cont,j) = aux; cont++;
            }else{
                tmp.at<Vec3b>(cont,j) = I0.at<Vec3b>(i,j);
                cont++;
            }
        }
    }

    I = tmp;
}

void add_vertical(Mat &I, int XF){
    Mat I0 = I;
    int X0 = I.cols;
    int X = X0,Y = I.rows;
    bool mark[Y][X];
    int pos[X][Y];

    memset(mark,false,sizeof mark);

    for(int i = 0;i < X;++i)
        for(int j = 0;j < Y;++j)
            pos[i][j] = i;

    for(int it = 0;it < XF - X0;++it){
        cvtColor(I,gray,CV_BGR2GRAY);
        energy = calculate_energy(gray);

        for(int x = 0;x < X;++x)
            dpV[x][0] = energy.at<int>(0,x);

        for(int y = 1;y < Y;++y){
            for(int x = 0;x < X;++x){
                int val = energy.at<int>(y,x);
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

        int bestV = dpV[0][Y - 1];
        int curx = 0;

        for(int x = 0;x < X;++x){
            if(dpV[x][Y - 1] < bestV){
                bestV = dpV[x][Y - 1];
                curx = x;
            }
        }

        Mat_<Vec3b> tmp(Y,X - 1);

        for(int y = Y - 1;y >= 0;--y){
            for(int i = 0;i < X;++i){
                if(i < curx){
                    tmp.at<Vec3b>(y,i) = I.at<Vec3b>(y,i);
                }else if(i > curx){
                    tmp.at<Vec3b>(y,i - 1) = I.at<Vec3b>(y,i);
                    pos[i - 1][y] = pos[i][y];
                }else{
                    mark[y][ pos[i][y] ] = true;
                }
            }

            if(y > 0)
                curx = curx + dirV[curx][y];
        }
        I = tmp;
        --X;
    }

    Mat_<Vec3b> tmp(Y,XF);

    for(int i = 0;i < Y;++i){
        int cont = 0;

        for(int j = 0;j < X0;++j){
            if(mark[i][j]){
                Vec3b aux;

                if(j == 0) aux = average(I0.at<Vec3b>(i,j),I0.at<Vec3b>(i,j + 1));
                else if(j == X0 - 1) aux = average(I0.at<Vec3b>(i,j),I0.at<Vec3b>(i,j - 1));
                else aux = average(I0.at<Vec3b>(i,j - 1),I0.at<Vec3b>(i,j + 1));

                tmp.at<Vec3b>(i,cont) = aux; cont++;
                tmp.at<Vec3b>(i,cont) = aux; cont++;
            }else{
                tmp.at<Vec3b>(i,cont) = I0.at<Vec3b>(i,j);
                cont++;
            }
        }
    }

    I = tmp;
}

void process(Mat &I, int YF, int XF){
    cout << "Process (" << I.rows << ", " << I.cols << ") -> (" << YF << ", " << XF << ")" << endl;

    if(YF < I.rows)
        remove_horizontal(I,YF);
    else if(YF > I.rows)
        add_horizontal(I,YF);

    if(XF < I.cols)
        remove_vertical(I,XF);
    else if(XF > I.cols)
        add_vertical(I,XF);
}

int main(){
    freopen("../input-image.txt","r",stdin);

    string file;
    cin >> file;
    cout << file << endl;

    Mat_<Vec3b> I = imread(file);

    imshow("seam-carving",I);
    waitKey(0);

    int Y0 = I.rows,X0 = I.cols,dy,dx;

    cout << "Original dimensions: Rows = " << Y0 << " Cols = " << X0 << '\n';

    cin >> dy;
    int YF = Y0 + dy;

    cin >> dx;
    int XF = X0 + dx;

    cout << "Desired dimension: Rows = " << YF << " Cols = " << XF << '\n';

    if(YF < Y0 && XF < X0){
        Mat_<Vec3b> I2 = I.clone();
        reduce(I,YF,XF);
        // reduce with forward energy
        reduce(I2,YF,XF,true);
    }else{
        process(I,YF,XF);
        imshow("seam-carving-out",I);
        waitKey(0);

        process(I,Y0,X0);
        imshow("seam-carving-out-2",I);
        waitKey(0);
    }

    fclose(stdin);

    return 0;
}
