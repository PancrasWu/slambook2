#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

const int N = 10; // 数据点数量

/*  函数声明  */
void GN(double *, double *, double *);
Mat jacobi(const Mat &, const Mat &);
Mat yEstimate(const Mat &, const Mat &);

int main(int argc, char **argv)
{
    double ar = 1.0, br = 2.0, cr = 1.0; // 真实参数值
    double est[] = {2.0, 4.0, 0.0};       // 估计参数值
    double w_sigma = 1.0;                 // 噪声Sigma值
    cv::RNG rng;                          // OpenCV随机数产生器

    double x_data[N], y_data[N]; // 生成真值数据
    for (int i = 0; i < N; i++)
    {
        double x = i /5.0;
        x_data[i] = x;
        y_data[i] =  exp(ar * x * x +br * x + cr) + rng.gaussian(w_sigma * w_sigma);
    }

    chrono::steady_clock::time_point t1 = chrono::steady_clock::now();
    GN(x_data, y_data, est);
    chrono::steady_clock::time_point t2 = chrono::steady_clock::now();
    chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>(t2 - t1);

    cout << "solve time cost = " << time_used.count() << " seconds. " << endl;
    return 0;
}

///高斯牛顿法
void GN(double *x, double *y, double *est0)
{
    int iterations = 20;                                     // 迭代次数
    double cost = 0, lastCost = 0;                           // 本次迭代的cost和上一次迭代的cost
    Mat_<double> xM(1, N, x), yM(1, N, y), estM(3, 1, est0); //x矩阵，y矩阵，参数矩阵，
    Mat_<double> jacobiM, estYM, errorM, bM, dxM;            //雅可比矩阵，评估值Y，误差矩阵，b值矩阵，deltaX矩阵
    for (int iter = 0; iter < iterations; iter++)
    {
        jacobiM = jacobi(estM, xM);
        cout<<"jacobiM is : "<< jacobiM.rows<<"__"<<jacobiM.cols<<endl;
        estYM = yEstimate(estM, xM);
        cout<<"estYM is : "<< estYM.rows<<"__"<<estYM.cols<<endl;
        errorM = yM - estYM;                     // e
        cost = errorM.dot(errorM);               // e^2
        bM = -jacobiM * errorM.t();               // b
        Mat_<double> HM = jacobiM * jacobiM.t(); // H
        cout<<"HM is : "<< HM.rows<<"__"<<HM.cols<<endl;
        cout<<"bM is : "<< bM.rows<<"__"<<bM.cols<<endl;
        if (solve(HM, bM, dxM))                  // 求解Hx = b  
        {
            if (isnan(dxM.at<double>(0)))
            {
                cout << "result is nan!" << endl;
                break;
            }
            if (iter > 0 && cost >= lastCost)
            {
                cout << "iteration: " << iter + 1 << ",cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
                cout << "THe Value, x: " << estM.at<double>(0) << ",y:" << estM.at<double>(1) << ",c:" << estM.at<double>(2) << endl;
                break;
            }
            estM = estM + dxM;
            lastCost = cost;
        }
        else
        {
            cout << "can not solve the function ." << endl;
        }
    }
}

/// est：估计值，X：X值
Mat jacobi(const Mat &est, const Mat &x)
{
    Mat_<double> J(est.rows,x.cols ), da, db, dc,A; //a,b,c的导数
    exp(x.mul(est.at<double>(0,0)*x)+est.at<double>(1,0)*x + est.at<double>(2,0),A);
    da = -x.mul( x.mul (A));
    db = -x .mul( A);
    dc = -A;
    da.copyTo(J(Rect(0, 0, J.cols,1)));
    db.copyTo(J(Rect(0, 1, J.cols,1)));
    dc.copyTo(J(Rect(0, 2, J.cols,1)));
    return J;
}
///计算 y = exp(ax^2+bx + c)
Mat yEstimate(const Mat &est, const Mat &x)
{
    Mat_<double> Y(x.rows, x.cols);
    exp(x.mul(est.at<double>(0,0)*x)+est.at<double>(1,0)*x + est.at<double>(2,0),Y);
    return Y;
}

