#include<bits/stdc++.h>
#include "opencv2/opencv.hpp"
#include <opencv2/tracking.hpp>
//#include "armadillo"

using namespace std;
using namespace cv;
//using namespace arma;

vector <Mat> vec; //Cada elemento do vetor representa uma feature
int features [30];

std::pair<std::vector<Mat>, std::vector< std::vector<Mat>>> generateIntImgs(const std::vector<Mat> &featureImgs) {
        //PRE:
    assert( (featureImgs.size() > 0) && (featureImgs.at(0).type() == CV_32FC1) );

        //generate integral images of features
    std::vector<Mat> regIntImgs;

    for (std::vector<Mat>::const_iterator it=featureImgs.begin(); it!=featureImgs.end(); ++it) {
        Mat intImg;
        integral(*it, intImg, CV_64FC1);
        regIntImgs.push_back(intImg);
    }

        //generate integral images of each permutation of two multiplied features
    std::vector< std::vector<Mat> > sqIntImgs;

    for (std::vector<Mat>::const_iterator it=featureImgs.begin(); it!=featureImgs.end(); ++it){
        std::vector<Mat> sqIntRow;

            //since sqIntImgs is symmetric matrix, only process unique elements
        const std::size_t rowCount = it - featureImgs.begin();
        for (std::vector<Mat>::const_iterator jt=featureImgs.begin(); jt<=featureImgs.begin()+rowCount; ++jt) {
            const Mat sqImg = it->mul(*jt);		//per element multiplication operation

            Mat sqIntImg;
            integral(sqImg, sqIntImg, CV_64FC1);
            sqIntRow.push_back(sqIntImg);
        }

        sqIntImgs.push_back(sqIntRow);
    }

    const std::pair<std::vector<Mat>, std::vector< std::vector<Mat> > > intImgs = std::make_pair(regIntImgs, sqIntImgs);


    return intImgs;
}


void storeObj(const Mat &img){

    //guardar o size do objeto

    //generate the feature images and their integrals for the whole object
    //const std::vector<Mat> objFeatureImgs = generateFeatures(obj);
    //const std::pair<std::vector<Mat>, std::vector< std::vector<Mat>>> objIntImgs = generateIntImgs(objFeatureImgs);
}

vector<Mat> generateFeatures(const Mat &img){

    std::vector<Mat> features;
    std::vector<Mat> bgrImgs;
    split(img, bgrImgs);

    for (int i=bgrImgs.size()-1; i>=0; --i) {
        Mat floatImg;
        bgrImgs.at(i).convertTo(floatImg, CV_32FC1);
        features.push_back(floatImg);
    }

        //calculate image gradients
    Mat greyImg;
    cvtColor(img, greyImg, CV_BGR2GRAY);

    Mat dx, dy, d2x, d2y;
    Sobel(greyImg, dx, CV_32FC1, 1, 0, 1);
    Sobel(greyImg, dy, CV_32F, 0, 1, 1);
    Sobel(greyImg, d2x, CV_32F, 2, 0, 1);
    Sobel(greyImg, d2y, CV_32F, 0, 2, 1);

    features.push_back(dx);
    features.push_back(dy);
    features.push_back(d2x);
    features.push_back(d2y);

    assert((features.size() > 0) && (features.at(0).type() == CV_32FC1));

    return features;
}

void covarianceDistance(Mat cov1, Mat cov2){

    double EPS = 1e-6;
    //eigen =  compute generalized eigenvalues(cov1+ EPS, cov2 + EPS)

    //dist = sum(log(diag(abs(eigen))));
}

int main() {

    cv:: Mat image = imread("/home/levy/OpencvData/lena.png");

    Rect2d rec = selectROI(image);

    cv:: imshow("Crop", image(rec));
    waitKey(0);
    return 0;
}
