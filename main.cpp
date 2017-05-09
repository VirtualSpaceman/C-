// DetectBlobs.cpp : Defines the entry point for the console application.
//
#include <iostream>
#include <cv.h>
#include <cxcore.h>
#include <highgui.h>
#include <string.h>
#include <string>
#include <time.h>

using namespace cv;
using namespace std;

const int NUMERO_DE_LINHAS_FOTO = 112;
const int NUMERO_DE_COLUNAS_FOTO = 92;
const int NUMERO_DE_PASTAS = 40;
const int NUMERO_DE_FOTOS_POR_PASTA = 10;
const int NUMERO_DE_FOTOS = 400;
const int NUMERO_COEFICIENTES_OTIMO = 60;
const int NUMERO_COEFICIENTES_MEDIO = 40;
const int NUMERO_COEFICIENTES_BAIXO = 20;



Mat norm_0_255(const Mat& src) {
    // Create and return normalized image:
    Mat dst;
    switch(src.channels()) {
    case 1:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC1);
        break;
    case 3:
        cv::normalize(src, dst, 0, 255, NORM_MINMAX, CV_8UC3);
        break;
    default:
        src.copyTo(dst);
        break;
    }
    return dst;
}

void fillTrainingGroupMatrix(Mat &mat);
void im2double( Mat &image);
void computeCovMatrix( Mat &image, Mat &cov, Mat &mu);
void transformMatrix (Mat image, vector<float> &cvec);
void rebuildMatrix(Mat mat, String name);
void addColumnVectorInMatrix(Mat &image, vector <float> &vec, int columnIndex);
void testeMatrix(Mat matrix, Mat matMenor, int boundaries);
void testeCompression(Mat u, Mat s, Mat vt,Mat mean, int numeroDeComponentes);
void fillMatrix(Mat &matrix);

int main(){

    /*Mat_<float> samples1 = (Mat_<float>(3, 3) << 500.0, 350.2, 942.8,
                                            500.5, 355.8, 625.3,
                                            498.7, 352.0, 674.6);
    Mat_<float> samples2 = (Mat_<float>(3, 3) << 800.0, 350.2, 65.3,
                                            560.5, 3532.8, 984.6,
                                            498.7, 365.0, 95.0);*/
    Mat matrix = Mat(NUMERO_DE_LINHAS_FOTO*NUMERO_DE_COLUNAS_FOTO, NUMERO_DE_PASTAS*(NUMERO_DE_FOTOS_POR_PASTA-1), CV_32F);
    Mat covMatrix, mu;
    Mat w,u,vt; // W - singular values, u - left singular values, vt - transposed matrix of right singular values
    Mat matrixNormalizada;
    //Mat teste = Mat(400, NUMERO_COEFICIENTES_OTIMO, CV_32F); // Dependendo do numero de coef. Farei uma matriz
    //Mat teste = Mat(NUMERO_COEFICIENTES_OTIMO, 400, CV_32F);
    //para guardar os direções que maximizam a variância

    fillMatrix(matrix);

    Mat groupTraing = Mat(10304, 40, CV_32F);

    fillTrainingGroupMatrix(groupTraing);

    rebuildMatrix(groupTraing.col(27), "8");

    //Coloca as medias calculadas em colunas.Ex: media da coluna 1 = todas as linhas da coluna 1 contem a média da foto 1
    Mat dasMedias = Mat(matrix.rows, matrix.cols, CV_32F);
    for(int i=0; i<matrix.cols; i++){
        Mat vec2;
        Mat vec3;
        meanStdDev(matrix.col(i),vec2, vec3, noArray());
        for(int j= 0; j < matrix.rows; j++){
            dasMedias.at<float>(j,i) = vec2.at<double>(0,0);
        }
    }
    matrixNormalizada = matrix - dasMedias; //Matriz normalizada contendo todas a imagens armazenadas em colunas



    /*Mat ones2 = Mat ::ones(1, NUMERO_DE_FOTOS, CV_32F);
    Mat mds = dasMedias*ones2;*/ //Matriz contendo as medias de cada coluna

    /*cout << "alo" << endl;
    Mat coef;
    PCA teste = PCA(matrix, noArray(), 1, NUMERO_COEFICIENTES_OTIMO);
    coef = teste.project(cof);*/


    /*Mat imgsSemMedia;
    Mat ones = Mat::ones(matrix.rows, matrix.cols, CV_32F);
    Mat meanDiag = meanDiag.diag(mds.diag(0));
    Mat result = ones*meanDiag;
    imgsSemMedia = matrix - mds;
    imgsSemMedia = matrix - result; //Matriz normalizada
    normalize(matrix, imgsSemMedia, 0, 1, NORM_L2, -1, noArray());*/


    /*computeCovMatrix(, covMatrix, meanDiag);
    cout << "dps" << endl;
    cout << mu.rows << " " << mu.cols << endl;
    imgSemMedia = matrix - mu;
    cout << "dps" << endl;
    cout << covMatrix << endl;
    cout << covMatrix.rows << " " << covMatrix.rows << endl;
    cout << "chegou até aqui" << endl;*/

    cout << "teste" << endl;
    //SVD::compute(matrixNormalizada, w, u, vt, 0);
    cout << "Fez a SVD" << endl;
    Mat coef;
    PCA teste = PCA(matrix, noArray(), 1, NUMERO_COEFICIENTES_OTIMO);
    cout << "teste" << endl;

    coef = teste.project(groupTraing);
    cout << coef << endl;

    //esteCompression(u, w, vt, mds, NUMERO_COEFICIENTES_OTIMO);
    //cout << u.col(0).rows << "x" << u.col(0).cols << endl;


    /*
    for(int i=1; i <= 10; i++){
        vectorToMatrix(u.col(i),i);
    }
    cout << w.rows <<  " Matriz W " << w.cols << endl;
    cout << u.rows <<  " Matriz U " << u.cols << endl;
    cout << vt.rows <<  " Matriz Vt " << vt.cols << endl;
    */

    waitKey(0);

    return 0;
}

void im2double(Mat &image){

    image.convertTo( image, CV_32F, 1.0/255);
}

void computeCovMatrix( Mat &image, Mat &cov, Mat &mu){
    calcCovarMatrix(image, cov, mu, CV_COVAR_NORMAL | CV_COVAR_ROWS);

    cov = cov / (image.rows - 1);
}

void transformMatrix (Mat image, vector<float> &vec){
    //Essa função coloca toda a matriz em um vetor.
    for(int i=0; i<image.cols; i++){
        for(int j=0; j<image.rows; j++){
            vec.push_back(image.at<float>(j,i));
        }
    }
}

void addColumnVectorInMatrix(Mat &image, vector <float> &vec, int columnIndex){
    for(int i=0; i < image.rows; i++){
        image.at<float>(i,columnIndex) = vec.at(0);
        vec.erase(vec.begin());
    }
}

void testeMatrix(Mat matrix, Mat matMenor, int boundaries){
    for (int linha = 0; linha < boundaries ; linha++){
        for(int coluna = 0; coluna < matrix.cols; coluna++){
            matMenor.at<float>(linha,coluna) = matrix.at<float>(linha,coluna);
        }
    }
    //TROCAR OS VALORE AQUI, POIS TROQUEI OS VALORES DE LINHA COM COLUNA , OLD - L = BOUNDARIES, C = MATRIX. COLS
}

void testeCompression(Mat u, Mat s, Mat vt,Mat mean,int numeroDeComponentes){
    Mat uMenor = Mat(u.rows,numeroDeComponentes, CV_32F);
    Mat sMenor = Mat(numeroDeComponentes,numeroDeComponentes, CV_32F);
    Mat vtMenor= Mat(numeroDeComponentes,vt.cols, CV_32F);

    for( int row = 0; row < u.rows; row++){
        for(int col = 0 ; col < numeroDeComponentes; col++){
            uMenor.at<float>(row,col) = u.at<float>(row,col);
        }
    }

    for( int row = 0; row < numeroDeComponentes; row++){
        for(int col = 0 ; col < numeroDeComponentes; col++){
            if(row != col){
                sMenor.at<float>(row,col) = 0;
            }
            else
                sMenor.at<float>(row,col) = s.at<float>(0,col);
        }
    }

    for( int row = 0; row < numeroDeComponentes; row++){
        for(int col = 0 ; col < vt.cols; col++){
            vtMenor.at<float>(row,col) = vt.at<float>(row,col);
        }
    }

    cout << sMenor << endl;


    Mat compressed = uMenor*sMenor*vtMenor + mean;
    imshow("Compressed", compressed);
}

void fillMatrix(Mat &matrix){

    string diretorio = "/home/levy/att_faces/s";
    vector <float> vec;
    int index = 0;
    for (int j = 1; j <= NUMERO_DE_PASTAS ; j++) {
        for (int i = 2; i <= NUMERO_DE_FOTOS_POR_PASTA; i++){
            stringstream ss;
            ss << j << "/" << i << ".pgm";
            string s = ss.str();
            Mat img = imread(diretorio + s , 0);
            //im2double(img);
            transformMatrix(img, vec); //Função para  transformar a imagem em um vetor
            addColumnVectorInMatrix(matrix, vec, index); //adicionar o vetor coluna na matriz principal
            index++;

        }
    }
}

void rebuildMatrix(Mat mat, string i){
    //Transforma um vetor em uma matriz 112x92, onde os primeiros 112 elementos são da coluna 1 e assim sucessivamente
    Mat imagem = Mat(112, 92, CV_32F);
    vector <float> vec;
    transformMatrix(mat, vec); // Coloca matriz em um vetor
    for(int col=0; col < imagem.cols; col++){
        for(int row=0; row < imagem.rows; row++){
            imagem.at<float>(row, col) = vec.at(0);
            vec.erase(vec.begin());
        }
    }
    imshow( i , imagem);
}

void fillTrainingGroupMatrix(Mat &mat){

    string diretorio = "/home/levy/faces/";
    vector <float> vec;
    int index = 0;
    for (int i = 1; i <= mat.cols ; i++) {
        stringstream ss;
        ss << "/" << i << ".pgm";
        string s = ss.str();
        Mat img = imread(diretorio + s , 0);
        im2double(img);
        transformMatrix(img, vec); //Função para  transformar a imagem em um vetor
        addColumnVectorInMatrix(mat, vec, index); //adicionar o vetor coluna na matriz principal
        index++;

    }

}
