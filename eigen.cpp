#include <cxcore.h>
#include <highgui.h>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace cv;
using namespace std;

const int NUMERO_DE_PASTAS = 40;
const int NUMERO_DE_FOTOS_POR_PASTA = 10;

// Reads the images and labels from a given CSV file, a valid file would
// look like this:
//
//      /path/to/person0/image0.jpg;0
//      /path/to/person0/image1.jpg;0
//      /path/to/person1/image0.jpg;1
//      /path/to/person1/image1.jpg;1
//      ...
//
void read_csv(const string& filename, vector<Mat>& images, vector<int>& labels) {
    std::ifstream file(filename.c_str(), ifstream::in);
    if(!file)
        throw std::exception();
    std::string line, path, classlabel;
    // For each line in the given file:
    while (std::getline(file, line)) {
        // Get the current line:
        std::stringstream liness(line);
        // Split it at the semicolon:
        std::getline(liness, path, ';');
        std::getline(liness, classlabel);
        // And push back the data into the result vectors:
        images.push_back(imread(path, IMREAD_GRAYSCALE));
        labels.push_back(atoi(classlabel.c_str()));
    }
}

// Normalizes a given image into a value range between 0 and 255.
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

// Converts the images given in src into a row matrix.
Mat asRowMatrix(const vector<Mat>& src, int rtype, double alpha = 1, double beta = 0) {
    // Number of samples:
    size_t n = src.size();
    // Return empty matrix if no matrices given:
    if(n == 0)
        return Mat();
    // dimensionality of (reshaped) samples
    size_t d = src[0].total();
    // Create resulting data matrix:
    Mat data(n, d, rtype);
    // Now copy data:
    for(unsigned i = 0; i < n; i++) {
        //
        if(src[i].empty()) {
            string error_message = format("Image number %d was empty, please check your input data.", i);
            CV_Error(CV_StsBadArg, error_message);
        }
        // Make sure data can be reshaped, throw a meaningful exception if not!
        if(src[i].total() != d) {
            string error_message = format("Wrong number of elements in matrix #%d! Expected %d was %d.", i, d, src[i].total());
            CV_Error(CV_StsBadArg, error_message);
        }
        // Get a hold of the current row:
        Mat xi = data.row(i);
        // Make reshape happy by cloning for non-continuous matrices:
        if(src[i].isContinuous()) {
            src[i].reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        } else {
            src[i].clone().reshape(1, 1).convertTo(xi, rtype, alpha, beta);
        }
    }
    return data;
}

int main(int argc, const char *argv[]) {
    // Holds some images:
    vector<Mat> db;

    // Load the greyscale images. The images in the example are
    // taken from the AT&T Facedatabase, which is publicly available
    // at:
    //
    //      http://www.cl.cam.ac.uk/research/dtg/attarchive/facedatabase.html
    //
    // This is the path to where I stored the images, yours is different!
    //
    string prefix = "/home/levy/att_faces/";

    string diretorio = "/home/levy/att_faces/s";
    vector <float> vec;
    int index = 0;
    for (int j = 1; j <= NUMERO_DE_PASTAS ; j++) {
        for (int i = 1; i <= NUMERO_DE_FOTOS_POR_PASTA; i++){
            stringstream ss;
            ss << j << "/" << i << ".pgm";
            string s = ss.str();
            db.push_back(imread(diretorio + s , 0));
            //im2double(img);
            //transformMatrix(img, vec); //Função para  transformar a imagem em um vetor
            //addColumnVectorInMatrix(matrix, vec, index); //adicionar o vetor coluna na matriz principal
            index++;

        }
    }

    /*db.push_back(imread(prefix + "s1/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s1/3.pgm", IMREAD_GRAYSCALE));

    db.push_back(imread(prefix + "s2/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s2/3.pgm", IMREAD_GRAYSCALE));

    db.push_back(imread(prefix + "s3/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s3/3.pgm", IMREAD_GRAYSCALE));

    db.push_back(imread(prefix + "s4/1.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/2.pgm", IMREAD_GRAYSCALE));
    db.push_back(imread(prefix + "s4/3.pgm", IMREAD_GRAYSCALE));*/


    // The following would read the images from a given CSV file
    // instead, which would look like:
    //
    //      /path/to/person0/image0.jpg;0
    //      /path/to/person0/image1.jpg;0
    //      /path/to/person1/image0.jpg;1
    //      /path/to/person1/image1.jpg;1
    //      ...
    //
    // Uncomment this to load from a CSV file:
    //

    /*
    vector<int> labels;
    read_csv("/home/philipp/facerec/data/at.txt", db, labels);
    */

    // Build a matrix with the observations in row:
    Mat data = asRowMatrix(db, CV_32FC1);

    // Number of components to keep for the PCA:
    int num_components = 60;

    // Perform a PCA:
    PCA pca(data, Mat(), CV_PCA_DATA_AS_ROW, num_components);

    // And copy the PCA results:
    Mat mean = pca.mean.clone();
    Mat eigenvalues = pca.eigenvalues.clone();
    Mat eigenvectors = pca.eigenvectors.clone();

    // The mean face:
    imshow("avg", norm_0_255(mean.reshape(1, db[0].rows)));

    // The first three eigenfaces:
    for(int i= 0; i < 20; i++){
        stringstream ss;
        ss << i+1 ;
        string s = ss.str();
        Mat teste = norm_0_255(pca.eigenvectors.row(i)).reshape(1, db[0].rows);
        imshow(s, norm_0_255(pca.eigenvectors.row(i)).reshape(1, db[0].rows));
    }

    /*imshow("pc1", norm_0_255(pca.eigenvectors.row(0)).reshape(1, db[0].rows));
    imshow("pc2", norm_0_255(pca.eigenvectors.row(1)).reshape(1, db[0].rows));
    imshow("pc3", norm_0_255(pca.eigenvectors.row(2)).reshape(1, db[0].rows));*/

    // Show the images:
    waitKey(0);

    // Success!
    return 0;
}
