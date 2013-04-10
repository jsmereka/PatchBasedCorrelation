#include <Eigen/Dense>

using namespace Eigen;

int get_rank(MatrixXd &signal);

void fft2_image_scalar(MatrixXd &img, MatrixXcd &img_freq, int siz1, int siz2);

void ifft2_image_scalar(MatrixXd &img, MatrixXcd &img_freq);