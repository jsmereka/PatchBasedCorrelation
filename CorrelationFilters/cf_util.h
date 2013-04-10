#include <Eigen/Dense>

using namespace Eigen;

int get_rank(MatrixXd &signal);

void fft2_image_scalar(MatrixXd &img, MatrixXcd &img_freq, int siz1, int siz2);

void ifft2_image_scalar(MatrixXd &img, MatrixXcd &img_freq);

void fft2_signal_scalar(VectorXd &img, VectorXcd &img_freq, int siz1);

void ifft2_signal_scalar(VectorXd &img, VectorXcd &img_freq);