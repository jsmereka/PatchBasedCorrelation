#include "stdafx.h"


/**
  
  Get the matrix rank

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    signal		the 2-d signal to be checked
  
  @return   matrix rank

*/
int get_rank(MatrixXd &signal) {
	int rank = 0;
	Eigen::JacobiSVD<MatrixXd> svd(signal,0);
	VectorXd vals = svd.singularValues();
	for(int i = 0; i<vals.size(); i++) {
		if(vals(i) != 0) {
			rank++; // count non-zero values
		}
	}
	return rank;
}



/**
  
  Put 2-d data into the frequency domain

  @author   Vishnu Naresh Boddetti & Jonathon M. Smereka
  @version  04-03-2013

  @param    img			the 2-d signal in spatial domain
  @param    img_freq	the 2-d signal in frequency domain
  @param    siz1		rows
  @param    siz2		cols
  
  @return   passed by ref to return data in img_freq

*/
void fft2_image_scalar(MatrixXd &img, MatrixXcd &img_freq, int siz1, int siz2) {
	int count = 0;
	int N = img.rows();
	int M = img.cols();

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);

	for (int i=0;i<N;i++){
		for (int j=0;j<M;j++){
			mat1[count][0] = img(i,j);
			mat1[count][1] = 0;
			count++;
		}
	}

	plan = fftw_plan_dft_2d(siz1,siz2,mat1,mat2,FFTW_FORWARD, FFTW_ESTIMATE);	

	fftw_execute(plan);		
	MatrixXd temp1(siz1,siz2), temp2(siz1,siz2);

	count = 0;
	for (int i=0;i<siz1;i++){
		for (int j=0;j<siz2;j++){				
			temp1(i,j) = mat2[count][0];
			temp2(i,j) = mat2[count][1];
			count++;
		}
	}
	img_freq.resizeLike(temp1);
	img_freq.real() = temp1;
	img_freq.imag() = temp2;
	fftw_destroy_plan(plan);
	delete [] mat1;
	delete [] mat2;
}



/**
  
  Put 2-d data into spatial domain from frequency domain

  @author   Vishnu Naresh Boddetti & Jonathon M. Smereka
  @version  04-03-2013

  @param    img			the 2-d signal in spatial domain
  @param    img_freq	the 2-d signal in frequency domain
  
  @return   passed by ref to return data in img

*/
void ifft2_image_scalar(MatrixXd &img, MatrixXcd &img_freq) {
	int count;
	int siz1 = img_freq.rows();
	int siz2 = img_freq.cols();

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);

	count = 0;
	for (int i=0;i<siz1;i++){
		for (int j=0;j<siz2;j++){
			mat1[count][0] = img_freq(i,j).real();
			mat1[count][1] = img_freq(i,j).imag();
			count++;
		}
	}

	plan = fftw_plan_dft_2d(siz1,siz2,mat1,mat2,FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	count = 0;
	for (int i=0;i<siz1;i++){
		for (int j=0;j<siz2;j++){
			img(i,j) = mat2[count][0]/(siz1*siz2);
			count++;
		}
	}

	fftw_destroy_plan(plan);
	delete [] mat1;
	delete [] mat2;
}



/**
  
  Put 1-d data into the frequency domain

  @author   Vishnu Naresh Boddetti & Jonathon M. Smereka
  @version  04-03-2013

  @param    img			the 1-d signal in spatial domain
  @param    img_freq	the 1-d signal in frequency domain
  @param    siz1		size
  
  @return   passed by ref to return data in img_freq

*/
void fft2_signal_scalar(VectorXd &img, VectorXcd &img_freq, int siz1) {
	int N = img.size();

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1);

	for (int i=0;i<N;i++){
		mat1[i][0] = img(i);
		mat1[i][1] = 0;
	}

	plan = fftw_plan_dft_1d(siz1,mat1,mat2,FFTW_FORWARD, FFTW_ESTIMATE);	

	fftw_execute(plan);		
	VectorXd temp1(siz1), temp2(siz1);

	for (int i=0;i<siz1;i++){			
		temp1(i) = mat2[i][0];
		temp2(i) = mat2[i][1];
	}

	img_freq.resizeLike(temp1);
	img_freq.real() = temp1;
	img_freq.imag() = temp2;
	fftw_destroy_plan(plan);
	delete [] mat1;
	delete [] mat2;
}



/**
  
  Put 1-d data into spatial domain from frequency domain

  @author   Vishnu Naresh Boddetti & Jonathon M. Smereka
  @version  04-03-2013

  @param    img			the 1-d signal in spatial domain
  @param    img_freq	the 1-d signal in frequency domain
  
  @return   passed by ref to return data in img

*/
void ifft2_signal_scalar(VectorXd &img, VectorXcd &img_freq) {
	int siz1 = img_freq.size();

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1);

	for (int i=0;i<siz1;i++){
		mat1[i][0] = img_freq(i).real();
		mat1[i][1] = img_freq(i).imag();
	}

	plan = fftw_plan_dft_1d(siz1,mat1,mat2,FFTW_BACKWARD, FFTW_ESTIMATE);
	fftw_execute(plan);

	for (int i=0;i<siz1;i++){
		img(i) = mat2[i][0]/(siz1);
	}

	fftw_destroy_plan(plan);
	delete [] mat1;
	delete [] mat2;
}