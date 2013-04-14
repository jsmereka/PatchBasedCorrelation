#include "stdafx.h"
#include "CorrelationFilters.h"
#include <algorithm>


/*
	Constructor
*/
template <class T>
filter<T>::filter() {
	auth_count = 0; imp_count = 0;
	input_row = 0; input_col = 0;
}



/*
	Destructor
*/
template <class T>
filter<T>::~filter() {
	H.resize(0,0);
	X.resize(0,0);
	X_hat.resize(0,0);
	U.resize(0);
}



/**
  
  Adds an authentic signal to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    newsig		the 1-d/2-d signal that is to be added
  
  @return   true or false if successfully added

*/
template <class T>
bool filter<T>::add_auth(T &newsig) {
	return check_rank(newsig, 1);
}



/**
  
  Adds an imposter signal to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    newsig		the 1-d/2-d signal that is to be added
  
  @return   true or false if successfully added

*/
template <class T>
bool filter<T>::add_imp(T &newsig) {
	return check_rank(newsig, 0);
}



/**
  
  Check the matrix rank against the other authentic or imposter signals

  @author   Jonathon M. Smereka
  @version  04-13-2013

  @param    signal		the 1-d/2-d signal that is to be added
  @param	auth		1 = authentic, 0 = imposter
  
  @return   true or false if to be added to the vector

*/
template <class T>
bool filter<T>::check_rank(T const& signal, bool auth) {
	int rank = 100;
	if((auth && auth_count == 0) || (!auth && imp_count == 0)) {
		if(signal.cols() > 1) {
			rank = get_rank(signal);
		}
		if(rank > 0) {
			input_row = signal.rows();
			input_col = signal.cols();
			addtoX(signal);
			addtoU(auth);
			return true;
		}
	} else {
		// put signal into matrix
		MatrixXd combsignal = X;
		combsignal.conservativeResize(X.rows(),X.cols()+1);

		// make sure it's the same size as the other signals
		if(X.rows() != (signal.rows()*signal.cols())) {
			signal = zero_pad(signal, input_row, input_col, 1);
		}

		if(signal.cols() == 1) {
			combsignal.col(X.cols()) = signal.data(); // i wonder if this will work
		} else {
			// vectorize
			int ct = 0;
			for(int j=0; j<signal.cols(); j++) {
				for(int i=0; i<signal.rows(); i++) {
					combsignal(ct,X.cols()) = signal(i,j);
					ct++;
				}
			}
		}
		// find rank
		rank = get_rank(combsignal);
		if(rank > 0) {
			addtoX(signal);
			addtoU(auth);
			return true;
		}
	}
	return false;
}



/**
  
  Add vectorized signal to X matrix

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    signal		the 1-d/2-d signal that is to be added
  
  @return   nothing

*/
template <class T>
void filter<T>::addtoX(T const& signal) {
	// put into matrix
	if(X.cols() == 0) {
		X.resize(signal.rows()*signal.cols(),1);
	} else {
		X.conservativeResize(X.rows(),X.cols()+1);
	}
	int ct = 0;
	if(signal.cols() == 1) {
		X.col(X.cols()-1) = signal.data(); // i wonder if this will work
	} else {
		// vectorize
		for(int j=0; j<signal.cols(); j++) {
			for(int i=0; i<signal.rows(); i++) {
				X(ct,X.cols()-1) = signal(i,j);
				ct++;
			}
		}
	}
	// compute fft
	MatrixXcd signal_hat;
	fft_scalar(signal, signal_hat, signal.rows(), signal.cols()); // may need to add padding here eventually

	// put into matrix
	if(X_hat.cols() == 0) {
		X_hat.resize(signal_hat.rows()*signal_hat.cols(),1);
	} else {
		X_hat.conservativeResize(X_hat.rows(),X_hat.cols()+1);
	}
	MatrixXd temp1(signal_hat.rows(),signal_hat.cols()), temp2(signal_hat.rows(),signal_hat.cols());
	temp1 = signal_hat.real();
	temp2 = signal_hat.imag();
	ct = 0;
	for(int i=0; i<signal_hat.rows(); i++) {
		for(int j=0; j<signal_hat.cols(); j++) {
			X_hat(ct,X_hat.cols()-1) = std::complex<double>(temp1(i,j),temp2(i,j));
			ct++;
		}
	}
}



/**
  
  Add a 1 or 0 to vector U based on authentic or impostor signal

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    auth		whether to add a 1 or 0
  
  @return   nothing

*/
template <class T>
void filter<T>::addtoU(bool auth) {
	if(auth) {
		auth_count++;
		if(U.size() == 0) {
			U.resize(1);
			U(0) = 1;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1) = 1;
		}
	} else {
		imp_count++;
		if(U.size() == 0) {
			U.resize(1);
			U(0) = 0;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1) = 0;
		}
	}
}



/**
  
  Put 1-d/2-d data into the frequency domain

  @author   Vishnu Naresh Boddetti & Jonathon M. Smereka
  @version  04-10-2013

  @param    sig			the 1-d/2-d signal in spatial domain
  @param    sig_freq	the 1-d/2-d signal in frequency domain
  @param    siz1		rows
  @param    siz2		cols
  
  @return   passed by ref to return data in sig_freq

*/
template <class T>
void filter<T>::fft_scalar(T const& sig, MatrixXcd &sig_freq, int siz1, int siz2) {
	int count = 0;
	int N = sig.rows();
	int M = sig.cols();

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);

	VectorXd tmp = sig.data();

	for(int i=0; i<tmp.size(); i++) {
		mat1[i][0] = tmp(i);
		mat1[i][1] = 0;
	}

	if(M == 1) {
		plan = fftw_plan_dft_1d(siz1*siz2,mat1,mat2,FFTW_FORWARD,FFTW_ESTIMATE);
	} else {
		plan = fftw_plan_dft_2d(siz1,siz2,mat1,mat2,FFTW_FORWARD,FFTW_ESTIMATE);	
	}

	fftw_execute(plan);		
	MatrixXd temp1(siz1,siz2), temp2(siz1,siz2);

	count = 0;
	for(int i=0; i<siz1; i++) {
		for(int j=0; j<siz2; j++) {				
			temp1(i,j) = mat2[count][0];
			temp2(i,j) = mat2[count][1];
			count++;
		}
	}
	sig_freq.resizeLike(temp1);
	sig_freq.real() = temp1;
	sig_freq.imag() = temp2;
	fftw_destroy_plan(plan);
	delete [] mat1;
	delete [] mat2;
}



/**
  
  Put 1-d/2-d data into spatial domain from frequency domain

  @author   Vishnu Naresh Boddetti & Jonathon M. Smereka
  @version  04-10-2013

  @param    sig			the 1-d/2-d signal in spatial domain
  @param    sig_freq	the 1-d/2-d signal in frequency domain
  
  @return   passed by ref to return data in sig

*/
template <class T>
void filter<T>::ifft_scalar(T &sig, MatrixXcd const& sig_freq) {
	int count;
	int siz1 = sig_freq.rows();
	int siz2 = sig_freq.cols();

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

	if(siz2 == 1) {
		plan = fftw_plan_dft_1d(siz1*siz2,mat1,mat2,FFTW_BACKWARD,FFTW_ESTIMATE);
	} else {
		plan = fftw_plan_dft_2d(siz1,siz2,mat1,mat2,FFTW_BACKWARD,FFTW_ESTIMATE);
	}

	fftw_execute(plan);

	if(siz2 == 1) {
		VectorXd tmp;
		for(int i=0; i<siz1; i++) {
			tmp(i) = mat2[i][0]/(siz1);
		}
		sig = tmp.data();
	} else {
		count = 0;
		for(int i=0; i<siz1; i++) {
			for(int j=0; j<siz2; j++) {
				sig(i,j) = mat2[count][0]/(siz1*siz2);
				count++;
			}
		}
	}

	fftw_destroy_plan(plan);
	delete [] mat1;
	delete [] mat2;
}



/**
  
  Get the matrix rank

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    signal		the 2-d signal to be checked
  
  @return   matrix rank

*/
template <class T>
int filter<T>::get_rank(MatrixXd const& signal) {
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
  
  Zero pad or crop the signal based on the input size parameters, simply using conserative resize won't work (uninitialized matrix elements)

  @author   Jonathon M. Smereka
  @version  04-13-2013

  @param    signal		the 1-d/2-d signal to be checked
  @param	siz1		final number of rows for the output
  @param	siz2		final number of columns for the output
  @param	center		pad/cut so the signal is centered
  
  @return   signal in selected size window

*/
template <class T>
T filter<T>::zero_pad(T const& signal, int siz1, int siz2, bool center) {
	int M = signal.rows();
	int N = signal.cols();

	if(M == siz1 && N == siz2) {
		return signal;
	} else {
		T result;
		MatrixXd temp;
		int str, stc;

		if(M == 1 || N == 1) {
			MatrixXd newsignal(max(M,N),1);
			// put into matrix (1-d matrix is used differently than 1-d vector)
			newsignal.col(0) = signal.data(); // i wonder if this will work
		} else {
			MatrixXd newsignal = signal;
		}
		// Perform any cropping operations
		if(siz1 < M || siz2 < N) {
			if(center) { // cut from the center
				str = floor(abs(siz1 - M)/2);
				stc = floor(abs(siz2 - N)/2);
			} else {
				str = 0; stc = 0;
			}
			if(siz1 < M && siz2 < N) { // crop both rows and columns
				temp.resize(siz1,siz2);
				temp = newsignal.block<siz1,siz2>(str,stc);
				M = siz1; N = siz2;
			} else if(siz1 < M) {
				temp.resize(siz1,max(N,siz2));
				temp.setZero();
				M = siz1;
				temp = newsignal.block<siz1,N>(str,stc);
			} else if(siz2 < N) {
				temp.resize(max(M,siz1),siz2);
				temp.setZero();
				N = siz2;
				temp = newsignal.block<M,siz2>(str,stc);
			}
		} else {
			temp.setZero(siz1,siz2);
		}
		// Perform any padding operations
		if(siz1 > M || siz2 > N) {
			if(center) { // pad from the center
				str = floor((siz1 - M)/2);
				stc = floor((siz2 - N)/2);
			}
			if(siz1 > M && siz2 > N) { // pad both rows and columns
				temp.conservativeResize(siz1,siz2);
			} else if(siz1 > M) {
				temp.conservativeResize(siz1,N);
			} else if(siz2 > N) {
				temp.conservativeResize(M,siz2);
			}
			temp.block<M,N>(str,stc) = newsignal;
		}
		// convert temp back to T
		result.resize(siz1,siz2);
		if(siz1 == 1 || siz2 == 1) {
			result = temp.data(); // i wonder if this will work
		} else {
			result = temp;
		}
		return result;
	}
}
