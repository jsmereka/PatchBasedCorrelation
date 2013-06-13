#ifndef CFS_H
#define CFS_H

#include <iostream>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <math.h>
#include <fftw3.h>


/**

  Base filter class for 1-d and 2-d signals

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-13-2013

*/
template <class T>									// designed for Matrix or Vector classes (non-complex)
class filter {
public:
	typedef Eigen::Matrix<std::complex<typename T::RealScalar>, Eigen::Dynamic, Eigen::Dynamic> MatCplx;
	typedef Eigen::Matrix<typename T::RealScalar, Eigen::Dynamic, Eigen::Dynamic> TMat;
	typedef Eigen::Matrix<std::complex<typename T::RealScalar>, Eigen::Dynamic, 1> VecCplx;
	typedef Eigen::Matrix<typename T::RealScalar, Eigen::Dynamic, 1> TVec;
	typedef typename T::Scalar TVar;

private:
	bool check_rank(T signal, bool auth);			// check the matrix rank against the authentic or imposter signals
	int get_rank(TMat const& signal);				// get the matrix rank and return it as an int
	void addtoX(T const& signal);					// adds the signal to X
	void addtoU(bool auth);							// add a 1 or 0 to vector U based on authentic or impostor signal

protected:
	TMat X;											// matrix of signals (authentic and impostor) in the spatial domain
	MatCplx X_hat;									// matrix of signals (authentic and impostor) in the frequency domain
	TVec U;											// vector of zeros and ones designating authentic and impostors in X

	MatCplx H_hat;									// the filter itself in the frequency domain

	int input_row, input_col;						// size of first signal added (new inputs will be resized - padded or cropped - to this)
	int auth_count, imp_count;						// authentic and impostor counts
	bool docomputerank;								// can turn on or off rank computation before adding samples (faster if off)
	bool cutfromcenter;								// zero-pad or crop from the center if true, otherwise top left corner

	void zero_pad(T &signal, const int siz1, const int siz2, bool center);// resizes signal window (zero pad/crop) based on input size

	void fft_scalar(T const& sig, MatCplx &sig_freq, int siz1, int siz2); // put signal into frequency domain
	void ifft_scalar(T &sig, MatCplx const& sig_freq);					  // get signal from frequency domain

public:
	filter(){
		// initialize counts
		auth_count = 0; imp_count = 0;
		input_row = 0; input_col = 0;
		docomputerank = true; cutfromcenter = false;
	}
	virtual ~filter() {
		// resize to zero to release memory
		H_hat.resize(0,0);
		X.resize(0,0);
		X_hat.resize(0,0);
		U.resize(0);
	}


	bool add_auth(T const& newsig);						// add an authentic class example for training
	bool add_imp(T const& newsig);						// add an impostor class example for training

	inline int getauthcount(void) { return auth_count; }	// # of authentics added
	inline int getimpcount(void) { return imp_count; }		// # of impostors added
	inline void computerank(bool tmp) { docomputerank = tmp; } // set whether to compute matrix rank (check if similar signal has been added already)
	inline void adjustfromcenter(bool tmp) { cutfromcenter = tmp; } // set whether to crop/pad from the center or top left corner of the signal

	virtual void trainfilter() = 0;					// train the filter - this varies with each filter type

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};






/**

  Optimal Trade-off Synthetic Discriminant Function (OTSDF) correlation filter
  Ref: P. Refregier, "Filter design for optical pattern recognition: multicriteria optimization approach," Opt. Lett.  15, 854-856 (1990).

  h = T^(-1) * X * (X^(+) * T^(-1) * X)^(-1) * U
  T = alpha * P + beta * D + gamma * S
	or
  T = lambda * D + (1 - lambda) * P

  P = output noise variance (ONV) = minimizing the effect of additive noise
  D = average correlation energy (ACE) = minimizing the energy in the resulting correlation plane (creating sharp peaks)
  S = average similarity measure (ASM) = minimizing variance in peak height

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-13-2013

*/

template <class T>
class OTSDF : public filter<T> {
private:
	using filter<T>::input_row;
	using filter<T>::input_col;
	using filter<T>::auth_count;
	using filter<T>::imp_count;
	typename filter<T>::VecCplx P, D, S;						// Matrices for ONV, ACE, and ASM (dxd diagonal matrix where d = dimension, but set as vectors for efficient computation)

public:
	// constructors
	OTSDF(double alph, double bet, double gam) {				// initializes all three parameters
		alpha = alph; beta = bet; gamma = gam;
	}
	OTSDF(double lam) {								// initializes filter as a trade-off between ACE and ONV, ignoring ASM
		alpha = 1 - lam; beta = lam; gamma = 0;
	}
	OTSDF() {										// initializes the parameters to their defaults
		alpha = pow(10,-5); beta = 1 - alpha; gamma = 0;
	}
	// destructor
	~OTSDF() {
		P.resize(0,0); D.resize(0,0); S.resize(0,0);
	}

	double alpha, beta, gamma;						// training parameters

	void trainfilter();								// train the filter, putting data into base class variable H

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



/**

  Trains the filter as an OTSDF

  @author   Jonathon M. Smereka
  @version  04-13-2013

  @return   nothing

*/
// TODO implement filter training
template <class T>
void OTSDF<T>::trainfilter() {
	if(auth_count != 0 || imp_count != 0) {
		//int d = input_row * input_col;

		// ONV
		if(alpha != 0) {
			//
		}
		// ACE
		if(beta != 0) {
			//
		}
		// ASM
		if(gamma != 0) {
			//
		}



	}
}



/**

  Adds an authentic signal to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    newsig		the 1-d/2-d signal that is to be added

  @return   true or false if successfully added

*/
template <class T>
bool filter<T>::add_auth(T const& newsig) {
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
bool filter<T>::add_imp(T const& newsig) {
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
bool filter<T>::check_rank(T signal, bool auth) {
	int rank = auth_count+imp_count+1;
	int N = signal.rows();
	int M = signal.cols();
	int sz = N*M;
	if(auth_count == 0 && imp_count == 0) {
		if(signal.cols() > 1 && docomputerank) {
			rank = get_rank(signal);
		}
		if(rank > 0) {
			input_row = N;
			input_col = M;
			addtoX(signal);
			addtoU(auth);
			return true;
		}
	} else {
		// make sure it's the same size as the other signals
		if(X.rows() != sz) {
			zero_pad(signal, input_row, input_col, cutfromcenter);
			N = signal.rows();
			M = signal.cols();
			sz = N*M;
		}

		if(docomputerank) {
			// put signal into matrix
			TMat combsignal = X;
			combsignal.conservativeResize(X.rows(),X.cols()+1);

			// TODO: fix vectorization
			// vectorize

			double *arrayd = (signal.template data());

			for(int j=0; j<sz; j++) {
				combsignal(j,X.cols()) = arrayd[j];
			}

			// find rank
			rank = get_rank(combsignal);
		}
		if(rank > (auth_count+imp_count)) {
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
	int N = signal.rows();
	int M = signal.cols();
	int sz = N*M;

	// put into matrix
	if(X.cols() == 0) {
		X.resize(sz,1);
	} else {
		X.conservativeResize(X.rows(),X.cols()+1);
	}
	// TODO: fix vectorization
	// vectorize
	const double *arrayd = signal.template data();
	//double arrayd[N*M];
	//Eigen::Map<T>(arrayd, N, M) = signal.template cast<double>();

	for(int j=0; j<sz; j++) {
		X(j,X.cols()-1) = arrayd[j];
	}

	// compute fft
	MatCplx signal_hat;
	fft_scalar(signal, signal_hat, N, M); // may need to add padding here eventually

	N = signal_hat.rows();
	M = signal_hat.cols();
	sz = N*M;

	// put into matrix
	if(X_hat.cols() == 0) {
		X_hat.resize(sz,1);
	} else {
		X_hat.conservativeResize(X_hat.rows(),X_hat.cols()+1);
	}
	TMat temp1(N,M), temp2(N,M);
	temp1 = signal_hat.real();
	temp2 = signal_hat.imag();
	int ct = 0;
	for(int i=0; i<N; i++) {
		for(int j=0; j<M; j++) {
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
		if(U.rows()*U.cols() == 0) {
			U.resize(1);
			U(0) = 1;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1) = 1;
		}
	} else {
		imp_count++;
		if(U.rows()*U.cols() == 0) {
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
void filter<T>::fft_scalar(T const& sig, MatCplx &sig_freq, int siz1, int siz2) {


	int count = 0;
	int N = sig.rows();
	int M = sig.cols();

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);
	// TODO: fix vectorization
	const double *arrayd = sig.template data();
	//double arrayd[N*M];
	//Eigen::Map<T>(arrayd, N, M) = sig.template cast<double>();

	for(int i=0; i<sig.rows()*sig.cols(); i++) {
		mat1[i][0] = arrayd[i];
		mat1[i][1] = 0;
	}

	if(M == 1 || N == 1) {
		plan = fftw_plan_dft_1d(siz1*siz2,mat1,mat2,FFTW_FORWARD,FFTW_ESTIMATE);
	} else {
		plan = fftw_plan_dft_2d(siz1,siz2,mat1,mat2,FFTW_FORWARD,FFTW_ESTIMATE);
	}

	fftw_execute(plan);
	TMat temp1(siz1,siz2);
	TMat temp2(siz1,siz2);

	count = 0;
	for(int i=0; i<siz1; i++) {
		for(int j=0; j<siz2; j++) {
			temp1(i,j) = mat2[count][0];
			temp2(i,j) = mat2[count][1];
			count++;
		}
	}
	sig_freq.conservativeResize(siz1,siz2);
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
void filter<T>::ifft_scalar(T &sig, MatCplx const& sig_freq) {
	int count;
	int siz1 = sig_freq.rows();
	int siz2 = sig_freq.cols();

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);

	TMat sig_real = sig_freq.real();
	TMat sig_imag = sig_freq.imag();

	count = 0;
	for (int i=0;i<siz1;i++){
		for (int j=0;j<siz2;j++){
			mat1[count][0] = sig_real(i,j);
			mat1[count][1] = sig_imag(i,j);
			count++;
		}
	}

	if(siz2 == 1 || siz1 == 1) {
		plan = fftw_plan_dft_1d(siz1*siz2,mat1,mat2,FFTW_BACKWARD,FFTW_ESTIMATE);
	} else {
		plan = fftw_plan_dft_2d(siz1,siz2,mat1,mat2,FFTW_BACKWARD,FFTW_ESTIMATE);
	}

	fftw_execute(plan);

	double* arrayd = sig.template data();

	for(int i=0; i<siz1*siz2; i++) {
		arrayd[i] = mat2[i][0]/(siz1*siz2);
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
int filter<T>::get_rank(TMat const& signal) {
	int rank = 0;
	Eigen::JacobiSVD<TMat> svd(signal,0);
	Eigen::VectorXd vals = svd.singularValues();
	for(int i = 0; i<vals.rows()*vals.cols(); i++) {
		if((int)(vals(i)) != 0) {
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
void filter<T>::zero_pad(T &signal, const int siz1, const int siz2, bool center) {
	int M = signal.rows();
	int N = signal.cols();

	if(M != siz1 || N != siz2) {
		// copy data into manipulative matrix
		TMat temp(siz1,siz2); // temporary variable

		bool nocut = false;
		int str = 0, stc = 0; // starting row and column (center or top corner)

		// Perform any cropping operations
		if(siz1 < M || siz2 < N) {
			if(siz1 < M && siz2 < N) { // crop both rows and columns
				if(center) { // cut from the center
					str = floor(abs(siz1 - M)/2);
					stc = floor(abs(siz2 - N)/2);
				}
				temp = signal.block(str,stc,siz1,siz2);
				M = siz1; N = siz2;
			} else if(siz1 < M) {
				if(center) { // cut from the center
					str = floor(abs(siz1 - M)/2);
				}
				temp.setZero(siz1,std::max(N,siz2));
				M = siz1;
				temp.block(0,0,siz1,N) = signal.block(str,stc,siz1,N);
			} else if(siz2 < N) {
				if(center) { // cut from the center
					stc = floor(abs(siz2 - N)/2);
				}
				temp.setZero(std::max(M,siz1),siz2);
				N = siz2;
				temp.block(0,0,M,siz2) = signal.block(str,stc,M,siz2);
			}
		} else {
			temp.setZero(siz1,siz2); nocut = true;
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
			if(nocut) {
				temp.block(str,stc,M,N) = signal;
			}
		}
		// resize signal
		signal.resize(siz1,siz2);
		signal.setZero(siz1,siz2);
		signal = temp;
	}
}


#endif
