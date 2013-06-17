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

  Base filter class for 1-d and 2-d data samples

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-13-2013

 */
template <class T>									// designed for Matrix or Vector classes (non-complex numeric types)
class filter {
public:
	typedef typename T::RealScalar TVar;
	typedef std::complex<TVar> TCplx;
	typedef Eigen::Matrix<TCplx, Eigen::Dynamic, Eigen::Dynamic> MatCplx;
	typedef Eigen::Matrix<TCplx, 4, 4> MatCplx4;
	typedef Eigen::Matrix<TCplx, 3, 3> MatCplx3;
	typedef Eigen::Matrix<TCplx, 2, 2> MatCplx2;
	typedef Eigen::Matrix<TVar, Eigen::Dynamic, Eigen::Dynamic> TMat;
	typedef Eigen::Matrix<TCplx, Eigen::Dynamic, 1> VecCplx;
	typedef Eigen::Matrix<TCplx, 1, Eigen::Dynamic> RowVecCplx;
	typedef Eigen::Matrix<TVar, Eigen::Dynamic, 1> TVec;


private:
	bool check_rank(T signal, bool auth);			// check the matrix rank against the authentic or imposter samples
	int get_rank(TMat const& signal);				// get the matrix rank and return it as an int
	void addtoX(T const& signal);					// adds the data sample to X
	void addtoU(bool auth);							// add a 1 or 0 to vector U based on authentic or impostor sample

protected:
	TMat X;											// matrix of samples (authentic and impostor) in the spatial domain
	MatCplx X_hat;									// matrix of samples (authentic and impostor) in the frequency domain
	VecCplx U;											// vector of zeros and ones designating authentic and impostors in X

	VecCplx H_hat;									// the filter itself in the frequency domain

	int input_row, input_col;						// size of first sample added (new inputs will be resized - padded or cropped - to this)
	int auth_count, imp_count;						// authentic and impostor counts
	bool docomputerank;								// can turn on or off rank computation before adding samples (faster if off)
	bool cutfromcenter;								// zero-pad or crop from the center if true, otherwise top left corner
	bool cleanupaftertrain;							// if the filter will not be retrained at any point, clean up X, X_hat, and U after training

	void zero_pad(T &signal, const int siz1, const int siz2, bool center);// resizes sample window (zero pad/crop) based on input size

	void fft_scalar(T const& sig, MatCplx &sig_freq, int siz1, int siz2); // put sample into frequency domain
	void ifft_scalar(T &sig, MatCplx const& sig_freq);					  // get sample from frequency domain

	void cleanclass(void) {
		if(cleanupaftertrain) {
			X.resize(0,0); X_hat.resize(0,0); U.resize(0);
			auth_count = 0; imp_count = 0; input_row = 0; input_col = 0;
		}
	}

public:
	filter(){
		// initialize counts
		auth_count = 0; imp_count = 0;
		input_row = 0; input_col = 0;
		docomputerank = true; cutfromcenter = false;
		cleanupaftertrain = true;
	}
	virtual ~filter() {
		// resize to zero to release memory
		H_hat.resize(0);
		cleanupaftertrain = true; cleanclass();
	}



	bool add_auth(T const& newsig);						// add an authentic class sample for training
	bool add_imp(T const& newsig);						// add an impostor class sample for training

	inline int getauthcount(void) { return auth_count; }	// # of authentics added
	inline int getimpcount(void) { return imp_count; }		// # of impostors added
	inline void computerank(bool tmp) { docomputerank = tmp; } // set whether to compute matrix rank (check if similar sample has been added already)
	inline void adjustfromcenter(bool tmp) { cutfromcenter = tmp; } // set whether to crop/pad from the center or top left corner of the sample

	inline void noretraining(bool tmp) { cleanupaftertrain = tmp; } // no retraining is going to be used, clean up unnecessary memory

	virtual void trainfilter() = 0;						// train the filter - this varies with each filter type

	virtual T applyfilter(T const& scene) = 0;			// apply the filter to a scene, this may vary per filter design

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
	using filter<T>::X_hat;
	using filter<T>::H_hat;
	using filter<T>::U;

	typename filter<T>::VecCplx TT;					// diagonal matrix (in our case a vector) containing ASM, ONV, and ACE components

public:
	// constructors
	OTSDF(double alph, double bet, double gam) {	// initializes all three parameters
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
		TT.resize(0);
	}

	double alpha, beta, gamma;						// training parameters

	void trainfilter();								// train the filter, putting data into base class variable H

	T applyfilter(T const& scene);					// apply the filter to a scene, returning the resulting similarity plane

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



/**

  Trains the filter as an OTSDF

  h = T^(-1) * X * (X^(+) * T^(-1) * X)^(-1) * U
  T = alpha * P + beta * D + gamma * S
	or
  T = lambda * D + (1 - lambda) * P

  h = derived filter, complex with dimension (d x 1), d = input_row * input_col, for 2D signals, it will be reshaped to (input_row x input_col)
  X = column matrix of input data, complex matrix with dimension (d x N), N = auth_count + imp_count
  D = diagonal matrix containing average power spectral density of the training data, complex matrix with dimension (d x d), diagonal matrix Di = Xi * Conj(Xi) = power spectrum of xi, D = 1/d * SUM{ Di }
  S = diagonal matrix containing similarity in correlation planes to the true class mean, complex matrix with dimension (d x d), S = 1/(N*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
  P = diagonal matrix containing the power spectral density of the input noise, complex matrix with dimension (d x d), P = constant * identity matrix for additive white noise

  @author   Jonathon M. Smereka
  @version  06-14-2013

  @return   nothing

 */
template <class T>
void OTSDF<T>::trainfilter() {
	int N = auth_count + imp_count;
	if(N > 0) {
		int d = input_row * input_col;

		/* Compute T matrix, labeled as TT cause T is the template type */

		TT.resize(d); TT.real().setZero(); TT.imag().setZero();

		if(alpha != 0.0 || beta != 0.0 || gamma != 0.0) {
			TT.real().setOnes(); // if no alpha, beta, or gamma, then filter = MVSDF under white noise = ECPSDF
		} else {
			typename filter<T>::VecCplx tmp(d);
			typename filter<T>::RowVecCplx vec_of_ones(d); vec_of_ones.real().setOnes(); vec_of_ones.imag().setOnes();

			// ACE
			if(beta != 0) { // if only beta, then filter = MACE
				// diagonal matrix Di = Xi * Conj(Xi) = power spectrum of xi, D = 1/d * SUM{ Di }
				TT.noalias() = X_hat.cwiseProduct(X_hat.conjugate()).lazyProduct(vec_of_ones);
				TT.real() = TT.real() * beta;
				TT.imag() = TT.imag() * beta;
			}
			// ASM
			if(gamma != 0) { // if only gamma, then filter = constrained MACH
				// diagonal matrix S = 1/(N*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
				tmp.noalias() = (X_hat - X_hat.colwise().mean()).cwiseProduct((X_hat - X_hat.colwise().mean()).conjugate()).lazyProduct(vec_of_ones);
				tmp.real() = tmp.real() * gamma;
				tmp.imag() = tmp.imag() * gamma;
				TT = TT + tmp;
			}
			// ONV
			if(alpha != 0) { // if only alpha, then filter = MVSDF
				// diagonal matrix P = constant * identity for additive white noise
				tmp.real().setOnes(); tmp.imag().setZero();
				tmp.real() = tmp.real() * alpha;
				TT = TT + tmp;
			}
		}

		TT.cwiseInverse(); TT.cwiseSqrt(); // T^(-1/2)


		/* Compute h as ECPSDF filter (X * (X^(+) *  X)^(-1) * U) but use T in application (saves some computation) */

		H_hat.resize(d);  H_hat.real().setZero(); H_hat.imag().setZero();

		// if number of input data samples is less than 5, then inverting (X^(+) *  X) can be done very quickly
		// note that (X^(+) * X) is positive semi-definite
		switch(N) {
		case 1:
			{
				typename filter<T>::VecCplx XX_inv; XX_inv.noalias() = (X_hat.adjoint() * X_hat);
				H_hat.noalias() = X_hat * XX_inv.cwiseInverse() * U;
			}
			break;
		case 2:
			{
				typename filter<T>::MatCplx2 XX_inv; XX_inv.noalias() = (X_hat.adjoint() * X_hat);
				H_hat.noalias() = X_hat * XX_inv.inverse() * U;
			}
			break;
		case 3:
			{
				typename filter<T>::MatCplx3 XX_inv; XX_inv.noalias() = (X_hat.adjoint() * X_hat);
				H_hat.noalias() = X_hat * XX_inv.inverse() * U;
			}
			break;
		case 4:
			{
				typename filter<T>::MatCplx4 XX_inv; XX_inv.noalias() = (X_hat.adjoint() * X_hat);
				H_hat.noalias() = X_hat * XX_inv.inverse() * U;
			}
			break;
		default:
			{
				typename filter<T>::MatCplx XX_inv; XX_inv.noalias() = (X_hat.adjoint() * X_hat);
				// TODO: invert matrix greater than 4x4
			}
			break;
		}
		filter<T>::cleanclass();
	}
}



/**

  Applys the trained filter to the scene

  similarity plane = h^(+) * (T^(-1/2) * scene)

  @author   Jonathon M. Smereka
  @version  06-17-2013

  @return   resulting similary plane

 */
template <class T>
T OTSDF<T>::applyfilter(T const& scene) {
	T simplane;
	// TODO: apply filter

	return simplane;
}



/**

  Adds an authentic data sample to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    newsig		the 1-d/2-d sample that is to be added

  @return   true or false if successfully added

 */
template <class T>
bool filter<T>::add_auth(T const& newsig) {
	return check_rank(newsig, 1);
}



/**

  Adds an imposter data sample to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    newsig		the 1-d/2-d sample that is to be added

  @return   true or false if successfully added

 */
template <class T>
bool filter<T>::add_imp(T const& newsig) {
	return check_rank(newsig, 0);
}



/**

  Check the matrix rank against the other authentic or imposter data samples

  @author   Jonathon M. Smereka
  @version  04-13-2013

  @param    signal		the 1-d/2-d sample that is to be added
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
			if(docomputerank){
				X.resize(sz,1);
				// vectorize
				TVar *arrayd = (signal.template data());
				for(int j=0; j<sz; j++) {
					X(j,0) = arrayd[j];
				}
			}
			addtoX(signal);
			addtoU(auth);
			return true;
		}
	} else {
		// make sure it's the same size as the other samples
		if(X.rows() != sz) {
			zero_pad(signal, input_row, input_col, cutfromcenter);
			N = signal.rows();
			M = signal.cols();
			sz = N*M;
		}

		if(docomputerank) {
			// put sample into matrix
			TMat combsignal = X;
			combsignal.conservativeResize(sz,X.cols()+1);

			// vectorize
			TVar *arrayd = (signal.template data());
			//double *arrayd;
			//Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(arrayd,N,M) = signal.template cast<double>();

			for(int j=0; j<sz; j++) {
				combsignal(j,X.cols()) = arrayd[j];
			}

			// find rank
			rank = get_rank(combsignal);
			if(rank > (auth_count+imp_count)) {
				X.resize(sz,X.cols()+1);
				X = combsignal;
			}
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

  Add vectorized sample in fourier domain to X_hat matrix

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    signal		the 1-d/2-d sample that is to be added

  @return   nothing

 */
template <class T>
void filter<T>::addtoX(T const& signal) {
	int N = signal.rows();
	int M = signal.cols();
	int sz;

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

	TCplx *arrayd = (signal_hat.template data());

	for(int i=0; i<sz; i++) {
		//X_hat(i,X_hat.cols()-1) = std::complex<double>(arrayd1[i],arrayd2[i]);
		X_hat(i,X_hat.cols()-1) = arrayd[i];
	}
}



/**

  Add a 1 or 0 to vector U based on authentic or impostor sample

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
			U(0).real() = 1; U(0).imag() = 0;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1).real() = 1;
			U(auth_count+imp_count-1).imag() = 0;
		}
	} else {
		imp_count++;
		if(U.rows()*U.cols() == 0) {
			U.resize(1);
			U(0).real() = 0; U(0).imag() = 0;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1).real() = 0;
			U(auth_count+imp_count-1).imag() = 0;
		}
	}
}



/**

  Put 1-d/2-d data sample into the frequency domain

  @author   Vishnu Naresh Boddetti & Jonathon M. Smereka
  @version  04-10-2013

  @param    sig			the 1-d/2-d sample in spatial domain
  @param    sig_freq	the 1-d/2-d sample in frequency domain
  @param    siz1		rows
  @param    siz2		cols

  @return   passed by ref to return data in sig_freq

 */
template <class T>
void filter<T>::fft_scalar(T const& sig, MatCplx &sig_freq, int siz1, int siz2) {
	int N = sig.rows();
	int M = sig.cols();

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*siz1*siz2);

	const TVar *arrayd = sig.template data();

	for(int i=0; i<N*M; i++) {
		mat1[i][0] = arrayd[i];
		mat1[i][1] = 0;
	}

	if(M == 1 || N == 1) {
		plan = fftw_plan_dft_1d(siz1*siz2,mat1,mat2,FFTW_FORWARD,FFTW_ESTIMATE);
	} else {
		plan = fftw_plan_dft_2d(siz1,siz2,mat1,mat2,FFTW_FORWARD,FFTW_ESTIMATE);
	}

	fftw_execute(plan);

	sig_freq.resize(siz1,siz2);

	TCplx *arrayc = sig_freq.template data();

	for(int i=0; i<siz1*siz2; i++) {
		arrayc[i] = std::complex<TVar>(mat2[i][0],mat2[i][1]);
	}

	fftw_destroy_plan(plan);
	fftw_free(mat1);
	fftw_free(mat2);
}



/**

  Put 1-d/2-d data sample into spatial domain from frequency domain

  @author   Vishnu Naresh Boddetti & Jonathon M. Smereka
  @version  04-10-2013

  @param    sig			the 1-d/2-d sample in spatial domain
  @param    sig_freq	the 1-d/2-d sample in frequency domain

  @return   passed by ref to return data in sig

 */
template <class T>
void filter<T>::ifft_scalar(T &sig, MatCplx const& sig_freq) {
	int siz1 = sig_freq.rows();
	int siz2 = sig_freq.cols();
	int sz = siz1*siz2;

	fftw_plan plan;

	fftw_complex *mat1;
	fftw_complex *mat2;

	mat1 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*sz);
	mat2 = (fftw_complex*) fftw_malloc(sizeof(fftw_complex)*sz);

	TMat sig_real = sig_freq.real();
	TMat sig_imag = sig_freq.imag();

	const TVar* arrayr = sig_real.template data();
	const TVar* arrayi = sig_imag.template data();

	for(int i=0; i<sz; i++) {
		mat1[i][0] = arrayr[i];
		mat1[i][1] = arrayi[i];
	}

	if(siz2 == 1 || siz1 == 1) {
		plan = fftw_plan_dft_1d(sz,mat1,mat2,FFTW_BACKWARD,FFTW_ESTIMATE);
	} else {
		plan = fftw_plan_dft_2d(siz1,siz2,mat1,mat2,FFTW_BACKWARD,FFTW_ESTIMATE);
	}

	fftw_execute(plan);

	TVar* arrayd = sig.template data();

	for(int i=0; i<sz; i++) {
		arrayd[i] = mat2[i][0]/(siz1*siz2);
	}

	fftw_destroy_plan(plan);
	fftw_free(mat1);
	fftw_free(mat2);
}



/**

  Get the matrix rank

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    signal		the 2-d sample to be checked

  @return   matrix rank

 */
template <class T>
int filter<T>::get_rank(TMat const& signal) {
	int rank = 0;
	Eigen::JacobiSVD<TMat> svd(signal,0);
	TVec vals = svd.singularValues();
	for(int i = 0; i<vals.rows()*vals.cols(); i++) {
		if((int)(vals(i)) != 0) {
			rank++; // count non-zero values
		}
	}
	return rank;
}



/**

  Zero pad or crop the sample based on the input size parameters, simply using conserative resize won't work (uninitialized matrix elements)

  @author   Jonathon M. Smereka
  @version  04-13-2013

  @param    signal		the 1-d/2-d sample to be checked
  @param	siz1		final number of rows for the output
  @param	siz2		final number of columns for the output
  @param	center		pad/cut so the sample is centered

  @return   passed by ref to return sample in selected size window

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
