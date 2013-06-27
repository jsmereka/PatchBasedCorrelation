#ifndef CFS_H
#define CFS_H

#include <iostream>
#include <complex>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <algorithm>
#include <math.h>
#include <fftw3.h>


// Define data structure for images in the correlation filter library.
class CDataStruct{
public:
	float *data;
	std::complex<float> *data_freq;
	int num_data;
	int num_channels;
	std::vector<float*> ptr_data;
	std::vector<std::complex <float> *> ptr_data_freq;
	Eigen::VectorXi siz_data;
	Eigen::VectorXi siz_data_freq;
	float *label;
	int num_elements_freq;

	CDataStruct() { }

	~CDataStruct(){
		delete[] data;
		delete[] data_freq;
	}
};

/**

  Base filter class for 1-d and 2-d data samples

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-13-2013

  TODO: option to normalize spectrum + training adjustments if spectrum is normalized (ACE = Identity)
  TODO: add in PSR and PCE metrics
  TODO: fftw threading for large images (during scene comparison and training)
  TODO: include and support ASEF/E-ASEF, DCCF/PDCCF, MACH/E-MACH/EE-MACH/OTMACH/UMACE, MOSSE/E-MOSSE, MACE/MACE-MRH/MINACE/MICE/QMACE, MMCF/QMMCF, QCF, OTCHF
  TODO: adjust framework for ZACF (note: not programming ZACF, just adjusting the base abstract class to handle it accordingly)
  TODO: vector implementations for filters
  TODO: 3D filters (probably going to do this last cause ActionMach is essentially optical flow, also 3D adds a new level of memory management)

  TODO: fix Naresh's 'fusion_matrix_multiply' and 'fusion_matrix_inverse' functions (maybe they'll be faster than eigen's inverse since we know the structure of the matrix we are inverting, but part of me doubts it cause i'm not that good yet)

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
	bool check_rank_scalar(T signal, bool auth);	// check the matrix rank against the authentic or imposter samples
	int get_rank_scalar(TMat const& signal);		// get the matrix rank and return it as an int
	void addtofourier_scalar(T &signal, bool auth);	// adds the data sample to X_hat/Y_hat

protected:
	TMat XY;										// matrix of samples (authentic and impostor) in the spatial domain
	MatCplx X_hat, Y_hat;							// matrix of samples (authentic and impostor) in the frequency domain

	T H;											// the filter itself

	int input_row, input_col;						// size of first sample added (new inputs will be resized - padded or cropped - to this)
	int auth_count, imp_count;						// authentic and impostor counts

	bool docomputerank;								// can turn on or off rank computation before adding samples (faster if off)
	bool cutfromcenter;								// zero-pad or crop from the center if true, otherwise top left corner
	bool cleanupaftertrain;							// if the filter will not be retrained at any point, clean up X, X_hat, and U after training
	bool zeropadtrnimgs;							// zero pad the training samples to prevent the effects of circular correlation during training (much more memory intensive)
	bool asm_onlytrueclass;							// when calculating ASM, only use the true class images, otherwise use all images
	int whitenimgs;									// whiten the average spectrum of the data samples, 0 = off (default), 1 = all samples, 2 = only authentic samples, 3 = only impostor samples
	bool trainedflag;								// simple flag to see if there is a trained filter to be used

	void zero_pad_scalar(T &signal, const int siz1, const int siz2, bool center); // resizes sample window (zero pad/crop) based on input size
	void zero_pad_cplx_scalar(MatCplx &signal, const int siz1, const int siz2, bool center); // resizes sample window (zero pad/crop) based on input size
	void rebuild_cplxmat_scalar(MatCplx &signal, const int siz, bool center); // resize complex matrix by pulling each column out of fourier domain and putting back into fourier domain at new size

	void fft_scalar(T const& sig, MatCplx &sig_freq, int siz1, int siz2); // put sample into frequency domain
	void ifft_scalar(T &sig, MatCplx const& sig_freq);					  // get sample from frequency domain

	VecCplx tradeoff_scalar(double alpha, double beta, double gamma);	  // builds diagonal trade off matrix between ASM, ONV, and ACE
	void cleanclass(void);							// call after training to rotate the filter and dump unnecessary memory

public:
	filter(){
		// initialize counts
		auth_count = 0; imp_count = 0;
		input_row = 0; input_col = 0;
		docomputerank = true; cutfromcenter = false;
		cleanupaftertrain = true; zeropadtrnimgs = false;
		whitenimgs = 0; trainedflag = false; asm_onlytrueclass = true;
	}
	virtual ~filter() {
		// resize to zero to release memory
		H.resize(0,0); XY.resize(0,0);
		X_hat.resize(0,0); Y_hat.resize(0,0);
	}

	bool add_auth(T const& newsig);						// add an authentic class sample for training
	bool add_imp(T const& newsig);						// add an impostor class sample for training

	inline int getauthcount(void) { return auth_count; }	// # of authentics added
	inline int getimpcount(void) { return imp_count; }		// # of impostors added

	inline void computerank(bool tmp) { docomputerank = tmp; } // set whether to compute matrix rank (check if similar sample has been added already)
	inline void adjustfromcenter(bool tmp) { cutfromcenter = tmp; } // set whether to crop/pad from the center or top left corner of the sample
	inline void noretraining(bool tmp) { cleanupaftertrain = tmp; } // no retraining is going to be used, clean up unnecessary memory
	inline void zeropadtrndata(bool tmp) { zeropadtrnimgs = tmp; } // set whether to zeropad the training samples to prevent effects of circular correlation during training
	inline void whitenspectrum(int tmp) { whitenimgs = tmp; } // whiten the average spectrum of the training samples (results in sharper peaks when compared against training images, though is less robust to distortion)
	inline void setASM_trueclass(bool tmp) { asm_onlytrueclass = tmp; } // compute the ASM criterion with only the true class images

	virtual void trainfilter() = 0;						// train the filter - this varies with each filter type

	virtual T applyfilter(T scene);						// apply the filter to a scene, this may vary per filter design

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



/**

  Call after training to rotate the filter and dump unnecessary memory

  @author   Jonathon M. Smereka
  @version  06-20-2013

  @return   nothing

 */
template <class T>
void filter<T>::cleanclass(void) {
	trainedflag = true;
	if(zeropadtrnimgs) {
		filter<T>::zero_pad_scalar(H,input_row,input_col,false);
	}

	// rotate 180 degrees
	if(input_row > 1) {
		// swap row
		int j = input_row-1;
		for(int i=0; i<floor(input_row/2); i++) {
			H.row(i).swap(H.row(j)); j--;
		}
	}
	if(input_col > 1) {
		// swap columns
		int j = input_col-1;
		for(int i=0; i<floor(input_col/2); i++) {
			H.col(i).swap(H.col(j)); j--;
		}
	}
	// clean up unnecessary storage
	if(cleanupaftertrain) {
		XY.resize(0,0); X_hat.resize(0,0); Y_hat.resize(0,0);
		auth_count = 0; imp_count = 0;
	}
}




/**

  Applies the trained filter to the scene

  similarity plane = h .* scene

  @author   Jonathon M. Smereka
  @version  06-17-2013

  @return   resulting similary plane

 */
template <class T>
T filter<T>::applyfilter(T scene) {
	int N = scene.rows();
	int M = scene.cols();
	int fftszN = ((N + input_row-1) > 0) ? (N + input_row-1) : 1;
	int fftszM = ((M + input_col-1) > 0) ? (M + input_col-1) : 1;

	T simplane(fftszN,fftszM);
	if(trainedflag) {
		// put scene in freq domain and pad accordingly to prevent circular correlation
		typename filter<T>::MatCplx scene_freq(fftszN, fftszM), Filt_freq(fftszN, fftszM);
		T Filt = H;

		filter<T>::zero_pad_scalar(scene, fftszN, fftszM, cutfromcenter);
		filter<T>::fft_scalar(scene, scene_freq, fftszN, fftszM);
		filter<T>::zero_pad_scalar(Filt, fftszN, fftszM, cutfromcenter);
		filter<T>::fft_scalar(Filt, Filt_freq, fftszN, fftszM);

		scene_freq = scene_freq.cwiseProduct(Filt_freq);

		filter<T>::ifft_scalar(simplane, scene_freq);
		filter<T>::zero_pad_scalar(simplane, N, M, true);
	}
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
	return check_rank_scalar(newsig, true);
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
	return check_rank_scalar(newsig, false);
}



/**

  Check the matrix rank against the other authentic or imposter data samples

  @author   Jonathon M. Smereka
  @version  04-13-2013

  @param    signal		the 1-d/2-d sample that is to be added
  @param	auth		true = authentic, false = imposter

  @return   true or false if to be added to the vector

 */
template <class T>
bool filter<T>::check_rank_scalar(T signal, bool auth) {
	int Num = auth_count+imp_count;
	int rank = Num+1;
	int N = signal.rows();
	int M = signal.cols();
	int sz = N*M;

	if(auth_count == 0 && imp_count == 0) {
		if((M > 1 || N > 1) && docomputerank) {
			rank = get_rank_scalar(signal);
		}
		if(rank > 0) {
			input_row = N;
			input_col = M;
			if(docomputerank){
				XY.resize(sz,1);
				// vectorize
				TVar *arrayd = (signal.template data());
				for(int j=0; j<sz; j++) {
					XY(j,0) = arrayd[j];
				}
			}
			addtofourier_scalar(signal,auth);
			return true;
		}
	} else {
		// make sure it's the same size as the other samples
		if(input_row != sz) {
			zero_pad_scalar(signal, input_row, input_col, cutfromcenter);
			N = signal.rows();
			M = signal.cols();
			sz = N*M;
		}

		if(docomputerank) {
			// put sample into matrix
			TMat combsignal = XY;
			combsignal.conservativeResize(sz,Num+1);

			// vectorize
			TVar *arrayd = (signal.template data());
			//double *arrayd;
			//Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic> >(arrayd,N,M) = signal.template cast<double>();

			for(int j=0; j<sz; j++) {
				combsignal(j,Num) = arrayd[j];
			}

			// find rank
			rank = get_rank_scalar(combsignal);
			if(rank > Num) {
				XY.resize(sz,Num+1);
				XY = combsignal;
			}
		}
		if(rank > Num) {
			addtofourier_scalar(signal,auth);
			return true;
		}
	}
	return false;
}



/**

  Add vectorized sample in fourier domain to X_hat/Y_hat matrix

  @author   Jonathon M. Smereka
  @version  04-10-2013

  @param    signal		the 1-d/2-d sample that is to be added
  @param	auth		true = authentic, false = imposter

  @return   nothing

 */
template <class T>
void filter<T>::addtofourier_scalar(T &signal, bool auth) {
	int N = signal.rows();
	int M = signal.cols();
	int sz;
	if(zeropadtrnimgs) {
		if(N > 1) {
			N = N * 2;
		}
		if(M > 1) {
			M = M * 2;
		}
		zero_pad_scalar(signal, N, M, false);
	}
	// compute fft
	MatCplx signal_hat;
	fft_scalar(signal, signal_hat, N, M);

	N = signal_hat.rows();
	M = signal_hat.cols();
	sz = N*M;

	if(auth) {
		// put into matrix
		if(auth_count == 0) {
			X_hat.resize(sz,1);
		} else {
			// TODO: might replace zero_pad with rebuild_cplxmat...need to see if needed or not
			if(X_hat.rows() < sz) {
				// need to zero-pad X_hat
				zero_pad_cplx_scalar(X_hat, sz, X_hat.cols()+1, false);
				if(Y_hat.rows() < sz && imp_count > 0) {
					zero_pad_cplx_scalar(Y_hat, sz, Y_hat.cols(), false);
				}
			} else if(X_hat.rows() > sz) {
				// need to cut down X_hat
				rebuild_cplxmat_scalar(X_hat, sz, false);
				X_hat.conservativeResize(X_hat.rows(),auth_count+1);
				if(Y_hat.rows() < sz && imp_count > 0) {
					rebuild_cplxmat_scalar(Y_hat, sz, false);
				}
			} else {
				X_hat.conservativeResize(X_hat.rows(),auth_count+1);
			}
		}

		TCplx *arrayd = (signal_hat.template data());

		for(int i=0; i<sz; i++) {
			X_hat(i,auth_count) = arrayd[i];
		}
		auth_count++;
	} else {
		// put into matrix
		if(imp_count == 0) {
			Y_hat.resize(sz,1);
		} else {
			if(Y_hat.rows() < sz) {
				// need to zero-pad Y_hat
				zero_pad_cplx_scalar(Y_hat, sz, Y_hat.cols()+1, false);
				if(X_hat.rows() < sz && auth_count > 0) {
					zero_pad_cplx_scalar(X_hat, sz, X_hat.cols(), false);
				}
			} else if(Y_hat.rows() > sz) {
				// need to cut down Y_hat
				rebuild_cplxmat_scalar(Y_hat, sz, false);
				Y_hat.conservativeResize(Y_hat.rows(),imp_count+1);
				if(X_hat.rows() < sz && auth_count > 0) {
					rebuild_cplxmat_scalar(X_hat, sz, false);
				}
			} else {
				Y_hat.conservativeResize(Y_hat.rows(),imp_count+1);
			}
		}

		TCplx *arrayd = (signal_hat.template data());

		for(int i=0; i<sz; i++) {
			Y_hat(i,imp_count) = arrayd[i];
		}
		imp_count++;
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
		plan = fftw_plan_dft_2d(siz2,siz1,mat1,mat2,FFTW_FORWARD,FFTW_ESTIMATE);
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
		plan = fftw_plan_dft_2d(siz2,siz1,mat1,mat2,FFTW_BACKWARD,FFTW_ESTIMATE);
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
int filter<T>::get_rank_scalar(TMat const& signal) {
	int rank = 0;
	Eigen::JacobiSVD<TMat> svd(signal,0);
	TVec vals = svd.singularValues();
	for(int i = 0; i<vals.rows()*vals.cols(); i++) {
		if((vals(i) >= 0.05) || (vals(i) <= -0.05)) {
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
void filter<T>::zero_pad_scalar(T &signal, const int siz1, const int siz2, bool center) {
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



/**

  rebuild complex matrix to adjust for smaller or larger size (pull original out of fourier domain, and put back into fourier domain at adjusted size)

  @author   Jonathon M. Smereka
  @version  06-27-2013

  @param    signal		the 1-d/2-d sample to be checked
  @param	siz			final number of rows for the output
  @param	center		pad/cut so the sample is centered

  @return   passed by ref to return sample in selected size window

*/
template <class T>
void filter<T>::rebuild_cplxmat_scalar(MatCplx &signal, const int siz, bool center) {
	int M = signal.rows();
	int N = signal.cols();

	if(M != siz) {
		// pull each column out of fourier domain, adjust size, then put back into fourier domain
		int oldRow, oldCol, newRow, newCol;
		if(M < siz) {
			oldRow = input_row; newRow = input_row * 2;
			oldCol = input_col; newCol = input_col * 2;
		} else { // M > siz1;
			oldRow = input_row * 2; newRow = input_row;
			oldCol = input_col * 2; newCol = input_col;
		}

		MatCplx temp(siz,N), tempvec2(siz,1); // the final result
		MatCplx tempvec(M,1), sig1(oldRow,oldCol), sig2(newRow,newCol);
		T sig1_spat(oldRow,oldCol);

		for(int i=0; i<signal.rows(); i++) {
			tempvec = signal.block(0,i,M,1);
			sig1 = Eigen::Map<MatCplx>(tempvec.data(), oldRow, oldCol); // map to 2d/1d size
			ifft_scalar(sig1_spat, sig1); // put int spatial domain
			zero_pad_scalar(sig1_spat, newRow, newCol, center); // zero-pad or cut
			fft_scalar(sig1_spat, sig2, newRow, newCol);  // sig2 has the correct sized sample
			tempvec2 = Eigen::Map<MatCplx>(sig2.data(), siz, 1); // vectorize
			temp.block(0,i,siz,1) = tempvec2;
		}
		signal.resize(siz, N);
		signal = temp;
	}
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
void filter<T>::zero_pad_cplx_scalar(MatCplx &signal, const int siz1, const int siz2, bool center) {
	int M = signal.rows();
	int N = signal.cols();

	if(M != siz1 || N != siz2) {
		// copy data into manipulative matrix
		MatCplx temp(siz1,siz2); // temporary variable

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




/**

  Build the diagonal trade off matrix T (between ACE, ONV, and ASM)

  @author   Jonathon M. Smereka
  @version  06-25-2013

  @param    alpha		weight of ONV criterion
  @param	beta		weight of ACE criterion
  @param	gamma		weight of ASM criterion

  @return   complex vector T containing trade off between ACE, ONV, and ASM

 */
template <class T>
typename filter<T>::VecCplx filter<T>::tradeoff_scalar(double alpha, double beta, double gamma) {
	int N = auth_count + imp_count;
	VecCplx TT;
	if(N > 0) {
		int d = input_row * input_col;
		if(zeropadtrnimgs) {
			if(input_row > 1) {
				d *= 2;
			}
			if(input_col > 1) {
				d *= 2;
			}
		}

		/* Compute T matrix, labeled as TT cause T is the template type */
		VecCplx tmp(d);
		VecCplx vec_of_ones(N);
		vec_of_ones.real().setOnes(); vec_of_ones.imag().setZero();

		TT.resize(d);
		TT.imag().setZero();

		if(alpha == 0.0 && beta == 0.0 && gamma == 0.0) {
			TT.real().setOnes(); // if no alpha, beta, or gamma, then filter = MVSDF under white noise = ECPSDF
		}

		// ASM
		if(gamma != 0.0 && auth_count > 0) { // if only gamma, then filter = MSESDF (constrained MACH)
			if(asm_onlytrueclass) {
				// diagonal matrix S = 1/(Nx*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
				TT.noalias() = (X_hat.colwise() - X_hat.rowwise().mean()).cwiseProduct((X_hat.colwise() - X_hat.rowwise().mean()).conjugate()).lazyProduct(vec_of_ones);
				TT.real() = TT.real() * (TVar)(gamma/(N*d));
				TT.imag() = TT.imag() * (TVar)(gamma/(N*d));
			} else {
				// diagonal matrix S = 1/(N*d) * SUM{ (Zi - mean(Zi)) * Conj(Zi - mean(Zi)) }
				TT.noalias() = (X_hat.colwise() - X_hat.rowwise().mean()).cwiseProduct((X_hat.colwise() - X_hat.rowwise().mean()).conjugate()).lazyProduct(vec_of_ones);
				TT.real() = TT.real() * (TVar)(gamma/(N*d));
				TT.imag() = TT.imag() * (TVar)(gamma/(N*d));
			}
		} else {
			TT.real().setZero();
		}

		// ONV
		if(alpha != 0.0) { // if only alpha, then filter = MVSDF
			// diagonal matrix P = constant * identity for additive white noise
			tmp.real().setOnes(); tmp.imag().setZero();
			tmp.real() = tmp.real() * (TVar)(alpha);
			TT = TT + tmp;
		}

		// ACE
		if(beta != 0.0 || whitenimgs > 0) { // if only beta, then filter = MACE
			// diagonal matrix Di = Zi * Conj(Zi) = power spectrum of zi (where zi is an authentic or impostor data sample), D = 1/(N*d) * SUM{ Di }
			int resetwhiten = whitenimgs;
			if(resetwhiten == 1 || resetwhiten > 3) {
				if(auth_count == 0) {
					resetwhiten = 3;
				} else if(imp_count == 0) {
					resetwhiten = 2;
				}
			}
			switch(resetwhiten) {
			case 2: // use authentic
			{
				if(auth_count > 0) {
					tmp.noalias() = X_hat.cwiseProduct(X_hat.conjugate()).lazyProduct(vec_of_ones.block(0,0,auth_count,1));
					tmp.real() = tmp.real() * (TVar)(beta/N*d);
					tmp.imag() = tmp.imag() * (TVar)(beta/N*d);
					TT = TT + tmp;
				}
			}
			break;
			case 3: // use impostor
			{
				if(imp_count > 0) {
					tmp.noalias() = Y_hat.cwiseProduct(Y_hat.conjugate()).lazyProduct(vec_of_ones.block(0,0,imp_count,1));
					tmp.real() = tmp.real() * (TVar)(beta/N*d);
					tmp.imag() = tmp.imag() * (TVar)(beta/N*d);
					TT = TT + tmp;
				}
			}
			break;
			default: // use all
			{
				MatCplx AllSamples;
				if(auth_count > 0 && imp_count > 0) {
					AllSamples.resize(d,N); AllSamples << X_hat, Y_hat;
				} else if(auth_count > 0) {
					AllSamples = X_hat;
				} else if(imp_count > 0) {
					AllSamples = Y_hat;
				}
				tmp.noalias() = AllSamples.cwiseProduct(AllSamples.conjugate()).lazyProduct(vec_of_ones);
				tmp.real() = tmp.real() * (TVar)(beta/N*d);
				tmp.imag() = tmp.imag() * (TVar)(beta/N*d);
				TT = TT + tmp;
			}
			break;
			}
		}

		TT = TT.cwiseInverse(); // T^(-1)
	}
	return TT;
}


#endif
