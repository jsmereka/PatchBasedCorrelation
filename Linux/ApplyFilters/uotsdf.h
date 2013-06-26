#ifndef UOTSDF_H_
#define UOTSDF_H_


#include "cfs.h"

/**

  Unconstrained Optimal Trade-off Synthetic Discriminant Function (UOTSDF) correlation filter
  Ref: A. Mahalanobis, B. VijayaKumar, S. Song, S. Sims, and J. Epperson, "Unconstrained correlation filters," Appl. Opt.  33, 3751-3759 (1994).

  h = T^(-1) * mean(X)
  T = alpha * P + beta * Dy + gamma * Sx
	or
  T = lambda * Dy + (1 - lambda) * P

  X = column matrix of the true class training samples
  Y = column matrix of the false class training samples
  P = output noise variance (ONV) = minimizing the effect of additive noise
  Dy = average correlation energy (ACE) of the false class training images = minimizing the energy in the resulting correlation plane (creating sharp peaks)
  Sx = average similarity measure (ASM) of the true class training images = minimizing variance in peak height

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  06-20-2013

 */
template <class T>
class UOTSDF : public filter<T> {
private:
	using filter<T>::zeropadtrnimgs;
	using filter<T>::asm_onlytrueclass;
	using filter<T>::whitenimgs;
	using filter<T>::input_row;
	using filter<T>::input_col;
	using filter<T>::X_hat;
	using filter<T>::H;

public:
	// constructors
	UOTSDF(double alph, double bet, double gam) {	// initializes all three parameters
		alpha = alph; beta = bet; gamma = gam;
	}
	UOTSDF(double lam) {								// initializes filter as a trade-off between ACE and ONV, ignoring ASM
		alpha = 1 - lam; beta = lam; gamma = 0;
	}
	UOTSDF() {										// initializes the parameters to their defaults
		alpha = pow(10,-5); beta = 1 - pow(10,-5); gamma = 0;
	}
	// destructor
	~UOTSDF() { }

	double alpha, beta, gamma;						// training parameters

	void trainfilter();								// train the filter, putting data into base class variable H

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



/**

  Trains the filter as an UOTSDF

  h = T^(-1) * mean(X)
  T = alpha * P + beta * Dy + gamma * Sx
	or
  T = lambda * Dy + (1 - lambda) * P

  h = derived filter, complex with dimension (d x 1), d = input_row * input_col, for 2D signals, it will be reshaped to (input_row x input_col)
  X = column matrix of input true class data, complex matrix with dimension (d x Nx), Nx = auth_count
  Y = column matrix of input false class data, complex matrix with dimension (d x Ny), Ny = imp_count
  Dy = diagonal matrix containing average power spectral density of the false class training data, complex matrix with dimension (d x d), diagonal matrix Di = Yi * Conj(Yi) = power spectrum of yi, D = 1/(Ny*d) * SUM{ Di }
  Sx = diagonal matrix containing similarity in correlation planes to the true class mean, complex matrix with dimension (d x d), S = 1/(Nx*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
  P = diagonal matrix containing the power spectral density of the input noise, complex matrix with dimension (d x d), P = constant * identity matrix for additive white noise

  @author   Jonathon M. Smereka
  @version  06-20-2013

  @return   nothing

 */
template <class T>
void UOTSDF<T>::trainfilter() {
	int acount = X_hat.cols();
	if(acount > 0) {
		int d = input_row * input_col;
		int rowmult = 1, colmult = 1;
		if(zeropadtrnimgs) {
			if(input_row > 1) {
				rowmult = 2;
			}
			if(input_col > 1) {
				colmult = 2;
			}
			d = d * rowmult * colmult;
		}

		/* Compute T matrix, labeled as TT cause T is the template type */
		// build filter
		{
			int tmpwhiten = whitenimgs;
			if(beta != 0.0) {
				whitenimgs = 3; // use Dy instead of Dxy
			}

			bool tmpasm = asm_onlytrueclass; // use Sx instead of Sxy
			typename filter<T>::VecCplx TT = filter<T>::tradeoff_scalar(alpha, beta, gamma);
			whitenimgs = tmpwhiten;
			asm_onlytrueclass = tmpasm;

			typename filter<T>::VecCplx tmp(d);
			typename filter<T>::MatCplx AllSamples(d,acount);

			AllSamples << X_hat;

			if(whitenimgs > 0) { // whiten
				for(int i=0; i<acount; i++) {
					AllSamples.col(i) = AllSamples.col(i).cwiseQuotient(TT);
				}
			}

			tmp.real().setZero(); tmp.imag().setZero();

			// h = T^(-1) * mean(X)
			tmp.noalias() = TT.cwiseProduct(AllSamples.rowwise().mean());

			TT.resize(0);
			typename filter<T>::MatCplx H_hat;
			H_hat = Eigen::Map<typename filter<T>::MatCplx>(tmp.data(), input_row*rowmult, input_col*colmult);
			tmp.resize(0); H.resize(input_row*rowmult,input_col*colmult);
			filter<T>::ifft_scalar(H, H_hat);
		}

		filter<T>::cleanclass();
	}
}




#endif /* UOTSDF_H_ */
