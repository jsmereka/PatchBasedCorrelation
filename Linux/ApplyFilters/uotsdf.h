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
	using filter<T>::whitenimgs;
	using filter<T>::cleanupaftertrain;
	using filter<T>::input_row;
	using filter<T>::input_col;
	using filter<T>::auth_count;
	using filter<T>::imp_count;
	using filter<T>::X_hat;
	using filter<T>::H;

public:
	// constructors
	UOTSDF(double alph, double bet, double gam) {	// initializes all three parameters
		alpha = alph; beta = bet; gamma = gam; trainedflag = false;
	}
	UOTSDF(double lam) {								// initializes filter as a trade-off between ACE and ONV, ignoring ASM
		alpha = 1 - lam; beta = lam; gamma = 0; trainedflag = false;
	}
	UOTSDF() {										// initializes the parameters to their defaults
		alpha = pow(10,-5); beta = 1 - pow(10,-5); gamma = 0; trainedflag = false;
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
	int N = auth_count + imp_count;
	if(N > 0) {
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
		typename filter<T>::MatCplx X_hat2; // only have to use this if whitenimgs && !cleanupaftertrain

		// build filter
		{
			typename filter<T>::VecCplx TT(d);
			typename filter<T>::VecCplx tmp(d);

			TT.imag().setZero();

			if(alpha == 0.0 && beta == 0.0 && gamma == 0.0 && !whitenimgs) {
				TT.real().setOnes(); // if no alpha, beta, or gamma, then filter =
			} else {
				typename filter<T>::VecCplx vec_of_ones(N);
				vec_of_ones.real().setOnes(); vec_of_ones.imag().setZero();
				// ASM
				if(gamma != 0.0) { // if only gamma, then filter =
					// TODO: fix for only true class data
					// diagonal matrix S = 1/(N*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
					TT.noalias() = (X_hat.colwise() - X_hat.rowwise().mean()).cwiseProduct((X_hat.colwise() - X_hat.rowwise().mean()).conjugate()).lazyProduct(vec_of_ones);
					TT.real() = TT.real() * (typename filter<T>::TVar)(gamma/(N*d));
					TT.imag() = TT.imag() * (typename filter<T>::TVar)(gamma/(N*d));
				} else {
					TT.real().setZero();
				}
				// ONV
				if(alpha != 0.0) { // if only alpha, then filter =
					// diagonal matrix P = constant * identity for additive white noise
					tmp.real().setOnes(); tmp.imag().setZero();
					tmp.real() = tmp.real() * (typename filter<T>::TVar)(alpha);
					TT = TT + tmp;
				}
				// ACE
				if(beta != 0.0 || whitenimgs) { // if only beta, then filter =
					// diagonal matrix Di = Xi * Conj(Xi) = power spectrum of xi, D = 1/(N*d) * SUM{ Di }
					tmp.noalias() = X_hat.cwiseProduct(X_hat.conjugate()).lazyProduct(vec_of_ones);
					tmp.real() = tmp.real() / (typename filter<T>::TVar)(N*d);
					tmp.imag() = tmp.imag() / (typename filter<T>::TVar)(N*d);
					if(whitenimgs) {
						if(!cleanupaftertrain) {
							X_hat2 = X_hat; // make copy to preserve for after training
						}
						for(int i=0; i<N; i++) {
							X_hat.col(i) = X_hat.col(i).cwiseQuotient(tmp);
						}
					}
					if(beta != 0.0) {
						tmp.real() = tmp.real() * (typename filter<T>::TVar)(beta);
						tmp.imag() = tmp.imag() * (typename filter<T>::TVar)(beta);
						TT = TT + tmp;
					}
				}
			}

			TT = TT.cwiseInverse(); // T^(-1)

			// H_hat = TT * mean(X)

			typename filter<T>::MatCplx H_hat;
			H_hat = Eigen::Map<typename filter<T>::MatCplx>(tmp.data(), input_row*rowmult, input_col*colmult);
			H.resize(input_row*rowmult,input_col*colmult);
			filter<T>::ifft_scalar(H, H_hat);
		}

		filter<T>::cleanclass();
		if(whitenimgs && !cleanupaftertrain) {
			X_hat = X_hat2;
			X_hat2.resize(0,0);
		}
	}
}




#endif /* UOTSDF_H_ */
