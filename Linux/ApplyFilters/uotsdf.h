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
	using filter<T>::X_hat;
	using filter<T>::Y_hat;
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
	int acount = X_hat.cols(), icount = Y_hat.cols();
	int N = acount + icount;
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
		// build filter
		{
			typename filter<T>::VecCplx TT(d);
			typename filter<T>::VecCplx tmp(d);
			typename filter<T>::MatCplx AllSamples;
			typename filter<T>::VecCplx U(N);
			U.real().setOnes(); U.imag().setZero();

			TT.imag().setZero();

			if(alpha == 0.0 && beta == 0.0 && gamma == 0.0) {
				TT.real().setOnes(); // if no alpha, beta, or gamma, then filter =
			}

			// ASM
			if(gamma != 0.0 && acount > 0) { // if only gamma, then filter = MACH
				// diagonal matrix Sx = 1/(N*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
				TT.noalias() = (X_hat.colwise() - X_hat.rowwise().mean()).cwiseProduct((X_hat.colwise() - X_hat.rowwise().mean()).conjugate()).lazyProduct(U);
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
			if(beta != 0.0 || whitenimgs > 0) { // if only beta, then filter =
				// diagonal matrix Dyi = Yi * Conj(Yi) = power spectrum of yi, Dy = 1/(N*d) * SUM{ Dyi }
				bool recalc = true; // recalculate tmp if not already containing Dy
				int resetwhiten = whitenimgs;
				if(resetwhiten == 1 || resetwhiten > 3) {
					if(acount == 0) {
						resetwhiten = 0;
					} else if(icount == 0) {
						resetwhiten = 2;
					}
				}
				switch(resetwhiten) {
				case 2: // use authentic
				{
					if(acount > 0) {
						tmp.noalias() = X_hat.cwiseProduct(X_hat.conjugate()).lazyProduct(U.block(0,0,acount,1));
						tmp.real() = tmp.real() / (typename filter<T>::TVar)(N*d);
						tmp.imag() = tmp.imag() / (typename filter<T>::TVar)(N*d);
					} else {
						resetwhiten = 0;
					}
				}
				break;
				case 3: // use impostor
				{
					if(icount > 0) {
						tmp.noalias() = Y_hat.cwiseProduct(Y_hat.conjugate()).lazyProduct(U.block(0,0,icount,1));
						tmp.real() = tmp.real() / (typename filter<T>::TVar)(N*d);
						tmp.imag() = tmp.imag() / (typename filter<T>::TVar)(N*d); recalc = false;
					} else {
						tmp.real().setOnes(); tmp.imag().setZero();
					}
				}
				break;
				default: // use all
				{
					if(acount > 0 && icount > 0) {
						AllSamples.resize(d,N); AllSamples << X_hat, Y_hat;
					} else if(acount > 0) {
						AllSamples = X_hat;
					} else if(icount > 0) {
						AllSamples = Y_hat; recalc = false;
					}
					tmp.noalias() = AllSamples.cwiseProduct(AllSamples.conjugate()).lazyProduct(U);
					tmp.real() = tmp.real() / (typename filter<T>::TVar)(N*d);
					tmp.imag() = tmp.imag() / (typename filter<T>::TVar)(N*d);
				}
				break;
				}
				if(acount > 0) {
					AllSamples = X_hat;
				} else if(icount > 0) {
					AllSamples.resize(d,1); AllSamples.real().setOnes(); AllSamples.imag().setZero(); resetwhiten = 0;
				}
				if(resetwhiten > 0) {
					for(int i=0; i<acount; i++) {
						AllSamples.col(i) = AllSamples.col(i).cwiseQuotient(tmp);
					}
				}
				if(beta != 0.0) {
					if(recalc) {
						if(icount > 0) {
							tmp.noalias() = Y_hat.cwiseProduct(Y_hat.conjugate()).lazyProduct(U.block(0,0,icount,1));
							tmp.real() = tmp.real() * (typename filter<T>::TVar)(beta/N*d);
							tmp.imag() = tmp.imag() * (typename filter<T>::TVar)(beta/N*d);
						} else {
							tmp.real().setZero(); tmp.imag().setZero();
						}
					} else {
						tmp.real() = tmp.real() * (typename filter<T>::TVar)(beta);
						tmp.imag() = tmp.imag() * (typename filter<T>::TVar)(beta);
					}
					TT = TT + tmp;
				}
			} else {
				if(acount > 0) {
					AllSamples = X_hat;
				} else if(icount > 0) {
					AllSamples.resize(d,1); AllSamples.real().setOnes(); AllSamples.imag().setZero();
				}
			}

			TT = TT.cwiseInverse(); // T^(-1)

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
