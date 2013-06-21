#ifndef OTSDF_H_
#define OTSDF_H_


#include "cfs.h"

/**

  Optimal Trade-off Synthetic Discriminant Function (OTSDF) correlation filter
  Ref: P. Refregier, "Filter design for optical pattern recognition: multicriteria optimization approach," Opt. Lett.  15, 854-856 (1990).

  h = T^(-1) * [X Y] * ([X Y]^(+) * T^(-1) * [X Y])^(-1) * U
  T = alpha * P + beta * D + gamma * S
	or
  T = lambda * D + (1 - lambda) * P

  U = vector of training labels
  X = column matrix of the true class training samples
  Y = column matrix of the false class training samples
  P = output noise variance (ONV) = minimizing the effect of additive noise
  D = average correlation energy (ACE) = minimizing the energy in the resulting correlation plane (creating sharp peaks)
  S = average similarity measure (ASM) = minimizing variance in peak height

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-13-2013

 */
template <class T>
class OTSDF : public filter<T> {
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
	using filter<T>::U;

public:
	// constructors
	OTSDF(double alph, double bet, double gam) {	// initializes all three parameters
		alpha = alph; beta = bet; gamma = gam; trainedflag = false;
	}
	OTSDF(double lam) {								// initializes filter as a trade-off between ACE and ONV, ignoring ASM
		alpha = 1 - lam; beta = lam; gamma = 0; trainedflag = false;
	}
	OTSDF() {										// initializes the parameters to their defaults
		alpha = pow(10,-5); beta = 1 - pow(10,-5); gamma = 0; trainedflag = false;
	}
	// destructor
	~OTSDF() { }

	double alpha, beta, gamma;						// training parameters

	void trainfilter();								// train the filter, putting data into base class variable H

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



/**

  Trains the filter as an OTSDF

  h = T^(-1) * [X Y] * ([X Y]^(+) * T^(-1) * [X Y])^(-1) * U
  T = alpha * P + beta * D + gamma * S
	or
  T = lambda * D + (1 - lambda) * P

  h = derived filter, complex with dimension (d x 1), d = input_row * input_col, for 2D signals, it will be reshaped to (input_row x input_col)
  X = column matrix of input true class data, complex matrix with dimension (d x Nx), Nx = auth_count
  Y = column matrix of input false class data, complex matrix with dimension (d x Ny), Ny = imp_count
  D = diagonal matrix containing average power spectral density of the training data, complex matrix with dimension (d x d), diagonal matrix Di = [Xi Yi] * Conj([Xi Yi]) = power spectrum of xi & yi, D = 1/((Nx+Ny)*d) * SUM{ Di }
  S = diagonal matrix containing similarity in correlation planes to the true class mean, complex matrix with dimension (d x d), S = 1/(Nx*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
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
				TT.real().setOnes(); // if no alpha, beta, or gamma, then filter = MVSDF under white noise = ECPSDF
			} else {
				typename filter<T>::VecCplx vec_of_ones(N);
				vec_of_ones.real().setOnes(); vec_of_ones.imag().setZero();
				// ASM
				if(gamma != 0.0) { // if only gamma, then filter = MSESDF (constrained MACH)
					// TODO: fix for only true class data
					// diagonal matrix S = 1/(N*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
					TT.noalias() = (X_hat.colwise() - X_hat.rowwise().mean()).cwiseProduct((X_hat.colwise() - X_hat.rowwise().mean()).conjugate()).lazyProduct(vec_of_ones);
					TT.real() = TT.real() * (typename filter<T>::TVar)(gamma/(N*d));
					TT.imag() = TT.imag() * (typename filter<T>::TVar)(gamma/(N*d));
				} else {
					TT.real().setZero();
				}
				// ONV
				if(alpha != 0.0) { // if only alpha, then filter = MVSDF
					// diagonal matrix P = constant * identity for additive white noise
					tmp.real().setOnes(); tmp.imag().setZero();
					tmp.real() = tmp.real() * (typename filter<T>::TVar)(alpha);
					TT = TT + tmp;
				}
				// ACE
				if(beta != 0.0 || whitenimgs) { // if only beta, then filter = MACE
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

			/* Compute h as ECPSDF filter (X * (X^(+) *  X)^(-1) * U) but transform by T (saves some computation) */
			tmp.real().setZero(); tmp.imag().setZero();

			// if number of input data samples is less than 5, then inverting (X^(+) * T^(-1) * X) can be done very quickly
			// note that (X^(+) * X) is positive semi-definite
			switch(N) {
			case 1:
				{
					typename filter<T>::VecCplx XX_inv; XX_inv.noalias() = (X_hat.adjoint() * TT.asDiagonal() * X_hat);
					tmp.noalias() = TT.asDiagonal() * X_hat * XX_inv.cwiseInverse() * U;
				}
				break;
			case 2:
				{
					typename filter<T>::MatCplx2 XX_inv; XX_inv.noalias() = (X_hat.adjoint() * TT.asDiagonal() * X_hat);
					tmp.noalias() = TT.asDiagonal() * X_hat * XX_inv.inverse() * U;
				}
				break;
			case 3:
				{
					typename filter<T>::MatCplx3 XX_inv; XX_inv.noalias() = (X_hat.adjoint() * TT.asDiagonal() * X_hat);
					tmp.noalias() = TT.asDiagonal() * X_hat * XX_inv.inverse() * U;
				}
				break;
			case 4:
				{
					typename filter<T>::MatCplx4 XX_inv; XX_inv.noalias() = (X_hat.adjoint() * TT.asDiagonal() * X_hat);
					tmp.noalias() = TT.asDiagonal() * X_hat * XX_inv.inverse() * U;
				}
				break;
			default:
				{
					typename filter<T>::MatCplx XX_inv(N,N); X_inv.noalias() = (X_hat.adjoint() * TT.asDiagonal() * X_hat);
					// TODO: invert matrix greater than 4x4
				}
				break;
			}

			// transform the filter by T
			//HH = HH.cwiseProduct(TT);

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



#endif /* OTSDF_H_ */
