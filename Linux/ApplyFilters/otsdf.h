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
	using filter<T>::X_hat;
	using filter<T>::Y_hat;
	using filter<T>::H;

public:
	// constructors
	OTSDF(double alph, double bet, double gam) {	// initializes all three parameters
		alpha = alph; beta = bet; gamma = gam;
	}
	OTSDF(double lam) {								// initializes filter as a trade-off between ACE and ONV, ignoring ASM
		alpha = 1 - lam; beta = lam; gamma = 0;
	}
	OTSDF() {										// initializes the parameters to their defaults
		alpha = pow(10,-5); beta = 1 - pow(10,-5); gamma = 0;
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
  D = diagonal matrix containing average power spectral density of the training data, complex matrix with dimension (d x d), diagonal matrix Di = Zi * Conj(Zi) = power spectrum of zi (where zi is an authentic or impostor data sample), D = 1/((Nx+Ny)*d) * SUM{ Di }
  S = diagonal matrix containing similarity in correlation planes to the true class mean, complex matrix with dimension (d x d), S = 1/(Nx*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
  P = diagonal matrix containing the power spectral density of the input noise, complex matrix with dimension (d x d), P = constant * identity matrix for additive white noise

  @author   Jonathon M. Smereka
  @version  06-14-2013

  @return   nothing

 */
template <class T>
void OTSDF<T>::trainfilter() {
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
			typename filter<T>::MatCplx AllSamples, SampTmp;
			typename filter<T>::VecCplx U(N);
			U.real().setOnes(); U.imag().setZero();

			TT.imag().setZero();

			if(alpha == 0.0 && beta == 0.0 && gamma == 0.0) {
				TT.real().setOnes(); // if no alpha, beta, or gamma, then filter = MVSDF under white noise = ECPSDF
			}

			// ASM
			if(gamma != 0.0 && acount > 0) { // if only gamma, then filter = MSESDF (constrained MACH)
				// diagonal matrix S = 1/(N*d) * SUM{ (Xi - mean(Xi)) * Conj(Xi - mean(Xi)) }
				TT.noalias() = (X_hat.colwise() - X_hat.rowwise().mean()).cwiseProduct((X_hat.colwise() - X_hat.rowwise().mean()).conjugate()).lazyProduct(U);
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
			if(beta != 0.0 || whitenimgs > 0) { // if only beta, then filter = MACE
				// diagonal matrix Di = Zi * Conj(Zi) = power spectrum of zi (where zi is an authentic or impostor data sample), D = 1/(N*d) * SUM{ Di }
				bool recalc = true; // recalculate tmp if not already containing D
				int resetwhiten = whitenimgs;
				if(resetwhiten == 1 || resetwhiten > 3) {
					if(acount == 0) {
						resetwhiten = 3; recalc = false;
					} else if(icount == 0) {
						resetwhiten = 2; recalc = false;
					}
				}
				switch(resetwhiten) {
				case 2: // use authentic
				{
					if(acount > 0) {
						if(icount > 0) {
							AllSamples.resize(d,N); AllSamples << X_hat, Y_hat;
						} else {
							AllSamples = X_hat; recalc = false;
						}
						tmp.noalias() = X_hat.cwiseProduct(X_hat.conjugate()).lazyProduct(U.block(0,0,acount,1));
						tmp.real() = tmp.real() / (typename filter<T>::TVar)(N*d);
						tmp.imag() = tmp.imag() / (typename filter<T>::TVar)(N*d);
					} else {
						AllSamples = Y_hat;
						tmp.real().setOnes(); tmp.imag().setZero();
					}
				}
				break;
				case 3: // use impostor
				{
					if(icount > 0) {
						if(acount > 0) {
							AllSamples.resize(d,N); AllSamples << X_hat, Y_hat;
						} else {
							AllSamples = Y_hat; recalc = false;
						}
						tmp.noalias() = Y_hat.cwiseProduct(Y_hat.conjugate()).lazyProduct(U.block(0,0,icount,1));
						tmp.real() = tmp.real() / (typename filter<T>::TVar)(N*d);
						tmp.imag() = tmp.imag() / (typename filter<T>::TVar)(N*d);
					} else {
						AllSamples =  X_hat;
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
						AllSamples = Y_hat;
					}
					tmp.noalias() = AllSamples.cwiseProduct(AllSamples.conjugate()).lazyProduct(U);
					tmp.real() = tmp.real() / (typename filter<T>::TVar)(N*d);
					tmp.imag() = tmp.imag() / (typename filter<T>::TVar)(N*d);
					recalc = false;
				}
				break;
				}
				if(resetwhiten > 0) {
					if(recalc) {
						SampTmp = AllSamples;
					}
					for(int i=0; i<N; i++) {
						AllSamples.col(i) = AllSamples.col(i).cwiseQuotient(tmp);
					}
				}
				if(beta != 0.0) {
					if(recalc) {
						tmp.noalias() = SampTmp.cwiseProduct(SampTmp.conjugate()).lazyProduct(U); SampTmp.resize(0,0);
						tmp.real() = tmp.real() * (typename filter<T>::TVar)(beta/N*d);
						tmp.imag() = tmp.imag() * (typename filter<T>::TVar)(beta/N*d);
					} else {
						tmp.real() = tmp.real() * (typename filter<T>::TVar)(beta);
						tmp.imag() = tmp.imag() * (typename filter<T>::TVar)(beta);
					}
					TT = TT + tmp;
				}
			} else {
				if(acount > 0 && icount > 0) {
					AllSamples.resize(d,N); AllSamples << X_hat, Y_hat;
				} else if(acount > 0) {
					AllSamples = X_hat;
				} else if(icount > 0) {
					AllSamples = Y_hat;
				}
			}

			TT = TT.cwiseInverse(); // T^(-1)

			/* Compute h as ECPSDF filter (X * (X^(+) *  X)^(-1) * U) but transform by T (saves some computation) */
			tmp.real().setZero(); tmp.imag().setZero();

			if(acount == 0) {
				U.real().setZero();
			} else if(icount > 0) {
				for(int i=acount; i<N; i++) {
					U(i).real() = (typename filter<T>::TVar)(0.0);
				}
			}

			// if number of input data samples is less than 5, then inverting ([X Y]^(+) * T^(-1) * [X Y]) can be done very quickly
			// note that ([X Y]^(+) * [X Y]) is positive semi-definite
			switch(N) {
			case 1:
				{
					typename filter<T>::VecCplx XX_inv; XX_inv.noalias() = (AllSamples.adjoint() * TT.asDiagonal() * AllSamples);
					tmp.noalias() = TT.asDiagonal() * AllSamples * XX_inv.cwiseInverse() * U;
				}
				break;
			case 2:
				{
					typename filter<T>::MatCplx2 XX_inv; XX_inv.noalias() = (AllSamples.adjoint() * TT.asDiagonal() * AllSamples);
					tmp.noalias() = TT.asDiagonal() * AllSamples * XX_inv.inverse() * U;
				}
				break;
			case 3:
				{
					typename filter<T>::MatCplx3 XX_inv; XX_inv.noalias() = (AllSamples.adjoint() * TT.asDiagonal() * AllSamples);
					tmp.noalias() = TT.asDiagonal() * AllSamples * XX_inv.inverse() * U;
				}
				break;
			case 4:
				{
					typename filter<T>::MatCplx4 XX_inv; XX_inv.noalias() = (AllSamples.adjoint() * TT.asDiagonal() * AllSamples);
					tmp.noalias() = TT.asDiagonal() * AllSamples * XX_inv.inverse() * U;
				}
				break;
			default:
				{
					typename filter<T>::MatCplx XX_inv(N,N); XX_inv.noalias() = (AllSamples.adjoint() * TT.asDiagonal() * AllSamples);
					// TODO: invert matrix greater than 4x4
				}
				break;
			}

			// transform the filter by T
			//HH = HH.cwiseProduct(TT);
			TT.resize(0);
			typename filter<T>::MatCplx H_hat;
			H_hat = Eigen::Map<typename filter<T>::MatCplx>(tmp.data(), input_row*rowmult, input_col*colmult);
			tmp.resize(0); H.resize(input_row*rowmult,input_col*colmult);
			filter<T>::ifft_scalar(H, H_hat);
		}

		filter<T>::cleanclass();
	}
}



#endif /* OTSDF_H_ */
