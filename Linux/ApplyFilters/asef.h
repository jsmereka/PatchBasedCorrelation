#ifndef ASEF_H_
#define ASEF_H_


#include "cfs.h"

/**

  Average of Synthetic Exact Filter (ASEF)
  Ref: Bolme, D.S.; Draper, B.A.; Beveridge, J.R., "Average of Synthetic Exact Filters," CVPR, pp.2105,2112, 20-25 June 2009

  h = mean(Hi)
  Hi = Gi ./ conj(Zi)
  Gi = 2D DFT of desired correlation plane from the training image Zi

  Z = column matrix of the true or false class training samples

  @author   Jonathon M. Smereka
  @version  06-21-2013

 */
template <class T>
class ASEF : public filter<T> {
private:
	using filter<T>::zeropadtrnimgs;
	using filter<T>::whitenimgs;
	using filter<T>::cleanupaftertrain;
	using filter<T>::input_row;
	using filter<T>::input_col;
	using filter<T>::X_hat;
	using filter<T>::H;

public:
	// constructors
	ASEF(double alph, double bet, double gam) {	// initializes all three parameters
		alpha = alph; beta = bet; gamma = gam;
	}
	ASEF(double lam) {								// initializes filter as a trade-off between ACE and ONV, ignoring ASM
		alpha = 1 - lam; beta = lam; gamma = 0;
	}
	ASEF() {										// initializes the parameters to their defaults
		alpha = pow(10,-5); beta = 1 - pow(10,-5); gamma = 0;
	}
	// destructor
	~ASEF() { }

	double alpha, beta, gamma;						// training parameters

	void trainfilter();								// train the filter, putting data into base class variable H

	EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};



/**

  Trains the filter as an ASEF

  h = mean(Hi)
  Hi = Gi ./ conj(Zi)

  h = derived filter, complex with dimension (d x 1), d = input_row * input_col, for 2D signals, it will be reshaped to (input_row x input_col)
  X = column matrix of input true class data, complex matrix with dimension (d x Nx), Nx = auth_count
  Y = column matrix of input false class data, complex matrix with dimension (d x Ny), Ny = imp_count
  Z = column matrix of the true or false class training samples, complex matrix with dimension (d x (Nx+Ny)), Z = [X Y]
  G = column matrix of the desired correlation plane from the training image Zi, complex matrix with dimension (d x (Nx+Ny)), Gi = Gaussian at location of match if authentic, and flat (small number close to zero) if impostor

  @author   Jonathon M. Smereka
  @version  06-21-2013

  @return   nothing

 */
template <class T>
void ASEF<T>::trainfilter() {
	//
}
}



#endif /* ASEF_H_ */
