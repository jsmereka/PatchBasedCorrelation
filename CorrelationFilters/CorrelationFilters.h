#include "stdafx.h"
#include <math.h>

#ifdef COMPILE_LIB
#define LIBEXP __declspec(dllexport)
#else
#define LIBEXP __declspec(dllimport)
#endif


using namespace Eigen;


/**
  
  Base filter class for 1-d and 2-d signals

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-13-2013    

*/
template <class T>									// designed for Matrix or Vector classes (non-complex)
class LIBEXP filter {
private:
	bool check_rank(T const& signal, bool auth);	// check the matrix rank against the authentic or imposter signals
	int get_rank(MatrixXd const& signal);			// get the matrix rank and return it as an int
	void addtoX(T const& signal);					// adds the signal to X		
	void addtoU(bool auth);							// add a 1 or 0 to vector U based on authentic or impostor signal

protected:
	MatrixXd X;										// matrix of signals (authentic and impostor) in the spatial domain
	MatrixXcd X_hat;								// matrix of signals (authentic and impostor) in the frequency domain
	VectorXd U;										// vector of zeros and ones designating authentic and impostors in X

	int input_row, input_col;						// size of first signal added (new inputs will be resized - padded or cropped - to this)

	T zero_pad(T const& signal, int siz1, int siz2, bool center);			// resizes signal window (zero pad/crop) based on input size

	void fft_scalar(T const& sig, MatrixXcd &sig_freq, int siz1, int siz2); // put signal into frequency domain
	void ifft_scalar(T &sig, MatrixXcd const& sig_freq);					// get signal from frequency domain

public:
	filter();
	~filter();

	MatrixXcd H_hat;								// the fitler itself in the frequency domain

	virtual void trainfilter() = 0;			// train the filter - this varies with each filter type

	int auth_count, imp_count;						// authentic and impostor counts

	bool add_auth(T &newsig);						// add an authentic class example for training
	bool add_imp(T &newsig);						// add an impostor class example for training
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
class LIBEXP OTSDF : public filter<T> {
private:
	VectorXcd P, D, S;								// Matrices for ONV, ACE, and ASM (dxd diagonal matrix where d = dimension, but set as vectors for efficient computation)

public:
	// constructors
	OTSDF(int alph, int bet, int gam) {				// initializes all three parameters
		alpha = alph; beta = bet; gamma = gam;
	}
	OTSDF(int lam) {								// initializes filter as a trade-off between ACE and ONV, ignoring ASM
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
};