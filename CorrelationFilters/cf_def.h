#include <Eigen/Dense>

using namespace Eigen;


/**
  
  Base filter class for 1-d and 2-d signals

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-10-2013    

*/
template <class T>									// designed for Matrix or Vector classes (non-complex)
class filter {
private:
	bool check_rank(T const& signal, bool auth);	// check the matrix rank against the authentic or imposter signals
	int get_rank(MatrixXd const& signal);			// get the matrix rank and return it as an int
	void addtoX(T const& signal);					// adds the signal to X		
	void addtoU(bool auth);							// add a 1 or 0 to vector U based on authentic or impostor signal
	
protected:
	MatrixXd X;										// matrix of signals (authentic and impostor) in the spatial domain
	MatrixXcd X_hat;								// matrix of signals (authentic and impostor) in the frequency domain
	VectorXd U;										// vector of zeros and ones designating authentic and impostors in X

	void fft_scalar(T const& sig, MatrixXcd &sig_freq, int siz1, int siz2); // put signal into frequency domain
	void ifft_scalar(T &sig, MatrixXcd const& sig_freq);					// get signal from frequency domain

public:
	filter();
	~filter();

	virtual void trainfilter() const = 0;			// train the filter - this varies with each filter type

	int auth_count, imp_count;						// authentic and impostor counts

	bool add_auth(T &newsig);						// add an authentic class example for training
	bool add_imp(T &newsig);						// add an impostor class example for training
};

