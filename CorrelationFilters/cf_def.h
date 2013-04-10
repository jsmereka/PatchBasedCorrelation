#include <Eigen/Dense>

using namespace Eigen;



/**
  
  Structures for 1-d and 2-d signals

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-09-2013    

*/
struct signal1d {
	VectorXd sig;			//input signal: spatial domain, 1-d dynamic double 
	VectorXcd sig_freq;		//input signal: frequency domain, 1-d dynamic complex double 
};

struct signal2d {
	MatrixXd sig;			//input signal: spatial domain, 2-d dynamic double 
	MatrixXcd sig_freq;		//input signal: frequency domain, 2-d dynamic complex double 
};



/**
  
  Filter for 1-d signal

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-09-2013    

*/
class filter1d {
private:
	bool check_rank(signal1d &signal, bool auth);	// check the matrix rank against the authentic or imposter signals

	void addtoX(signal1d &signal);					// adds the signal to X		
	void addtoU(bool auth);							// add a 1 or 0 to vector U based on authentic or impostor signal

protected:
	signal2d X;										// matrix of signals (authentic and impostor) in the spatial/frequency domain
	VectorXd U;										// vector of zeros and ones designating authentic and impostors in X

public:
	filter1d();
	~filter1d();

	void trainfilter();								// train the filter - this varies with each filter type

	int auth_count, imp_count;						// authentic and impostor counts

	bool add_auth(signal1d &newsig);				// add an authentic class example for training
	bool add_imp(signal1d &newsig);					// add an impostor class example for training
};


/**
  
  Filter for 2-d signal

  @author   Jonathon M. Smereka & Vishnu Naresh Boddetti
  @version  04-09-2013    

*/
class filter2d {
private:
	bool check_rank(signal2d &signal, bool auth);	// check the matrix rank against the authentic or imposter signals
	
	void addtoX(signal2d &signal);					// adds the signal to X		
	void addtoU(bool auth);							// add a 1 or 0 to vector U based on authentic or impostor signal

protected:
	signal2d X;										// matrix of signals (authentic and impostor) in the spatial/frequency domain
	VectorXd U;										// vector of zeros and ones designating authentic and impostors in X

public:
	filter2d();
	~filter2d();

	void trainfilter();								// train the filter - this varies with each filter type

	int auth_count, imp_count;						// authentic and impostor counts

	bool add_auth(signal2d &newsig);				// add an authentic class example for training
	bool add_imp(signal2d &newsig);					// add an impostor class example for training
};