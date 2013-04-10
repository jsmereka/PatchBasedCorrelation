#include "cf_def.h"
#include "cf_util.h"
#include <algorithm>



/*********************/
/*    1-d Signals    */
/*********************/


/*
	Constructor
*/
filter1d::filter1d() {
	auth_count = 0; imp_count = 0;
}



/*
	Destructor
*/
filter1d::~filter1d() {
	X.sig.resize(0,0);
	X.sig_freq.resize(0,0);
	U.resize(0);
}



/**
  
  Adds an authentic signal to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    newsig		the 1-d signal that is to be added
  
  @return   true or false if successfully added

*/
bool filter1d::add_auth(signal1d &newsig) {
	return check_rank(newsig, 1);
}



/**
  
  Adds an imposter signal to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    newsig		the 1-d signal that is to be added
  
  @return   true or false if successfully added

*/
bool filter1d::add_imp(signal1d &newsig) {
	return check_rank(newsig, 0);
}



/**
  
  Check the matrix rank against the other authentic or imposter signals

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    signal		the 1-d signal that is to be added
  @param	auth		1 = authentic, 0 = imposter
  
  @return   true or false if to be added to the vector

*/
bool filter1d::check_rank(signal1d &signal, bool auth) {
	int rank = 0;
	if((auth && auth_count == 0) || (!auth && imp_count == 0)) {
		addtoX(signal);
		addtoU(auth);
		return true;
	} else {
		// put signal into matrix
		MatrixXd combsignal = X.sig;
		combsignal.conservativeResize(X.sig.rows(),X.sig.cols()+1);
		for(int i=0; i<signal.sig.rows(); i++) {
			combsignal(i,X.sig.cols()) = signal.sig(i);
		}
		// find rank
		rank = get_rank(combsignal);
		if(rank > 0) {
			addtoX(signal);
			addtoU(auth);
			return true;
		}
	}
	return false;
}



/**
  
  Add vectorized signal to X matrix

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    signal		the 1-d signal that is to be added
  
  @return   nothing

*/
void filter1d::addtoX(signal1d &signal) {
	// put into matrix
	if(X.sig.rows() == 0 || X.sig.cols() == 0) {
		X.sig.resize(signal.sig.rows()+signal.sig.cols(),1);
	} else {
		X.sig.conservativeResize(X.sig.rows(),X.sig.cols()+1);
	}
	for(int i=0; i<signal.sig.rows(); i++) {
		X.sig(i,X.sig.cols()) = signal.sig(i);
	}
	// put into matrix
	if(signal.sig_freq.rows() == 0 || signal.sig_freq.cols() == 0) {
		// TODO
		// compute fft
		//fft2_image_scalar(signal.sig, signal.sig_freq, signal.sig.rows(), signal.sig.cols());
	}
	if(X.sig_freq.rows() == 0 || X.sig_freq.cols() == 0) {
		X.sig_freq.resize(signal.sig_freq.rows()+signal.sig_freq.cols(),1);
	} else {
		X.sig_freq.conservativeResize(X.sig_freq.rows(),X.sig_freq.cols()+1);
	}
	for(int i=0; i<signal.sig_freq.rows(); i++) {
		X.sig_freq(i,X.sig_freq.cols()).real() = signal.sig_freq(i).real();
		X.sig_freq(i,X.sig_freq.cols()).imag() = signal.sig_freq(i).imag();
	}
}



/**
  
  Add a 1 or 0 to vector U based on authentic or impostor signal

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    auth		whether to add a 1 or 0
  
  @return   nothing

*/
void filter1d::addtoU(bool auth) {
	if(auth) {
		auth_count++;
		if(U.size() == 0) {
			U.resize(1);
			U(0) = 1;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1) = 1;
		}
	} else {
		imp_count++;
		if(U.size() == 0) {
			U.resize(1);
			U(0) = 0;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1) = 0;
		}
	}
}



/*********************/
/*    2-d Signals    */
/*********************/



/*
	Constructor
*/
filter2d::filter2d() {
	auth_count = 0; imp_count = 0;
}



/*
	Destructor
*/
filter2d::~filter2d() {
	X.sig.resize(0,0);
	X.sig_freq.resize(0,0);
	U.resize(0);
}



/**
  
  Adds an authentic signal to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    newsig		the 2-d signal that is to be added
  
  @return   true or false if successfully added

*/
bool filter2d::add_auth(signal2d &newsig) {
	return check_rank(newsig, 1);
}



/**
  
  Adds an imposter signal to those used to train the filter

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    newsig		the 2-d signal that is to be added
  
  @return   true or false if successfully added

*/
bool filter2d::add_imp(signal2d &newsig) {
	return check_rank(newsig, 0);
}



/**
  
  Check the matrix rank against the other authentic or imposter signals

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    signal		the 2-d signal that is to be added
  @param	auth		1 = authentic, 0 = imposter
  
  @return   true or false if to be added to the vector

*/
bool filter2d::check_rank(signal2d &signal, bool auth) {
	int rank = 0;
	if((auth && auth_count == 0) || (!auth && imp_count == 0)) {
		// find rank of the signal
		rank = get_rank(signal.sig);
		if(rank > 0) {
			addtoX(signal);
			addtoU(auth);
			return true;
		}
	} else {
		// put signal into matrix
		MatrixXd combsignal = X.sig;
		combsignal.conservativeResize(X.sig.rows(),X.sig.cols()+1);
		int ct = 0;
		for(int i=0; i<signal.sig.rows(); i++) {
			for(int j=0; j<signal.sig.cols(); j++) {
				combsignal(ct,X.sig.cols()) = signal.sig(i,j);
				ct++;
			}
		}
		// find rank
		rank = get_rank(combsignal);
		if(rank > 0) {
			addtoX(signal);
			addtoU(auth);
			return true;
		}
	}
	return false;
}



/**
  
  Add vectorized signal to X matrix

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    signal		the 2-d signal that is to be added
  
  @return   nothing

*/
void filter2d::addtoX(signal2d &signal) {
	// vectorize 2-d spatial signal and put into matrix
	if(X.sig.rows() == 0 || X.sig.cols() == 0) {
		X.sig.resize(signal.sig.rows()+signal.sig.cols(),1);
	} else {
		X.sig.conservativeResize(X.sig.rows(),X.sig.cols()+1);
	}
	int ct = 0;
	for(int i=0; i<signal.sig.rows(); i++) {
		for(int j=0; j<signal.sig.cols(); j++) {
			X.sig(ct,X.sig.cols()) = signal.sig(i,j);
			ct++;
		}
	}
	// vectorize 2-d frequency signal and put into matrix
	if(signal.sig_freq.rows() == 0 || signal.sig_freq.cols() == 0) {
		// compute fft
		fft2_image_scalar(signal.sig, signal.sig_freq, signal.sig.rows(), signal.sig.cols());
	}
	if(X.sig_freq.rows() == 0 || X.sig_freq.cols() == 0) {
		X.sig_freq.resize(signal.sig_freq.rows()+signal.sig_freq.cols(),1);
	} else {
		X.sig_freq.conservativeResize(X.sig_freq.rows(),X.sig_freq.cols()+1);
	}
	ct = 0;
	for(int i=0; i<signal.sig_freq.rows(); i++) {
		for(int j=0; j<signal.sig_freq.cols(); j++) {
			X.sig_freq(ct,X.sig_freq.cols()).real() = signal.sig_freq(i,j).real();
			X.sig_freq(ct,X.sig_freq.cols()).imag() = signal.sig_freq(i,j).imag();
			ct++;
		}
	}
}



/**
  
  Add a 1 or 0 to vector U based on authentic or impostor signal

  @author   Jonathon M. Smereka
  @version  04-03-2013

  @param    auth		whether to add a 1 or 0
  
  @return   nothing

*/
void filter2d::addtoU(bool auth) {
	if(auth) {
		auth_count++;
		if(U.size() == 0) {
			U.resize(1);
			U(0) = 1;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1) = 1;
		}
	} else {
		imp_count++;
		if(U.size() == 0) {
			U.resize(1);
			U(0) = 0;
		} else {
			U.conservativeResize(auth_count+imp_count);
			U(auth_count+imp_count-1) = 0;
		}
	}
}