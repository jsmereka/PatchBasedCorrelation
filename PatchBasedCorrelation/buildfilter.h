// buildfilter.h : class and functions for building filters
//

// project libraries
#include "getimgs.h"
#include <fftw3.h>
#include <complex>
#include <Eigen/Dense>
#include <Eigen/Core>

// correlation filter library
#include <CorrelationFilters.h>


class FILTstruct {
private:
	// params
	string st_dir;
	string left_right, dataset;
	int height, width, subset_num;
	bool getnewimgs;
	int true_train, false_train;
	bool buildfilters;

	bool configlock; // configuration lock to ensure parameters are set

	string leftname, rightname; // save/load names

	OTSDF<MatrixXd> thefilt;

public:
	FILTstruct();
	~FILTstruct();

	void configure(RunOptions InputParams);

	void getfilters();

};