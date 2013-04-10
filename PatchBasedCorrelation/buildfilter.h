// buildfilter.h : class and functions for building filters
//

#include "getimgs.h"
#include "fftw3.h"

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

public:
	FILTstruct();
	~FILTstruct();

	void configure(RunOptions InputParams);

	void getfilters();

};