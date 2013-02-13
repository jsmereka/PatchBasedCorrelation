
#include "buildfilter.h"


// Constructor
FILTstruct::FILTstruct() {

	configlock = true;


}


// Destructor
FILTstruct::~FILTstruct() {
	//
}


void FILTstruct::configure(RunOptions InputParams) {

	buildfilters = InputParams.build_fitlers;
	true_train = InputParams.true_train;
	false_train = InputParams.false_train;
	st_dir = InputParams.st_dir;
	dataset = InputParams.dataset;
	subset_num = InputParams.subset_num;
	left_right = InputParams.left_right;
	height = InputParams.height;
	width = InputParams.width;

	if(st_dir.back() != '/')
		st_dir.push_back('/');
	if(!checkdirectory(st_dir) || !checkdirectory(st_dir + dataset + "/") || !checkdirectory(st_dir + dataset + "/cfs/")) {
		DRWN_LOG_WARNING("Setting to build fitlers");
		getnewimgs = true;
	}

	stringstream name, name2, name3;
	name.clear();
	if(subset_num > 0) {
		checkdirectory(st_dir + dataset + "/cfs/sub" + intToString(subset_num));
		name << st_dir << dataset << "/cfs/sub" << subset_num << "/CFsL_" << width << "x" << height;
	} else {
		name << st_dir << dataset << "/cfs/" << "CFsL_" << width << "x" << height;
	}
	leftname = name.str();

	name.clear();
	if(subset_num > 0) {
		name << st_dir  << dataset << "/cfs/sub" << subset_num << "/CFsR_" << width << "x" << height;
	} else {
		name << st_dir << dataset << "/cfs/" << "CFsR_" << width << "x" << height;
	}
	rightname = name.str();

	configlock = false;

}



void FILTstruct::getfilters() {
	if(!configlock) {
		if(buildfilters) {
			//
		} else {
			// load up binary
			if(left_right == "left" || left_right == "both") {

			}
			if(left_right == "right" || left_right == "both") {

			}
		}
	} else {
		DRWN_LOG_WARNING("Cannot get filters until parameters are configured");
	}
}