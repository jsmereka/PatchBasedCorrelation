

#include "getparams.h"



// Constructor
Parameters::Parameters() : drwnConfigurableModule("Params") {
	// initialize
	par.dataset = "FOCS";
	par.subset_num = 1;
	par.height = 128; par.width = 128;
	par.segheight = par.height; par.segwidth = par.width;
	par.rows = 6; par.cols = 6; par.numpatches = 12;
	par.segchoice = "rectangular";
	def_height = 0; def_width = 0;
	initial_subsetID = subsetID();
	par.sim_measure = "peak height";
	par.left_right = "left";
	par.comp_ratio = 2.0;
	par.true_train = 1; par.false_train = 0;
	runwin = true;
	par.st_dir = "/";
	par.get_images = false;
	par.build_fitlers = true;
	par.compare = true;

	// lock parameters until check is complete
	paramlock = true; datasetchange = false;
}


// Set the parameters as they load
void Parameters::setConfiguration(const char *name, const char *value) {
	if(!strcmp(name, "dataset")) {
		par.dataset = string(value);
	} else if(!strcmp(name, "subset")) {
		par.subset_num = atoi(value);
	} else if(!strcmp(name, "img_height")) {
		par.height = atoi(value);
	} else if(!strcmp(name, "img_width")) {
		par.width = atoi(value);
	} else if(!strcmp(name, "seg_style")) {
		par.segchoice = string(value);
	} else if(!strcmp(name, "seg_height")) {
		par.segheight = atoi(value);
	} else if(!strcmp(name, "seg_width")) {
		par.segwidth = atoi(value);
	} else if(!strcmp(name, "rows")) {
		par.rows = atoi(value);
	} else if(!strcmp(name, "cols")) {
		par.cols = atoi(value);
	} else if(!strcmp(name, "num_patches")) {
		par.numpatches = atoi(value);
	} else if(!strcmp(name, "sim_measure")) {
		par.sim_measure = string(value);
	} else if(!strcmp(name, "true_train")) {
		par.true_train = atoi(value);
	} else if(!strcmp(name, "false_train")) {
		par.false_train = atoi(value);
	} else if(!strcmp(name, "compare_ratio")) {
		par.comp_ratio = atof(value);
	} else if(!strcmp(name, "left_right")) {
		par.left_right = string(value);
	} else if(!strcmp(name, "get_images")) {
		par.get_images = (atoi(value) != 0);
	} else if(!strcmp(name, "build_filters")) {
		par.build_fitlers = (atoi(value) != 0);
	} else if(!strcmp(name, "compare")) {
		par.compare = (atoi(value) != 0);
	} else if(!strcmp(name, "st_dir")) {
		par.st_dir = string(value);
	} else if(!strcmp(name, "dir")) {
		datasetchange = true;
		directory.clear();
		directory.push_back(string(value));
	} else {
		DRWN_LOG_FATAL("unrecognized configuration option " << name << " for " << this->name());
	}
}


// Validate the parameters before running any tests 
void Parameters::checkParameters() {
	// check dataset
	string tmp = subsetID();
	if(tmp == "error2") {
		DRWN_LOG_FATAL("the dataset choice of " << par.dataset << " is not available");
	} else if(tmp == "error1") {
		DRWN_LOG_FATAL("subset choice is not available for the " << par.dataset << " dataset");
	}

	// check image settings
	if(par.height <= 0)
		par.height = def_height;
	if(par.height < 32)
		DRWN_LOG_WARNING("image height selection is less than 32 pixels : currently set at " << par.height);
	if(par.width <= 0)
		par.width = def_width;
	if(par.width < 32)
		DRWN_LOG_WARNING("image width selection is less than 32 pixels : currently set at " << par.width);

	// check segmentation style
	if(par.segchoice == "rectangular") {
		if(par.rows <= 0) {
			par.rows = 1;
			DRWN_LOG_WARNING("resetting rows entry");
		}
		if(par.cols <= 0) {
			par.cols = 1;
			DRWN_LOG_WARNING("resetting columns entry");
		}
		if(par.rows == 1 && par.cols == 1)
			DRWN_LOG_WARNING("performing full image comparison - no segmentation");
	} else if(par.segchoice == "superpixel") {
		if(par.numpatches <= 0) {
			par.numpatches = 1;
		}
		if(par.numpatches == 1) {
			DRWN_LOG_WARNING("performing full image comparison - no segmentation");
		}
		if(par.segheight <= 0)
			par.segheight = par.height;
		if(par.segheight < 32)
			DRWN_LOG_WARNING("segmentation image height selection is less than 32 pixels : currently set at " << par.segheight);
		if(par.segwidth <= 0)
			par.segwidth = par.width;
		if(par.segwidth < 32)
			DRWN_LOG_WARNING("segmentation image width selection is less than 32 pixels : currently set at " << par.segwidth);
	} else if(par.segchoice == "dynamic") {
		//
	} else {
		DRWN_LOG_FATAL("segmentation choice of: " << par.segchoice << " is not available");
	}

	// build options
	if(par.sim_measure == "peak height") {
		//
	} else if(par.sim_measure == "pce") {
		//
	} else if(par.sim_measure == "psr") {
		//
	} else {
		DRWN_LOG_FATAL("similarity measurement of: " << par.sim_measure << " is not available");
	}
	if(par.comp_ratio < 1.0) 
		par.comp_ratio = 1.0;
	if(par.true_train < 1)
		par.true_train = 1;
	if(par.false_train < 0)
		par.false_train = 0;

	// runtime options
	if(par.left_right != "left" && par.left_right != "right" && par.left_right != "both")
		DRWN_LOG_FATAL("comparisons can be done on either the left, right, or both ocular regions");

	if(par.get_images && !par.build_fitlers && par.compare) {
		par.build_fitlers = true;
		DRWN_LOG_WARNING("After loading images, filters must be built in order to properly perform matching");
	}
	if(!datasetchange) {
		par.ld_dir = directory[par.subset_num];
	} else {
		par.ld_dir = directory[0];
	}
	if(!par.st_dir.empty())
		if(par.st_dir.back() != '/')
			par.st_dir.push_back('/');
	par.st_dir += "results/";

	paramlock = false;
	printParameters();
}


// Return a string of the subset choice
string Parameters::subsetID() {
	if(par.dataset == "FOCS") {
		def_height = 600; def_width = 750;
		if(!datasetchange) {
			directory.clear();
			if(runwin) {
				directory.push_back("E:/Datasets/Ocular/FOCS/full/");
				directory.push_back("E:/Datasets/Ocular/FOCS/min_focs/");
				directory.push_back("E:/Datasets/Ocular/FOCS/min_focs2/");
				directory.push_back("E:/Datasets/Ocular/FOCS/min_focs3/");
			} else {
				directory.push_back("/afs/ece.cmu.edu/project/kumar/work1/focs/OcularStillChallenge1/BEST_Data/MBGC_FaceNIRVideo_Clips/");
				directory.push_back("/afs/ece.cmu.edu/project/kumar/work5/smerekaj/Ocular/min_focs/");
				directory.push_back("/afs/ece.cmu.edu/project/kumar/work5/smerekaj/Ocular/min_focs2/");
				directory.push_back("/afs/ece.cmu.edu/project/kumar/work5/smerekaj/Ocular/min_focs3/");
			}
		}
		switch(par.subset_num) {
		case 0:
			return "full set";
		case 1:
			return "74 img left eye set";
		case 2:
			return "404 img left eye set";
		case 3:
			return "309 img set";
		default:
			return "error1";
		}
	} else if(par.dataset == "BDCP") {
		if(!datasetchange) {
			directory.clear();
			if(runwin) {
				directory.push_back("E:/Datasets/Ocular/BDCP/all/");
				directory.push_back("E:/Datasets/Ocular/BDCP/CFAIRS/");
				directory.push_back("E:/Datasets/Ocular/BDCP/LG4000/");
			} else {
				directory.push_back("/afs/ece.cmu.edu/project/kumar/work1/bdcp_algos/data_processed/");
				directory.push_back("/afs/ece.cmu.edu/project/kumar/work1/bdcp_algos/data_processed/");
				directory.push_back("/afs/ece.cmu.edu/project/kumar/work1/bdcp_algos/data_processed/");
			}
		}
		switch(par.subset_num) {
		case 0:
			return "CFAIRS vs LG4000";
			def_height = 480; def_width = 640;
		case 1:
			return "CFAIRS";
			def_height = 600; def_width = 750;
		case 2:
			return "LG4000";
			def_height = 480; def_width = 640;
		default:
			return "error1";
		}
	} else if(par.dataset == "FRGC") {
		if(!datasetchange) {
			directory.clear();
			if(runwin) {
				directory.push_back("E:/Datasets/Ocular/FRGC/cropped_imgs/");
			} else {
				directory.push_back("/afs/ece.cmu.edu/project/kumar/work5/smerekaj/Ocular/FRGC_cropped/");
			}
		}
		if(par.subset_num == 0) {
			return "full set";
			def_height = 241; def_width = 226;
		} else {
			return "error1";
		}
	} else {
		return "error2";
	}
}


void Parameters::printParameters() {
	if(!paramlock){
		if(par.get_images || par.build_fitlers || par.compare){
			cout << "\nOperations: ";
			if(par.get_images && par.build_fitlers && par.compare)
				cout << "Getting Images, Building Filters, and Performing Matching\n";
			if(par.get_images && par.build_fitlers && !par.compare)
				cout << "Getting Images, and Building Filters\n";
			if(!par.get_images && par.build_fitlers && par.compare)
				cout << "Building Filters, and Performing Matching\n";
			if(!par.get_images && !par.build_fitlers && par.compare)
				cout << "Performing Matching\n";
			if(!par.get_images && par.build_fitlers && !par.compare)
				cout << "Building Filters\n";
			if(par.get_images && !par.build_fitlers && !par.compare)
				cout << "Getting Images\n";

			if(par.left_right == "both") {
				cout << "Tests on both ocular regions invidually will be performed\n";
			} else {
				cout << "Tests will only be performed on the " << par.left_right << " ocular region\n";
			}
			cout << "\nRunning on " << par.dataset << " - " << subsetID() << endl;
			if(datasetchange)
				cout << " -- directory change to: " << directory[0] << endl;
			if(par.height != def_height || par.width != def_width)
				cout << "\nImages resized to: " << par.height << "x" << par.width << " pixels\n";
			else
				cout << "Images at default height and width: " << par.height << "x" << par.width << " pixels\n";
			cout << "Segmentation style: " << par.segchoice;
			if(par.segchoice == "rectangular")
				cout << " using a " << par.rows << "x" << par.cols << " configuration\n";
			else if(par.segchoice == "superpixel") {
				cout << " using at most " << par.numpatches << " patches\n";
				cout << "Segmentation image size: " << par.segheight << "x" << par.segwidth << " pixels\n";
			}
			cout << "\nCorrelation similarity measure: " << par.sim_measure << " with a comparison ratio of " << par.comp_ratio << endl;
			cout << "Filters are trained with: " << par.true_train << " authentic and " << par.false_train << " impostor images\n\n";
		} else {
			DRWN_LOG_WARNING("No Operations Set - Program Exiting");
		}
	} else {
		cout << "Parameters have not yet been verified\n" << endl;
	}
}
