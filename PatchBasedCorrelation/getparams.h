// getparams.h : class and functions for loading up parameters
//

// c++ standard headers
#include <cstdlib>
#include <cstdio>
#include <vector>

// darwin library headers
#include "drwnBase.h"


using namespace std;

struct RunOptions {
	// what to load
	string dataset;
	int subset_num;
	string ld_dir, st_dir;

	// how to format it
	int height, width;
	string segchoice;
	int rows, cols, numpatches;
	int segheight, segwidth;

	// how to compare
	string sim_measure;
	double comp_ratio;
	int true_train, false_train;

	// run options
	string left_right;
	bool get_images, build_fitlers, compare;
};


// configuration manager -----------------------------------------------------

class Parameters : public drwnConfigurableModule {
private:
	bool paramlock; // lock until check is complete
	bool datasetchange; // lock if dataset location is defined by user
	int def_height, def_width; // original height and width of dataset image

	bool runwin; // windows or unix (for directory defaults)

	string initial_subsetID;
	vector<string> directory;

public:
   Parameters();
   ~Parameters() { }

   RunOptions par;

   void setConfiguration(const char *name, const char *value);

   string subsetID();

   void checkParameters();

   void printParameters();

   void usage(ostream &os) const {
	   os << "    Available configurable parameters\n";
	   os << "      dataset         :: which dataset to load/run (default: " << par.dataset << ")\n";
	   os << "      subset          :: integer of which subset to use (default: " << initial_subsetID << ")\n";
	   os << "      img_height      :: (if != 0) resize image (default: " << par.height << " pixels)\n";
	   os << "      img_width       :: (if != 0) resize image (default: " << par.width << " pixels)\n";
	   os << "      seg_style       :: how to segment the image (default: " << par.segchoice << ")\n";
	   os << "      seg_height      :: size at which to get superpixels (default: " << par.segheight << " pixels)\n";
	   os << "      seg_width       :: size at which to get superpixels (default: " << par.segwidth << " pixels)\n";
	   os << "      rows            :: split image into rectangular patches (default: " << par.rows << " rows)\n";
	   os << "      cols            :: split image into rectangular patches (default: " << par.cols << " cols)\n";
	   os << "      num_patches     :: # of patches when using superpixels (default: " << par.numpatches << ")\n";
	   os << "      sim_measure     :: CF plane similarity measurement (default: " << par.sim_measure << ")\n";
	   os << "      true_train      :: # of true class images to train filters (default: " << par.true_train << ")\n";
	   os << "      false_train     :: # of false class images to train filters (default: " << par.false_train << ")\n";
	   os << "      compare_ratio   :: ratio of space to compare patch against (default: " << par.comp_ratio << ")\n";
	   os << "      dir             :: directory to load dataset (default: " << endl;
	   if(directory.size() > 0) {
		   for(int i=0; i<directory.size(); i++){
			   if(i == directory.size()-1) {
				   os << "                           " << directory[i] << ")\n";
			   } else {
				   os << "                           " << directory[i] << endl;
			   }
		   }
	   }
	   os << "      st_dir          :: directory to load/store results (default: " << par.st_dir << ")\n";
	   os << endl;
	   os << "    Available runtime parameters\n";
	   os << "      left_right      :: run left or right ocular region (default: " << par.left_right << ")\n";
	   os << "      get_images      :: get the images from file (default: " << par.get_images << ")\n";
	   os << "      build_filters   :: build the correlation filters (default: " << par.build_fitlers << ")\n";
	   os << "      compare         :: performs matching (default: " << par.compare << ")\n";
   }

};


