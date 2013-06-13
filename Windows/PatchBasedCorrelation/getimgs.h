// getimgs.h : class and functions for loading up the images
//

// project libraries
#include "stdafx.h"
#include "getparams.h"
#include "util.h"
#include <map>

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"


using namespace cv;

typedef map<string, int>::iterator it_usr;
typedef map<int, int>::iterator it_idx;


class IMGstruct {
private:
	vector<string> Compat; // checks that the loaded images meet extensions specs 
	vector<string> ImgList_L, ImgList_R; // list of images

	// save/load names
	string leftname, rightname;
	string leftidx, rightidx, leftusr, rightusr;

	int getimagelist(string directory); // get names of images to load based on dataset
	drwnPersistentStorage storage;

	bool configlock; // configuration lock to ensure parameters are set

	// params
	string ld_dir, st_dir;
	string left_right, dataset;
	int height, width, subset_num;
	bool getnewimgs;

	// index 
	void saveindices(int left_or_right);
	void loadindices(int left_or_right);

public:
	IMGstruct();
	~IMGstruct();

	vector<Mat> rawL, rawR; // vector of loaded images

	map<int,int> userindex_L, userindex_R; // vector of which user refers to each image in the lists
	map<string,int> userL, userR; // list of user IDs

	void configure(RunOptions InputParams);

	void loadimages(); // load images off of image list and resize (individually from file or from binary)

};