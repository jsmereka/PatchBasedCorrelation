// util.h : class and functions for use throughout program
//

#include "stdafx.h"

// c++ standard headers
#include <dirent.h>
#include <direct.h>
#include <iomanip>

// darwin library headers
#include "drwnBase.h"
#include "drwnIO.h"
#include "drwnVision.h"

using namespace cv;


string intToString(int x);

bool checkdirectory(string dir);


class MatRecord {
private:

	bool saveMat(ofstream& out, const Mat& M);
	bool readMat(ifstream& in, Mat& M);

public:
	MatRecord() { }
	~MatRecord() { data.clear(); }

	// number of bytes needed to represent the data on disk include length field
	size_t numBytesOnDisk() const { return data.size() * sizeof(Mat) + sizeof(size_t); }

	bool write(const string filename); // write the data to disk
	bool read(const string filename); // read the data from disk

	vector<Mat> data;
};

