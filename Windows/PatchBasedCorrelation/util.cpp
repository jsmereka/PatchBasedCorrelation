// project libraries
#include "stdafx.h"
#include "util.h"


string intToString(int x) {
	string a;
	stringstream b;
	b << x;
	a = b.str();
	return a;
}



bool checkdirectory(string dir) {
	DIR *d;
	d = opendir(dir.c_str());
	if(!d){
		_mkdir(dir.c_str());
		d = opendir(dir.c_str());
		if(!d){
			DRWN_LOG_WARNING("unable to open: " << dir);
			return false;
		}
	}
	closedir(d);
	return true;
}



// Save matrix to binary file
bool MatRecord::saveMat(ofstream& out, const Mat& M){
	if(!out) {
		return false;
	}
	if(M.empty()) {
		return false;
	}

	int cols = M.cols;
	int rows = M.rows;
	int chan = M.channels();
	int eSiz = (M.dataend-M.datastart)/(cols*rows*chan);

	// Write header
	out.write((char*)&cols,sizeof(cols));
	out.write((char*)&rows,sizeof(rows));
	out.write((char*)&chan,sizeof(chan));
	out.write((char*)&eSiz,sizeof(eSiz));

	// Write data.
	if(M.isContinuous()) {
		out.write((char *)M.data,cols*rows*chan*eSiz);
	} else {
		return false;
	}
	return true;
}


bool MatRecord::write(const string filename) {
	ofstream os(filename.c_str(), ios::out|ios::binary);
	if(!os)
		return false;
	bool noerror;
	if((os.rdstate() & ifstream::failbit ) != 0) {
		DRWN_LOG_WARNING("error opening file: " << filename);
		return false;
	}
	for(int i=0; i<data.size(); i++) {
		noerror = saveMat(os,data.at(i));
		if(!noerror)
			break;
	}
	os.close();
	if(!noerror) {
		DRWN_LOG_WARNING("writing file returned with errors: " << filename);
	} else {
		DRWN_LOG_MESSAGE("saved: " << filename);
	}
	return noerror;
}



// Read matrix from binary file
bool MatRecord::readMat(ifstream& in, Mat& M) {
	if(!in) {
		return false;
	}
	if(in.eof()) {
		return false;
	}
	int cols;
	int rows;
	int chan;
	int eSiz;

	// Read header
	in.read((char*)&cols,sizeof(cols));
	in.read((char*)&rows,sizeof(rows));
	in.read((char*)&chan,sizeof(chan));
	in.read((char*)&eSiz,sizeof(eSiz));

	// Determine type of the matrix 
	int type = 0;
	switch (eSiz){
	case sizeof(char):
		type = CV_8UC(chan);
		break;
	case sizeof(float):
		type = CV_32FC(chan);
		break;
	case sizeof(double):
		type = CV_64FC(chan);
		break;
	}

	// Alocate Matrix.
	M = Mat(rows,cols,type,Scalar(1));  

	// Read data.
	if(M.isContinuous()) {
		in.read((char *)M.data,cols*rows*chan*eSiz);
	} else {
		return false;
	}
	return true;
}


bool MatRecord::read(const string filename) {
	ifstream is(filename.c_str(), ios::in|ios::binary);
	if(!is){
		data.clear();
		return false;
	}
	if((is.rdstate() & ifstream::failbit ) != 0) {
		DRWN_LOG_WARNING("error opening file: " << filename);
		return false;
	}
	bool noerror = true; Mat tmp;
	char c = is.peek();
	while(!is.eof() && noerror) {
		noerror = readMat(is,tmp); 
		if(noerror) {
			data.push_back(tmp);
		}
		c = is.peek();
	}
	is.close();
	if(!noerror) {
		DRWN_LOG_WARNING("reading file returned with errors: " << filename);
	} else {
		DRWN_LOG_MESSAGE("loaded: " << filename);
	}
	return noerror;
}