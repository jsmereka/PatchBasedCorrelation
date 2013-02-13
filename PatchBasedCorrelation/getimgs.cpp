

#include "getimgs.h"
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;


// Constructor
IMGstruct::IMGstruct() {

	configlock = true;

	ImgList_L.clear(); ImgList_R.clear();

	/* 
	Image file formats supported by OpenCV:
	BMP, DIB, JPEG, JPG, JPE, JP2, PNG, PBM, PGM, PPM, SR, RAS, TIFF, TIF
	*/
	Compat.clear(); // backward to make it a bit faster to check
	Compat.push_back("pmb");  Compat.push_back("bid");  Compat.push_back("gepj"); 
	Compat.push_back("gpj");  Compat.push_back("epj");  Compat.push_back("gnp");
	Compat.push_back("mbp");  Compat.push_back("mgp");  Compat.push_back("mpp");  
	Compat.push_back("rs");   Compat.push_back("sar");  Compat.push_back("ffit");
	Compat.push_back("fit");  Compat.push_back("2pj");
	/*
	Compat.push_back("bmp");  Compat.push_back("dib");  Compat.push_back("jpeg"); 
	Compat.push_back("jpg");  Compat.push_back("jpe");  Compat.push_back("png");
	Compat.push_back("pbm");  Compat.push_back("pgm");  Compat.push_back("ppm");  
	Compat.push_back("sr");   Compat.push_back("ras");  Compat.push_back("tiff");
	Compat.push_back("tif");  Compat.push_back("jp2");
	*/

}


// Destructor
IMGstruct::~IMGstruct() {

	if(!Compat.empty()) {
		Compat.clear();
	}
	DRWN_LOG_MESSAGE("Removing labels and user mappings from memory");
	if(!ImgList_L.empty()) {
		ImgList_L.clear();
	}
	if(!ImgList_R.empty()) {
		ImgList_R.clear();
	}
	if(!userindex_L.empty()) {
		userindex_L.clear();
	}
	if(!userindex_R.empty()) {
		userindex_R.clear();
	}
	if(!userL.empty()) {
		userL.clear();
	}
	if(!userR.empty()) {
		userR.clear();
	}
	DRWN_LOG_MESSAGE("Removing images from memory");
	if(!rawL.empty()) {
		rawL.clear();
	}
	if(!rawR.empty()) {
		rawR.clear();
	}

}


void IMGstruct::configure(RunOptions InputParams) {

	ld_dir = InputParams.ld_dir;
	st_dir = InputParams.st_dir;
	getnewimgs = InputParams.get_images;
	dataset = InputParams.dataset;
	subset_num = InputParams.subset_num;
	left_right = InputParams.left_right;
	height = InputParams.height;
	width = InputParams.width;

	if(ld_dir.back() != '/')
		ld_dir.push_back('/');
	if(st_dir.back() != '/')
		st_dir.push_back('/');
	if(!checkdirectory(st_dir) || !checkdirectory(st_dir + dataset + "/") || !checkdirectory(st_dir + dataset + "/imgs/")) {
		DRWN_LOG_WARNING("Setting to load images from " << ld_dir);
		getnewimgs = true;
	}

	stringstream name, name2, name3;
	name.clear();
	if(subset_num > 0) {
		checkdirectory(st_dir + dataset + "/imgs/sub" + intToString(subset_num));
		name << st_dir << dataset << "/imgs/sub" << subset_num << "/ImgsL_" << width << "x" << height;
		name2 << st_dir << dataset << "/imgs/sub" << subset_num << "/IndexL_" << width << "x" << height;
		name3 << st_dir << dataset << "/imgs/sub" << subset_num << "/UsrL_" << width << "x" << height;
	} else {
		name << st_dir << dataset << "/imgs/" << "ImgsL_" << width << "x" << height;
		name2 << st_dir << dataset << "/imgs/" << "IndexL_" << width << "x" << height;
		name3 << st_dir << dataset << "/imgs/" << "UsrL_" << width << "x" << height;
	}
	leftname = name.str();
	leftidx = name2.str();
	leftusr = name3.str();

	name.clear();
	if(subset_num > 0) {
		name << st_dir  << dataset << "/imgs/sub" << subset_num << "/ImgsR_" << width << "x" << height;
		name2 << st_dir << dataset << "/imgs/sub" << subset_num << "/IndexR_" << width << "x" << height;
		name3 << st_dir << dataset << "/imgs/sub" << subset_num << "/UsrR_" << width << "x" << height;
	} else {
		name << st_dir << dataset << "/imgs/" << "ImgsR_" << width << "x" << height;
		name2 << st_dir << dataset << "/imgs/" << "IndexR_" << width << "x" << height;
		name3 << st_dir << dataset << "/imgs/" << "UsrR_" << width << "x" << height;
	}
	rightname = name.str();
	rightidx = name2.str();
	rightusr = name3.str();

	configlock = false;

}


void IMGstruct::saveindices(int left_or_right) {
	ofstream os;
	if(left_or_right == 0) { // left
		os.open(leftusr.c_str(), ios::out|ios::binary);
		if(!os || (os.rdstate() & ifstream::failbit) != 0) {
			DRWN_LOG_WARNING("error opening file: " << leftusr);
		} else {
			for(it_usr iterator = userL.begin(); iterator != userL.end(); iterator++) {
				// iterator->first = key, iterator->second = value
				os << iterator->first;
				os << " ; ";
				os << iterator->second;
				os << " ; ";
				//os.write((char*)&iterator->second,sizeof(iterator->second));
			}
			DRWN_LOG_MESSAGE("saved: " << leftusr);
		}
		os.close();
		os.open(leftidx.c_str(), ios::out|ios::binary);
		if(!os || (os.rdstate() & ifstream::failbit) != 0) {
			DRWN_LOG_WARNING("error opening file: " << leftidx);
		} else {
			for(it_idx iterator = userindex_L.begin(); iterator != userindex_L.end(); iterator++) {
				// iterator->first = key, iterator->second = value
				os << iterator->first;
				os << " ; ";
				os << iterator->second;
				os << " ; ";
				//os.write((char*)&iterator->first,sizeof(iterator->first));
				//os.write((char*)&iterator->second,sizeof(iterator->second));
			}
			DRWN_LOG_MESSAGE("saved: " << leftidx);
		}
		os.close();
	} else if(left_or_right == 1) { // right
		os.open(rightusr.c_str(), ios::out|ios::binary);
		if(!os || (os.rdstate() & ifstream::failbit) != 0) {
			DRWN_LOG_WARNING("error opening file: " << rightusr);
		} else {
			for(it_usr iterator = userR.begin(); iterator != userR.end(); iterator++) {
				// iterator->first = key, iterator->second = value
				os.write((char*)&iterator->first,sizeof(iterator->first));
				os.write((char*)&iterator->second,sizeof(iterator->second));
			}
			DRWN_LOG_MESSAGE("saved: " << rightusr);
		}
		os.close();
		os.open(rightidx.c_str(), ios::out|ios::binary);
		if(!os || (os.rdstate() & ifstream::failbit) != 0) {
			DRWN_LOG_WARNING("error opening file: " << rightidx);
		} else {
			for(it_idx iterator = userindex_R.begin(); iterator != userindex_R.end(); iterator++) {
				// iterator->first = key, iterator->second = value
				os.write((char*)&iterator->first,sizeof(iterator->first));
				os.write((char*)&iterator->second,sizeof(iterator->second));
			}	
			DRWN_LOG_MESSAGE("saved: " << rightidx);
		}
		os.close();		
	}
}


void IMGstruct::loadindices(int left_or_right) {
	char c, colon; string tmp; int tmp2, tmp3;
	ifstream is;
	if(left_or_right == 0) { // left
		is.open(leftusr.c_str(), ios::in|ios::binary);
		if(!is || (is.rdstate() & ifstream::failbit) != 0) {
			DRWN_LOG_WARNING("error opening file: " << leftusr);
		} else {
			c = is.peek();
			while(!is.eof()) {
				is >> tmp;
				is >> colon;
				is >> tmp2;
				is >> colon;
				//is.read((char*)&tmp2,sizeof(tmp2));
				userL[tmp] = tmp2; 
				c = is.peek();
			}
			DRWN_LOG_MESSAGE("loaded: " << leftusr);
		}
		is.close();
		is.open(leftidx.c_str(), ios::in|ios::binary);
		if(!is || (is.rdstate() & ifstream::failbit) != 0) {
			DRWN_LOG_WARNING("error opening file: " << leftidx);
		} else {
			c = is.peek();
			while(!is.eof()) {
				//is.read((char*)&tmp2,sizeof(tmp2));
				//is.read((char*)&tmp3,sizeof(tmp3));
				is >> tmp2;
				is >> colon;
				is >> tmp3;
				is >> colon;
				userindex_L[tmp2] = tmp3;
				c = is.peek();
			}
			DRWN_LOG_MESSAGE("loaded: " << leftidx);
		}
		is.close();
	} else if(left_or_right == 1) { // right
		is.open(rightusr.c_str(), ios::in|ios::binary);
		if(!is || (is.rdstate() & ifstream::failbit) != 0) {
			DRWN_LOG_WARNING("error opening file: " << rightusr);
		} else {
			c = is.peek();
			while(!is.eof()) {
				is.read((char*)&tmp,sizeof(tmp));
				is.read((char*)&tmp2,sizeof(tmp2));
				userR[tmp] = tmp2; 
				c = is.peek();
			}
			DRWN_LOG_MESSAGE("loaded: " << rightusr);
		}
		is.close();
		is.open(rightidx.c_str(), ios::in|ios::binary);
		if(!is || (is.rdstate() & ifstream::failbit) != 0) {
			DRWN_LOG_WARNING("error opening file: " << rightidx);
		} else {
			c = is.peek();
			while(!is.eof()) {
				is.read((char*)&tmp2,sizeof(tmp2));
				is.read((char*)&tmp3,sizeof(tmp3));
				userindex_R[tmp2] = tmp3;
				c = is.peek();
			}
			DRWN_LOG_MESSAGE("loaded: " << rightidx);
		}
		is.close();
	}
}


int IMGstruct::getimagelist(string directory) {	
	DIR *d;
	struct dirent *dir;   // file pointer within directory
	d = opendir(directory.c_str());
	if(!d) {
		return -1;
	}
	int i, j, k, ct, ctL, ctR, usrLct, usrRct; 
	ct = 0; ctL = 0; ctR = 0; usrLct = 0; usrRct = 0;
	string imgChk, name;
	while((dir = readdir(d)) != NULL) { // not empty
		k = dir->d_namlen;
		if(k > 2 && dir->d_type == DT_REG) { // not a folder or '.', '..'
			j = 0; imgChk.clear(); name.clear();
			name.assign(dir->d_name);
			for(i=k-1; i>0; i--) { // get file extention (backward)
				if(name[i] == '.') {
					break;
				} else {
					imgChk.push_back(tolower(name[i]));
				}
			}
			for(i=0; i<Compat.size(); i++) { // compare with compatibility list
				if(imgChk.compare(Compat[i]) == 0) {
					break;
				}
			}
			if(i < Compat.size()) { // image is of compatible format

				// sort left from right
				// FOCS & FRGC && BDCP
				if(name.compare(name.size() - imgChk.size() - 2, 1, "l") == 0 || name.compare(name.size() - imgChk.size() - 2, 1, "L") == 0 
					|| name.compare(name.size() - imgChk.size() - 3, 1, "l") == 0 || name.compare(name.size() - imgChk.size() - 3, 1, "L") == 0) { // left
						ImgList_L.push_back(name);
						// map user id string to user #
						if(userL.empty()) {
							userL[name.substr(0,5)] = usrLct; usrLct++;
						} else {
							if(userL.find(name.substr(0,5)) == userL.end()) { // new user
								userL[name.substr(0,5)] = usrLct; usrLct++;
							}
						}
						// map loaded image # to user #
						userindex_L[ctL] = userL[name.substr(0,5)];	ctL++;
				} else if(name.compare(name.size() - imgChk.size() - 2, 1, "r") == 0 || name.compare(name.size() - imgChk.size() - 2, 1, "R") == 0
					|| name.compare(name.size() - imgChk.size() - 3, 1, "r") == 0 || name.compare(name.size() - imgChk.size() - 3, 1, "R") == 0) { // right
						ImgList_R.push_back(name);
						// map user id string to user #
						if(userR.empty()) {
							userR[name.substr(0,5)] = usrRct; usrRct++;
						} else {
							if(userR.find(name.substr(0,5)) == userR.end()) { // new user
								userR[name.substr(0,5)] = usrRct; usrRct++;
							}
						}
						// map loaded image # to user #
						userindex_R[ctR] = userR[name.substr(0,5)];	ctR++;
				}

				ct++;
			}
		}
	}
	closedir(d);
	return ct;
}


void IMGstruct::loadimages() {
	if(!configlock) {
		MatRecord *rec = new MatRecord;
		if(getnewimgs) {
			bool loadleft = false, loadright = false, errorflag = false;
			string loadname;
			// load images individually from file
			int ct = getimagelist(ld_dir);
			if(ct == 0) {
				DRWN_LOG_FATAL("no valid images found in " << ld_dir);
			} else if(ct == -1) {
				DRWN_LOG_FATAL("unable to open/find " << ld_dir);
			}
			if(userindex_L.size() + userindex_R.size() != ct) {
				DRWN_LOG_FATAL("was not able to properly parse all images as either the left or right ocular region - labels unavailable");
			}
			DRWN_LOG_MESSAGE("Directory and labels successfully parsed");
			if(left_right == "left") {
				if(ImgList_L.size() == 0) {
					DRWN_LOG_FATAL("no valid left ocular images found in " << ld_dir);
				} else {
					loadleft = true;
				}
			} else if(left_right == "right") {
				if(ImgList_R.size() == 0) {
					DRWN_LOG_FATAL("no valid right ocular images found in " << ld_dir);
				} else {
					loadright = true;
				}
			} else if(left_right == "both") {
				if(ImgList_L.size() == 0) {
					DRWN_LOG_WARNING("no valid left ocular images found in " << ld_dir);
				} else {
					loadleft = true;
				}
				if(ImgList_R.size() == 0) {
					DRWN_LOG_WARNING("no valid right ocular images found in " << ld_dir);
				} else {
					loadright = true;
				}
			}
			Mat tmpimg, resizedimg;
			if(loadleft) {
				cout << "Loading left ocular region\n";
				for(int i=0; i<ImgList_L.size(); i++) {
					loadname = ld_dir + ImgList_L[i];
					cout << "Image " << i << ":";
					tmpimg = imread(loadname.c_str(),0);
					if(!tmpimg.data){
						DRWN_LOG_WARNING("image data did not load properly for " << loadname);
						errorflag = true;
					} else {
						cout << " Loaded";
						if(tmpimg.rows != height || tmpimg.cols != width) {
							resize(tmpimg, resizedimg, Size(width,height), 0, 0, INTER_CUBIC);
							cout << ", Resized (" << tmpimg.rows << "x" << tmpimg.cols << "->" << height << "x" << width << ")";
							rawL.push_back(resizedimg);
						} else {
							cout << ", No Resizing";
							rawL.push_back(tmpimg);
						}
						cout << ", Complete\n";
					}
				}
			}
			if(!errorflag && loadleft) {
				DRWN_LOG_MESSAGE("All left ocular images loaded successfully - saving to binary");		
				// save
				rec->data.clear();
				rec->data = rawL;
				rec->write(leftname);
				saveindices(0);
			} else if(errorflag && loadleft) {
				DRWN_LOG_WARNING("There were errors in loading the left ocular images - not saving to binary");
			}
			errorflag = false;

			if(loadright) {
				cout << "Loading right ocular region\n";
				for(int i=0; i<ImgList_R.size(); i++) {
					loadname = ld_dir + ImgList_R[i];
					cout << "Image " << i << ":";
					tmpimg = imread(loadname.c_str(),0);
					if(!tmpimg.data){
						DRWN_LOG_WARNING("image data did not load properly for " << loadname);
						errorflag = true;
					} else { 
						cout << " Loaded";
						if(tmpimg.rows != height || tmpimg.cols != width) {
							resize(tmpimg, resizedimg, Size(width,height), 0, 0, INTER_CUBIC);
							cout << ", Resized (" << tmpimg.rows << "x" << tmpimg.cols << "->" << height << "x" << width << ")";
							rawR.push_back(resizedimg);
						} else {
							cout << ", No Resizing";
							rawR.push_back(tmpimg);
						}
						cout << ", Complete\n";
					}
				}
			}
			if(!errorflag && loadright) {
				DRWN_LOG_MESSAGE("All right ocular images loaded successfully - saving to binary");
				// save
				rec->data.clear();
				rec->data = rawR;
				rec->write(rightname);
				saveindices(1);
			} else if(errorflag && loadright) {
				DRWN_LOG_WARNING("There were errors in loading the right ocular images - not saving to binary");
			}
			errorflag = false;
		} else {
			// load up binary
			if(left_right == "left" || left_right == "both") {
				if(rec->read(leftname)) {
					rawL = rec->data;
					loadindices(0);
				} else {
					rawL.clear();
				}
				rec->data.clear();
			}
			if(left_right == "right" || left_right == "both") {
				if(rec->read(rightname)) {
					rawR = rec->data;
					loadindices(1);
				} else {
					rawR.clear();
				}
				rec->data.clear();
			}
		}
	} else {
		DRWN_LOG_WARNING("Cannot load images until parameters are configured");
	}
}

