// load up an image, build a filter, and apply correlation for result
#include <iostream>
#include <vector>
#include <string>
#include <dirent.h>

#include "cfs.h"
#include "otsdf.h"
#include "uotsdf.h"
#include <Eigen/Core>
#include <Eigen/Dense>

#include <cv.h>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#define cvtype CV_64F

// simple containers to label authentic and impostor images (they come in unordered)
std::vector<int> authimg;
std::vector<int> impimg;


// type conversion from darwin framework edited to work properly (opencv is row-major, while eigen is column-major)
Eigen::MatrixXd cvMat2eigen(const CvMat *m) {
	// TODO: make this work for any scalar type
	Eigen::MatrixXd d;

	if(m == NULL){ return d; }

	d.resize(m->rows, m->cols);

	switch (cvGetElemType(m)) {
	case CV_8UC1:
	{
		int ct = 0;
		const unsigned char *p = (unsigned char *)CV_MAT_ELEM_PTR(*m, 0, 0);
		for(int i = 0; i < m->rows; i++) {
			for(int j = 0; j < m->cols; j++) {
				d(i,j) = (double)p[ct]; ct++;
			}
		}
	}
	break;

	case CV_8SC1:
	{
		int ct = 0;
		const char *p = (char *)CV_MAT_ELEM_PTR(*m, 0, 0);
		for(int i = 0; i < m->rows; i++) {
			for(int j = 0; j < m->cols; j++) {
				d(i,j) = (double)p[ct]; ct++;
			}
		}
	}
	break;

	case CV_32SC1:
	{
		Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp;
		temp = Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((int *)m->data.ptr, m->rows, m->cols);
		d = temp.cast<double>();
	}
	break;

	case CV_32FC1:
	{
		Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp;
		temp = Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((float *)m->data.ptr, m->rows, m->cols);
		d = temp.cast<double>();
	}
	break;

	case CV_64FC1:
	{
		Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> temp;
		temp = Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((double *)m->data.ptr, m->rows, m->cols);
		d = temp.cast<double>();
	}
	break;

	default:
		std::cout << "unrecognized openCV matrix type: " << cvGetElemType(m) << "\n";
		break;
	}

	return d;
}

// type conversion from darwin framework
CvMat *eigen2cvMat(const Eigen::MatrixXd &m, int mType) {
	// TODO: make this work for any scalar type
	CvMat *d = cvCreateMat(m.rows(), m.cols(), mType);
	if(d == NULL) return NULL;

	switch (mType) {
	case CV_8UC1:
	case CV_8SC1:
		std::cout << "not implemented yet\n";
		break;

	case CV_32SC1:
	{
		Eigen::MatrixXi temp = m.cast<int>();
		Eigen::Map<Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((int *)d->data.ptr, d->rows, d->cols) = temp;
	}
	break;

	case CV_32FC1:
	{
		Eigen::MatrixXf temp = m.cast<float>();
		Eigen::Map<Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((float *)d->data.ptr, d->rows, d->cols) = temp;
	}
	break;

	case CV_64FC1:
	{
		Eigen::MatrixXd temp = m.cast<double>();
		Eigen::Map<Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor> >((double *)d->data.ptr, d->rows, d->cols) = temp;
	}
	break;

	default:
		std::cout << "unrecognized openCV matrix type: " << mType << "\n";
		break;
	}

	return d;
}



void EigShowImg(const Eigen::MatrixXd &mat) {
	// TODO: make this work for any scalar type
	CvMat *convert = eigen2cvMat(mat, cvtype);
	cv::Mat image = convert;
	image.convertTo(image, CV_8U);
	cv::namedWindow("Display window", CV_WINDOW_AUTOSIZE); // Create a window for display.
	cv::imshow("Display window", image);                   // Show our image inside it.
	cv::waitKey(0);
}



void loadimages(std::vector<Eigen::MatrixXd> &imgs, std::string directory, int width, int height) {
	std::vector<std::string> Compat; // checks that the loaded images meet extensions specs
	/* Image file formats supported by OpenCV: BMP, DIB, JPEG, JPG, JPE, JP2, PNG, PBM, PGM, PPM, SR, RAS, TIFF, TIF */
	Compat.clear(); // backward to make it a bit faster to check
	Compat.push_back("pmb");  Compat.push_back("bid");  Compat.push_back("gepj");
	Compat.push_back("gpj");  Compat.push_back("epj");  Compat.push_back("gnp");
	Compat.push_back("mbp");  Compat.push_back("mgp");  Compat.push_back("mpp");
	Compat.push_back("rs");   Compat.push_back("sar");  Compat.push_back("ffit");
	Compat.push_back("fit");  Compat.push_back("2pj");
	DIR *d; struct dirent *dir;   // file pointer within directory
	d = opendir(directory.c_str());
	if(!d) {
		return;
	}
	int i, k, ct=0, ct2=0;
	std::string imgChk, name;
	std::string loadname;
	cv::Mat tmpimg, resizedimg;
	while((dir = readdir(d)) != NULL) { // not empty
		k = dir->d_reclen;
		if(k > 2 && dir->d_type == DT_REG) { // not a folder or '.', '..'
			imgChk.clear(); name.clear();
			name.assign(dir->d_name);
			for(i=name.size()-1; i>0; i--) { // get file extention (backward)
				if(!name.compare(i,1,".")) {
					break;
				} else {
					imgChk.push_back(tolower(name[i]));
				}
			}
			for(i=0; i<(int)(Compat.size()); i++) { // compare with compatibility list
				if(imgChk.compare(Compat[i]) == 0) {
					break;
				}
			}
			if(i < (int)(Compat.size())) { // image is of compatible format
				ct++;
				loadname = directory + name;
				std::cout << "Image " << ct << ":";
				tmpimg = cv::imread(loadname.c_str(), 0); // force to be grayscale
				if(!tmpimg.data){
					std::cout << "image data did not load properly for " << loadname << ", ";
				} else {
					Eigen::MatrixXd newmat; CvMat convert;
					std::cout << " Loaded " << name;

					if(!name.compare(0,4,"auth")) {
						// authentic image
						authimg.push_back(ct2);
					} else if(!name.compare(0,3,"imp")) {
						impimg.push_back(ct2);
					}

					if(tmpimg.rows != height || tmpimg.cols != width) {
						resize(tmpimg, resizedimg, cv::Size(width,height), 0, 0, cv::INTER_CUBIC);
						std::cout << ", Resized (" << tmpimg.rows << "x" << tmpimg.cols << "->" << height << "x" << width << ")";
						resizedimg.convertTo(resizedimg, cvtype);
						convert = resizedimg;
					} else {
						std::cout << ", No Resizing";
						tmpimg.convertTo(tmpimg, cvtype);
						convert = tmpimg;
					}
					newmat = cvMat2eigen(&convert);
					imgs.push_back(newmat);
					std::cout << ", Complete\n"; ct2++;
				}
			} else {
				std::cout << "Not Loading: " << name << "\n";
			}
		}
	}
	std::cout << "Loaded a total of " << ct << " images\n";
	closedir(d);
}


int main(int argc, char *argv[]) {

	std::vector<Eigen::MatrixXd> imgs; imgs.clear();

	// get image from directory
	if(argc > 1) {
		loadimages(imgs, argv[1], 92, 112);
		//loadimages(imgs, argv[1], 30, 40);
	}
	bool runtests = true;

	if(imgs.size() > 0 && authimg.size() > 0) {
		bool nontested = runtests; unsigned int i = 0;
		// build filters
		UOTSDF<Eigen::MatrixXd> thefilter(pow(10,-5), 1-pow(10,-5), 0.0); // matrix of doubles
		thefilter.computerank(true); // is defaulted as true, but we can set it anyway
		thefilter.whitenspectrum(0); // use all samples to whiten spectrum, 0 = off (default), 1 = all samples, 2 = only authentic samples, 3 = only impostor samples
		thefilter.zeropadtrndata(true); // default = false (to save some memory)
		thefilter.adjustfromcenter(false); // default = false, but will work either way as long as it's consistent for each sample

		// save one auth img and one imp image for comparison

		for(unsigned int j=0; j<authimg.size()-1; j++) { // use all but one to train
			i = authimg[j];

			if(imgs[i].SizeMinusOne != 0) {

				if(nontested) {
					/* 2D */
					std::cout << "TESTING BASIC FUNCTIONALITY (2D):\n";
					///// Basic Tests
					{
						// Successfully add first image to the filter
						std::cout << "\tAdds image: ";
						if(thefilter.add_auth(imgs[i])) {
							std::cout << "Success\n";
						} else {
							std::cout << "Failure!\n";
						}
						// Fail to add the same image again to the filter
						std::cout << "\tRejects the same image (authentic): ";
						if(thefilter.add_auth(imgs[i])) {
							std::cout << "Failure!\n";
						} else {
							std::cout << "Success\n";
						}
						// Fail to add the same image again to the filter as an impostor
						std::cout << "\tRejects the same image (impostor): ";
						if(thefilter.add_imp(imgs[i])) {
							std::cout << "Failure!\n";
						} else {
							std::cout << "Success\n";
						}
					}

					///// Resize original image and add to the filter
					{
						// pad image on right and bottom sides
						Eigen::MatrixXd resizedimg = imgs[i];
						resizedimg.conservativeResize(imgs[i].rows()+20, imgs[i].cols()+40);
						thefilter.adjustfromcenter(false);
						// fail to add the image
						std::cout << "\tRejects same image after being padded (non-centered): ";
						if(thefilter.add_auth(resizedimg)) {
							std::cout << "Failure!\n";
						} else {
							std::cout << "Success\n";
						}
						// pad image on all sides
						resizedimg.setZero(imgs[i].rows()+20, imgs[i].cols()+40);
						resizedimg.block(10,20,imgs[i].rows(),imgs[i].cols()) = imgs[i];
						thefilter.adjustfromcenter(true);
						// fail to add the image
						std::cout << "\tRejects same image after being padded (centered): ";
						if(thefilter.add_auth(resizedimg)) {
							std::cout << "Failure!\n";
						} else {
							std::cout << "Success\n";
						}
						thefilter.adjustfromcenter(false);
					}
					//EigShowImg(imgs[i]);
					nontested = false; // only need to run test once
				} else {
					thefilter.add_auth(imgs[i]);
				}
			} // endif
		} // endfor

		if(impimg.size() > 0) {
			for(unsigned int j=0; j<impimg.size()-1; j++) { // use all but one to train
				i = impimg[j];
				if(imgs[i].SizeMinusOne != 0) {
					thefilter.add_imp(imgs[i]);
				}
			} // endfor
		}

		// train
		std::cout << "\tTraining the Filter\n";
		thefilter.trainfilter();

		// apply
		std::cout << "\tApplying the filter against trained authentic: ";
		Eigen::MatrixXd plane = thefilter.applyfilter(imgs[authimg.front()]);
		Eigen::MatrixXd::Index maxRow, maxCol; //get location of maximum
		double pmax = plane.maxCoeff(&maxRow, &maxCol);
		std::cout << "Max: " << pmax << ", at: " << maxRow << "," << maxCol << "\n";

		std::cout << "\tApplying the filter against trained impostor: ";
		plane = thefilter.applyfilter(imgs[impimg.front()]);
		pmax = plane.maxCoeff(&maxRow, &maxCol);
		std::cout << "Max: " << pmax << ", at: " << maxRow << "," << maxCol << "\n";

		std::cout << "\tApplying the filter against untrained authentic: ";
		plane = thefilter.applyfilter(imgs[authimg.back()]);
		pmax = plane.maxCoeff(&maxRow, &maxCol);
		std::cout << "Max: " << pmax << ", at: " << maxRow << "," << maxCol << "\n";

		std::cout << "\tApplying the filter against untrained impostor: ";
		plane = thefilter.applyfilter(imgs[impimg.back()]);
		pmax = plane.maxCoeff(&maxRow, &maxCol);
		std::cout << "Max: " << pmax << ", at: " << maxRow << "," << maxCol << "\n";


	}
	// Not limited to RowVector, testing just cause...
	OTSDF<Eigen::RowVectorXf> thefiltervec(pow(10,-5), 1-pow(10,-5), 0.0); // vector of floats

	Eigen::RowVectorXf truesig(15); truesig << 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0;

	/* 1D */
	thefiltervec.computerank(true); // is defaulted as true, but we can set it anyway

	if(runtests) {
		std::cout << "TESTING BASIC FUNCTIONALITY (1D):\n";
		///// Basic Tests
		{
			// Successfully add first image to the filter
			std::cout << "\tAdds signal: ";
			if(thefiltervec.add_auth(truesig)) {
				std::cout << "Success\n";
			} else {
				std::cout << "Failure!\n";
			}
			// Fail to add the same image again to the filter
			std::cout << "\tRejects the same signal (authentic): ";
			if(thefiltervec.add_auth(truesig)) {
				std::cout << "Failure!\n";
			} else {
				std::cout << "Success\n";
			}
			// Fail to add the same image again to the filter as an impostor
			std::cout << "\tRejects the same signal (impostor): ";
			if(thefiltervec.add_imp(truesig)) {
				std::cout << "Failure!\n";
			} else {
				std::cout << "Success\n";
			}
		}

		///// Resize original image and add to the filter
		{
			// pad image on right and bottom sides
			Eigen::RowVectorXf resizedvec = truesig;
			resizedvec.conservativeResize(truesig.cols()+6); // added data is garbage, but it doesn't matter cause it gets cut out anyway
			thefiltervec.adjustfromcenter(false);
			// fail to add the image
			std::cout << "\tRejects same signal after being padded (non-centered): ";
			if(thefiltervec.add_auth(resizedvec)) {
				std::cout << "Failure!\n";
			} else {
				std::cout << "Success\n";
			}
			// pad image on all sides
			resizedvec.setZero(truesig.cols()+6);
			resizedvec.block(0,3,1,truesig.cols()) = truesig;
			thefiltervec.adjustfromcenter(true);
			// fail to add the image
			std::cout << "\tRejects same signal after being padded (centered): ";
			if(thefiltervec.add_auth(resizedvec)) {
				std::cout << "Failure!\n";
			} else {
				std::cout << "Success\n";
			}
			thefiltervec.adjustfromcenter(false);
		}
	} else {
		thefiltervec.add_auth(truesig);
	}

	truesig << 1, 0, 1, 0, 0, 1, 1, 1, 1, 1, 0, 0, 1, 0, 1;
	thefiltervec.add_auth(truesig);

	truesig << 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1;
	thefiltervec.add_imp(truesig);

	thefiltervec.trainfilter();

	truesig.resize(20);

	// apply against authentic
	truesig << 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0;
	std::cout << "\tApply against authentic: ";
	Eigen::RowVectorXf plane = thefiltervec.applyfilter(truesig);
	//get location of maximum
	Eigen::RowVectorXf::Index maxRow, maxCol;
	float pmax = plane.maxCoeff(&maxRow, &maxCol);
	std::cout << "Max: " << pmax << ", at: " << maxRow << "," << maxCol << "\n";

	// apply against impostor
	truesig << 0, 1, 0, 1, 1, 0, 0.5, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0.5, 0.5, 0.75, 0;
	std::cout << "\tApply against impostor: ";
	plane = thefiltervec.applyfilter(truesig);
	pmax = plane.maxCoeff(&maxRow, &maxCol);
	std::cout << "Max: " << pmax << ", at: " << maxRow << "," << maxCol << "\n";

}
