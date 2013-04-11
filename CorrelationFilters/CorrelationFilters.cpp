// CorrelationFilters.cpp : Defines the exported functions for the DLL application.
//

#include "stdafx.h"

template <class T>
class OTSDF : public filter<T> {
	double alpha, beta;
	void trainfilter(){ };
};



