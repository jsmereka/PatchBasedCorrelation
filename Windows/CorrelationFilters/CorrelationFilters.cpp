// CorrelationFilters.cpp : Defines the exported functions for the DLL application.
//

#include "CorrelationFilters.h"

/**
  
  Trains the filter as an OTSDF

  @author   Jonathon M. Smereka
  @version  04-13-2013
  
  @return   nothing

*/
template <class T>
void OTSDF<T>::trainfilter() {
	if(this->auth_count != 0 || this->imp_count != 0) {
		int d = input_row*input_col;

		// ONV
		if(alpha != 0) {
			//
		}
		// ACE
		if(beta != 0) {
			//
		}
		// ASM
		if(gamma != 0) {
			//
		}

		

	}
}
