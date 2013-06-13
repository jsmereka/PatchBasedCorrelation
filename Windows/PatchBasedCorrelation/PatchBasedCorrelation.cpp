// PatchBasedCorrelation.cpp : Defines the entry point for the console application.
//

// project libraries
#include "stdafx.h"
#include "buildfilter.h"

static Parameters Params;



void usage()
{
    cerr << DRWN_USAGE_HEADER << endl;
    cerr << "USAGE: ./cmdline [OPTIONS]\n";
    cerr << "OPTIONS:\n"
         << "  -d <dataset>      :: string of dataset to load/run (default: " << Params.par.dataset << ")\n"
		 << "  -s <subset #>     :: integer of which subset to use (default: " << Params.subsetID() << ")\n"
		 << "  -seg	<style>      :: string of segmentation style (default: " << Params.par.segchoice << ")\n"
         << DRWN_STANDARD_OPTIONS_USAGE
         << endl;
}

int main(int argc, char* argv[])
{
	IMGstruct Imgs;
	FILTstruct Filters;

	/*******************/
	/* Load Parameters */
	/*******************/
    const char *dataset = NULL;
	const char *segstyle = NULL;
	int subset = 100;

	// getting command line parameters
    DRWN_BEGIN_CMDLINE_PROCESSING(argc, argv)
        DRWN_CMDLINE_STR_OPTION("-d", dataset)
		DRWN_CMDLINE_INT_OPTION("-s", subset)
		DRWN_CMDLINE_STR_OPTION("-seg", segstyle)
    DRWN_END_CMDLINE_PROCESSING(usage());

    if(dataset != NULL)
        Params.par.dataset = string(dataset);
	if(subset != 100)
		Params.par.subset_num = subset;
	if(segstyle != NULL)
		Params.par.segchoice = segstyle;

	Params.checkParameters();
	
	// transfer necessary parameters to each class
	Imgs.configure(Params.par);
	Filters.configure(Params.par);

	/*******************/
	/* Get/Load Images */
	/*******************/
	Imgs.loadimages();

	/*********************/
	/* Get/Build Filters */
	/*********************/
	Filters.getfilters();

	/*******************/
	/* Run Comparisons */
	/*******************/

	
	/***************************/
	/* Analyze & Store Results */
	/***************************/


    return 0;
}
