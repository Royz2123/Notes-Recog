// **********************************************************************************
//
// BSD License.
// This file is part of a Hough Transformation tutorial,
// see: http://www.keymolen.com/2013/05/hough-transformation-c-implementation.html
//
// Copyright (c) 2013, Bruno Keymolen, email: bruno.keymolen@gmail.com
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//
// Redistributions of source code must retain the above copyright notice,
// this list of conditions and the following disclaimer.
// Redistributions in binary form must reproduce the above copyright notice, this
// list of conditions and the following disclaimer in the documentation and/or other
// materials provided with the distribution.
// Neither the name of "Bruno Keymolen" nor the names of its contributors may be
// used to endorse or promote products derived from this software without specific
// prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
// ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED.
// IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
// INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT
// NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.
//
// **********************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <dirent.h>
#include <unistd.h>
#include <string>
#include <map>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "hough.h"

#define MAX_SLOPE 10
#define MIN_SLOPE 0.5

#define ROWS_SPACING 4

extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;

std::string img_path;
int threshold = 0;

const char* CW_IMG_ORIGINAL 	= "Result";
const char* CW_IMG_EDGE		= "Canny Edge Detection";
const char* CW_ACCUMULATOR  	= "Accumulator";

// define types
typedef std::pair<int, int> point;
typedef std::pair<point, point> line;
typedef std::pair<line, line> row;

// Declerations
void doTransform(std::string, int threshold);
float findSlope(point first, point second);
bool pairCompare(line firstLine, line secondLine);

// Usage explanation
void usage(char * s)
{
	fprintf( stderr, "\n");
  fprintf( stderr, "%s -s <source file> [-t <threshold>] - hough transform. build: %s-%s \n", s, __DATE__, __TIME__);
	fprintf( stderr, "   s: path image file\n");
	fprintf( stderr, "   t: hough threshold\n");
	fprintf( stderr, "\nexample:  ./hough -s ./img/russell-crowe-robin-hood-arrow.jpg -t 195\n");
	fprintf( stderr, "\n");
}

int main(int argc, char** argv) {
	int c;
	while ( ((c = getopt( argc, argv, "s:t:?" ) ) ) != -1 )
	{
	    switch (c)
	    {
	    case 's':
	    	img_path = optarg;
	    	break;
	    case 't':
	    	threshold = atoi(optarg);
	    	break;
	    case '?':
        default:
			usage(argv[0]);
			return -1;
	    }
	}

	if(img_path.empty())
	{
		usage(argv[0]);
		return -1;
	}

  cv::namedWindow(CW_IMG_ORIGINAL, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(CW_IMG_EDGE, 	 cv::WINDOW_AUTOSIZE);
  cv::namedWindow(CW_ACCUMULATOR,	 cv::WINDOW_AUTOSIZE);

  cvMoveWindow(CW_IMG_ORIGINAL, 10, 10);
  cvMoveWindow(CW_IMG_EDGE, 680, 10);
  cvMoveWindow(CW_ACCUMULATOR, 1350, 10);

  doTransform(img_path, threshold);
	return 0;
}

void doTransform(std::string file_path, int threshold)
{
	cv::Mat img_edge;
	cv::Mat img_blur;

	cv::Mat img_ori = cv::imread( file_path, 1 );
	cv::blur( img_ori, img_blur, cv::Size(5,5) );
	cv::Canny(img_blur, img_edge, 100, 150, 3);

	int w = img_edge.cols;
	int h = img_edge.rows;

	//Transform
	keymolen::Hough hough;
	hough.Transform(img_edge.data, w, h);

	if(threshold == 0)
		threshold = w>h?w/4:h/4;

	while(1)
	{
		cv::Mat img_res = img_ori.clone();
		std::vector<line> horizontal;
		std::vector<row> spaces;
		std::vector<row> rows;

		//Search the accumulator
		std::vector<line> lines = hough.GetLines(threshold);

		//Draw the results
		std::vector<line>::iterator it;
		for(it=lines.begin();it!=lines.end();it++)
		{
			// check if the line is horizontal or vertical
			// if not, dismiss it
			float lineSlope = findSlope(it->first, it->second);
			if (lineSlope > MAX_SLOPE || lineSlope < MIN_SLOPE) {
				// Visualize
/*
				cv::line(
					img_res,
					cv::Point(it->first.first, it->first.second),
					cv::Point(it->second.first, it->second.second),
					cv::Scalar( 0, 0, 255), 2, 8
				);
*/
				// add to our vector of horizontal lines
				if (lineSlope > MAX_SLOPE) {
					horizontal.push_back(std::make_pair(it->first, it->second));
				}
			}
		}

		// now we have all of the horizontal lines.
		// find the ones clustered together and declare them as a row
		std::sort(horizontal.begin(), horizontal.end(), pairCompare);

		// define row as two lines that have a large distance from their neighbours
		float prevDist = 0, currDist = 0;
		row currRow;
		int currLine = 2;

		// first find the large gaps
		for(it=horizontal.begin();it!=horizontal.end()-1;it++) {
			currDist = (it+1)->first.second - it->first.second;
			std::cout << currDist;
			if (currDist >= ROWS_SPACING*prevDist) {
				// start of row
				currRow.first = make_pair(it->first, it->second);
				// end of row
				currRow.second =  make_pair((it+1)->first, (it+1)->second);
				spaces.push_back(currRow);
			}
			prevDist = currDist;
		}

		// now "shift" the pairs we have created so they match the rows
		std::vector<row>::iterator it2;
		for(it2=spaces.begin()+1;it2!=spaces.end()-1;it2++) {
			currRow.first = it2->second;
			currRow.second = (it2+1)->first;
			rows.push_back(currRow);
		}

		for(it2=rows.begin();it2!=rows.end();it2++) {
			cv::line(
				img_res,
				cv::Point(it2->first.first.first, it2->first.first.second),
				cv::Point(it2->first.second.first, it2->first.second.second),
				cv::Scalar( 0, 0, 255), 2, 8
			);

			cv::line(
				img_res,
				cv::Point(it2->second.first.first, it2->second.first.second),
				cv::Point(it2->second.second.first, it2->second.second.second),
				cv::Scalar( 0, 0, 255), 2, 8
			);
		}

		//Visualize all
		int aw, ah, maxa;
		aw = ah = maxa = 0;

		const unsigned int* accu = hough.GetAccu(&aw, &ah);

		for(int p=0;p<(ah*aw);p++)
		{
			if((int)accu[p] > maxa)
				maxa = accu[p];
		}
		double contrast = 1.0;
		double coef = 255.0 / (double)maxa * contrast;

		cv::Mat img_accu(ah, aw, CV_8UC3);
		for(int p=0;p<(ah*aw);p++)
		{
			unsigned char c = (double)accu[p] * coef < 255.0 ? (double)accu[p] * coef : 255.0;
			img_accu.data[(p*3)+0] = 255;
			img_accu.data[(p*3)+1] = 255-c;
			img_accu.data[(p*3)+2] = 255-c;
		}

		cv::imshow(CW_IMG_ORIGINAL, img_res);
		cv::imshow(CW_IMG_EDGE, img_edge);
		cv::imshow(CW_ACCUMULATOR, img_accu);

		char c = cv::waitKey(360000);
		if(c == '+')
			threshold += 5;
		if(c == '-')
			threshold -= 5;
		if(c == 27)
			break;
	}
}


// Utility functions

// finds the slope of a line
float findSlope(point first, point second) {
	return abs(
		(float)(first.first - second.first)
		/ (float)(first.second - second.second)
	);
}

// Compares 2 horizontal lines
bool pairCompare(line firstLine, line secondLine) {
  return firstLine.first.second < secondLine.first.second;
}
