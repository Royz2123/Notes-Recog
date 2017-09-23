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
#include <sstream>
#include <map>
#include <iostream>

#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/core/core.hpp"

#include "hough.h"

#define MAX_SLOPE 10
#define MIN_SLOPE 0.5

#define ROWS_SPACING 4
#define COLUMNS_SPACING 10

#define LEFT_BORDER 20
#define RIGHT_BORDER 20

#define MIN_BAR_SIZE 20
#define TOP_BAR_PADDING 10

#define CROPPED_BARS_PATH "./cropped_bars/bar"

extern FILE *stdin;
extern FILE *stdout;
extern FILE *stderr;

std::stringstream out;
std::string img_path;
int threshold = 0;

const char* CW_IMG_CROP	= "crop";
const char* CW_IMG_ORIGINAL 	= "Result";
const char* CW_IMG_EDGE		= "Canny Edge Detection";
const char* CW_ACCUMULATOR  	= "Accumulator";

// define types
typedef std::pair<int, int> point;
typedef std::pair<point, point> line;
typedef std::pair<line, line> row;
typedef bool(*pairCompare)(line, line);

// Declerations
void doTransform(std::string, int threshold);
float findSlope(point first, point second);
bool horizCompare(line firstLine, line secondLine);
bool vertCompare(line firstLine, line secondLine);
void findSpaces(
	std::vector<line> lines,
	std::vector<row>& spaces,
	pairCompare comparer,
	int spacing
);
void findRows(std::vector<line> horizontal, std::vector<row>& rows);
void findColumns(
	std::vector<line> vertical,
	std::vector<row>& columns,
	int imageSize
);
void drawLine(cv::Mat img_res, line currLine);
void printRows(std::vector<row>& rows);

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

	cv::namedWindow(CW_IMG_CROP, cv::WINDOW_AUTOSIZE);
	cv::namedWindow(CW_IMG_ORIGINAL, cv::WINDOW_AUTOSIZE);
  cv::namedWindow(CW_IMG_EDGE, 	 cv::WINDOW_AUTOSIZE);
  cv::namedWindow(CW_ACCUMULATOR,	 cv::WINDOW_AUTOSIZE);

	cvMoveWindow(CW_IMG_CROP, 100, 100);
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
	cv::Mat img_crop;
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

		// Region of interest
		cv::Rect roi;
		float rectTopLeftX, rectTopLeftY, rectBottomLeftX, rectBottomLeftY;

		// line vectors
		std::vector<line> horizontal;
		std::vector<line> vertical;
		std::vector<row> rows;
		std::vector<row> columns;

		// Iterators
		std::vector<line>::iterator lineIt;
		std::vector<row>::iterator rowIt;
		std::vector<row>::iterator rowIt2;

		//Search the accumulator
		std::vector<line> lines = hough.GetLines(threshold);

		//Draw the results
		for(lineIt=lines.begin();lineIt!=lines.end();lineIt++)
		{
			// check if the line is horizontal or vertical
			// if not, dismiss it
			float lineSlope = findSlope(lineIt->first, lineIt->second);
			// add to our vector of horizontal lines
			if (lineSlope > MAX_SLOPE) {
				horizontal.push_back(std::make_pair(lineIt->first, lineIt->second));
			} else {
				vertical.push_back(std::make_pair(lineIt->first, lineIt->second));
			}
		}

		// now we have all of the horizontal lines.
		// find the ones clustered together and declare them as a row
		findRows(horizontal, rows);
		findColumns(vertical, columns, img_res.cols);

		// now crop based on each row
		int barIndex = 0;
		for(rowIt=rows.begin();rowIt!=rows.end();rowIt++) {
			for(rowIt2=columns.begin();rowIt2!=columns.end();rowIt2++) {
				// each iteration gives us a bar!
				rectTopLeftX = rowIt2->first.first.first;
				rectTopLeftY = rowIt->first.first.second;
				rectBottomLeftX = rowIt2->second.first.first;
				rectBottomLeftY = rowIt->second.first.second;

				roi.x = rectTopLeftX;
		    roi.y = rectTopLeftY - TOP_BAR_PADDING;
		    roi.width = rectBottomLeftX - rectTopLeftX;
		    roi.height = rectBottomLeftY - rectTopLeftY + 2*TOP_BAR_PADDING;

		    /* Crop the original image to the defined ROI */
				// save image
				img_crop = img_res(roi);
				out << barIndex;
				std::string path = CROPPED_BARS_PATH + out.str() + ".jpg";
				cv::imwrite(path, img_crop);
				out.str(std::string());
				barIndex++;
			}
		}

		//Draw the results:

		// Visualize all lines
		for(lineIt=lines.begin();lineIt!=lines.end();lineIt++)
		{
			// check if the line is horizontal or vertical
			// if not, dismiss it
			float lineSlope = findSlope(lineIt->first, lineIt->second);
			if (lineSlope > MAX_SLOPE || lineSlope < MIN_SLOPE) {
				// Visualize

				cv::line(
					img_res,
					cv::Point(lineIt->first.first, lineIt->first.second),
					cv::Point(lineIt->second.first, lineIt->second.second),
					cv::Scalar( 0, 0, 255), 1, 8
				);
			}
		}

		// Visualize rows
		for(rowIt=rows.begin();rowIt!=rows.end();rowIt++) {
			drawLine(img_res, rowIt->first);
			drawLine(img_res, rowIt->second);
		}

		// Visualize columns
		for(rowIt=columns.begin();rowIt!=columns.end();rowIt++) {
			drawLine(img_res, rowIt->first);
			drawLine(img_res, rowIt->second);
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


void drawLine(cv::Mat img_res, line currLine) {
	cv::line(
		img_res,
		cv::Point(currLine.first.first, currLine.first.second),
		cv::Point(currLine.second.first, currLine.second.second),
		cv::Scalar( 0, 255, 0), 5, 8
	);
}

// Utility functions
void findRows(std::vector<line> horizontal, std::vector<row>& rows) {
	// spaces vector
	std::vector<row> spaces;
	row currRow;

	// iterator
	std::vector<row>::iterator rowIt;

	findSpaces(horizontal, spaces, horizCompare, ROWS_SPACING);

	// now "shift" the pairs we have created so they match the rows
	for(rowIt=spaces.begin()+1;rowIt!=spaces.end()-1;rowIt++) {
		currRow.first = rowIt->second;
		currRow.second = (rowIt+1)->first;
		rows.push_back(currRow);
	}
}


void findColumns(
	std::vector<line> vertical,
	std::vector<row>& columns,
	int imageSize
) {
	// spaces vector
	std::vector<row> spaces;
	row currRow;

	// iterator
	std::vector<line>::iterator lineIt;
	std::vector<row>::iterator rowIt;

	std::sort(vertical.begin(), vertical.end(), vertCompare);

	// first remove all of the bordering vertical lines
	for(lineIt=vertical.begin();lineIt!=vertical.end();lineIt++) {
		// start of row
		if (
			lineIt->first.first < LEFT_BORDER
			|| lineIt->first.first > (imageSize - RIGHT_BORDER)
		) {
			vertical.erase(lineIt);
		}
	}

	// find all the spaces (don't add small ones)
	float currDist;
	for(lineIt=vertical.begin();lineIt!=vertical.end()-1;lineIt++) {
		// start of row
		currDist = (lineIt+1)->first.first - lineIt->first.first;
		if (currDist > MIN_BAR_SIZE) {
			currRow = make_pair(
				make_pair(lineIt->first, lineIt->second),
				make_pair((lineIt+1)->first, (lineIt+1)->second)
			);
			spaces.push_back(currRow);
		}
	}

	// no shifting, the ones we want are spaced out.
	// however remove whitespace from beginning and end
	spaces.erase(spaces.begin());
	spaces.erase(spaces.end());
	columns = spaces;

	printRows(spaces);
}

void printRows(std::vector<row>& rows) {
	std::vector<row>::iterator rowIt;
	int firstLineX, secondLineX;

	// print out distances
	for(rowIt=rows.begin();rowIt!=rows.end();rowIt++) {
		firstLineX = rowIt->first.first.first;
		secondLineX = rowIt->second.first.first;
		std::cout << "|" << firstLineX << "\t"  << secondLineX << "|" << std::endl;
	}
}


void findSpaces(
	std::vector<line> lines,
	std::vector<row>& spaces,
	pairCompare comparer,
	int spacing
) {
	// iterator
	std::vector<line>::iterator lineIt;

	std::sort(lines.begin(), lines.end(), comparer);

	// define row as two lines that have a large distance from their neighbours
	float prevDist = 0, currDist = 0;
	row currRow;

	// first find the large gaps
	for(lineIt=lines.begin();lineIt!=lines.end()-1;lineIt++) {
		currDist = (lineIt+1)->first.second - lineIt->first.second;
		if (currDist >= ROWS_SPACING*prevDist) {
			// start of row
			currRow.first = make_pair(lineIt->first, lineIt->second);
			// end of row
			currRow.second =  make_pair((lineIt+1)->first, (lineIt+1)->second);
			spaces.push_back(currRow);
		}
		prevDist = currDist;
	}
}


// finds the slope of a line
float findSlope(point first, point second) {
	return abs(
		(float)(first.first - second.first)
		/ (float)(first.second - second.second)
	);
}

// Compares 2 horizontal lines
bool horizCompare(line firstLine, line secondLine) {
  return firstLine.first.second < secondLine.first.second;
}

// Compares 2 vertical lines
bool vertCompare(line firstLine, line secondLine) {
  return firstLine.first.first < secondLine.first.first;
}
