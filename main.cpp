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

#define LEFT_BORDER 20
#define RIGHT_BORDER 20
#define TOP_BORDER 10
#define BOTTOM_BORDER 10

#define MIN_BAR_HEIGHT 20
#define MIN_BAR_WIDTH 30

// padding on top of each bar for other notes
#define TOP_BAR_PADDING 10

#define SEGMENTS 4

#define CROPPED_BARS_PATH "./cropped_bars/bar"
#define CROPPED_NOTES_PATH "./cropped_notes/note"

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
void findRows(
	std::vector<line> horizontal,
	std::vector<row>& rows,
	int imageSize
);
void findColumns(
	std::vector<line> vertical,
	std::vector<row>& columns,
	int imageSize
);
void findSpaces(
	std::vector<row>& ans,
	std::vector<line>& lines,
	bool vertical
);
void findRowSpaces(
	std::vector<row>& ans,
	std::vector<line>& lines,
	bool vertical
);
void removeBorders(
	std::vector<line>& ans,
	std::vector<line>& lines,
	bool vertical,
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
	Hough hough;
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
			} else if (lineSlope < MIN_SLOPE) {
				vertical.push_back(std::make_pair(lineIt->first, lineIt->second));
			}
		}

		// now we have all of the horizontal lines.
		// find the ones clustered together and declare them as a row
		findRows(horizontal, rows, img_res.rows);
		findColumns(vertical, columns, img_res.cols);

		// now crop based on each row
		int noteIndex = 0;
		for(rowIt=rows.begin();rowIt!=rows.end();rowIt++) {
			for(rowIt2=columns.begin();rowIt2!=columns.end();rowIt2++) {
				// each iteration gives us a bar!
				rectTopLeftX = rowIt2->first.first.first;
				rectTopLeftY = rowIt->first.first.second;
				rectBottomLeftX = rowIt2->second.first.first;
				rectBottomLeftY = rowIt->second.first.second;

				roi.x = rectTopLeftX;
		    roi.y = rectTopLeftY - TOP_BAR_PADDING;
		    roi.width = (rectBottomLeftX - rectTopLeftX) / SEGMENTS;
		    roi.height = rectBottomLeftY - rectTopLeftY + 2*TOP_BAR_PADDING;

				// normalizations
				roi.y = (roi.y < 0) ? 0 : roi.y;
				roi.height = (roi.y + roi.height > img_res.rows) ? img_res.rows - roi.y : roi.height;

		    /* Crop the original image to the defined ROI */
				// save image
				if(roi.area() > 0) {
					// write all notes
					for(int i = 0; i < SEGMENTS; i++){
						img_crop = img_res(roi);
						out << noteIndex;
						std::string path = CROPPED_NOTES_PATH + out.str() + ".jpg";
						cv::imwrite(path, img_crop);
						out.str(std::string());

						// move on to the next note
						roi.x += roi.width;
						noteIndex++;
					}
				}
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
void findSpaces(
	std::vector<row>& ans,
	std::vector<line>& lines,
	bool vertical
) {
	// find all the spaces (don't add small ones)
	std::vector<line>::iterator lineIt;
	int currDist;
	int minLength = (vertical) ? MIN_BAR_WIDTH : MIN_BAR_HEIGHT;

	if (!lines.size()){
		return;
	}

	for(lineIt=lines.begin();lineIt < lines.end()-1;lineIt++) {
		// start of row
		if (vertical) {
			currDist = (lineIt+1)->first.first - lineIt->first.first;
		} else {
			currDist = (lineIt+1)->first.second - lineIt->first.second;
		}
		if (currDist > minLength) {
			ans.push_back(make_pair(*lineIt, *(lineIt + 1)));
		}
	}
}


// function in beta, slightly different approach from above
void findRowSpaces(
	std::vector<row>& ans,
	std::vector<line>& lines,
	bool vertical
) {
	// find all the spaces (don't add small ones)
	std::vector<line>::iterator lineIt;
	line firstLine;
	bool finished = true;
	int currDist;
	int minLength = (vertical) ? MIN_BAR_WIDTH : MIN_BAR_HEIGHT;

	if (!lines.size()){
		return;
	}

	for(lineIt=lines.begin();lineIt < lines.end()-1;lineIt++) {
		// start of row
		if (vertical) {
			currDist = (lineIt+1)->first.first - lineIt->first.first;
		} else {
			currDist = (lineIt+1)->first.second - lineIt->first.second;
		}
		if (currDist > minLength) {
			ans.push_back(make_pair(firstLine, *lineIt));
			finished = true;
		} else if (finished) {
			firstLine = *lineIt;
			finished = false;
		}
	}
}


void removeBorders(
	std::vector<line>& ans,
	std::vector<line>& lines,
	bool vertical,
	int imageSize
) {
	// remove all of the bordering lines
	std::vector<line>::iterator lineIt;

	for(lineIt=lines.begin();lineIt!=lines.end();lineIt++) {
		// start of row
		if (vertical) {
			if (
				lineIt->first.first > LEFT_BORDER
				&& lineIt->first.first < (imageSize - RIGHT_BORDER)
			){
				// push 2 points into the new vector
				ans.push_back(*lineIt);
			}
		} else {
			if (
				lineIt->first.second > TOP_BORDER
				&& lineIt->first.second < (imageSize - BOTTOM_BORDER)
			){
				// push 2 points into the new vector
				ans.push_back(*lineIt);
			}
		}
	}
}

void findRows(
	std::vector<line> horizontal,
	std::vector<row>& rows,
	int imageSize
) {
	// spaces vector
	std::vector<row> spaces;
	std::vector<line> nonBorders;
	row currRow;

	// iterator
	std::vector<row>::iterator rowIt;

	std::sort(horizontal.begin(), horizontal.end(), horizCompare);

	removeBorders(
		nonBorders,
		horizontal,
		false,
		imageSize
	);
	findRowSpaces(
		spaces,
		nonBorders,
		false
	);

	if(!spaces.size()){
		return;
	}

	rows = spaces;
}

void findColumns(
	std::vector<line> vertical,
	std::vector<row>& columns,
	int imageSize
) {
	// spaces vector
	std::vector<row> spaces;
	std::vector<line> nonBorders;
	row currRow;

	// iterator
	std::vector<line>::iterator lineIt;
	std::vector<row>::iterator rowIt;

	std::sort(vertical.begin(), vertical.end(), vertCompare);

	removeBorders(
		nonBorders,
		vertical,
		true,
		imageSize
	);
	findSpaces(
		spaces,
		nonBorders,
		true
	);
	// no shifting, the ones we want are spaced out.
	// however remove whitespace from beginning and end
	columns = spaces;
}

void printRows(std::vector<row>& rows) {
	std::vector<row>::iterator rowIt;
	int firstLineX, secondLineX;

	// print out distances
	for(rowIt=rows.begin();rowIt!=rows.end();rowIt++) {
		firstLineX = rowIt->first.first.second;
		secondLineX = rowIt->second.first.second;
		std::cout << "|" << firstLineX << "\t"  << secondLineX << "|" << std::endl;
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
