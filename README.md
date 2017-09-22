# Notes Recognition

An image processing software that gets as input an image of some notes, and as an output plays the melody
of those notes

## Theory + Algorithms

- Hough Transform
- Simple NN for recognizing the bars

## Prequesites

- OpenCV

## Building & Running the Code

```bash
cd \...\Notes-Recog-master
make
.\notes
```

## What we did:

We seperated our project into 2 main parts:
Step 1)
Identifying the individual bars

Step 2)
Converting the bars' notes into a format we can play


Step 1)
We ran first a binarization on the image.

Next we ran a hough transform for straight lines,
using an algorithm we have seen in class.

We eliminated all diagonal lines by calculating the
slope and increasing the threshold (most music sheets contain perpendicular lines only)

We then hunted for the rows that we wanted (a
collection of bars). This was done by finding the
large gaps between the horizontal lines, and then taking the inverse - we wanted rows not gaps.

Finally, we sliced each row into bars using the vertical lines found. each bar image was cropped and numbered, and was passed onto stage 2.
