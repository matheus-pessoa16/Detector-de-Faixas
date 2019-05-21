#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <iostream>
#include <vector>
#include <tuple>

using namespace std;

#define lowThresh  50
#define highThresh lowThresh*3
#define rho 1
#define theta CV_PI/180
#define hough_threshold 20
#define min_line_len 20
#define max_line_gap 300

const int morph_size = 3;

tuple<cv::Mat, cv::Mat> regionOfInterest(cv::Mat img)
{
    cv::Point vertices[2][2];
    cv::Mat mask = cv::Mat::zeros( cv::Size(img.cols, img.rows), CV_8UC3);

    vertices[0][0] = cv::Point(110,540); // x = 130  y = 540 -> canto inferior esquerdo
    vertices[0][1] = cv::Point(410,350); // x = 410  y = 350 -> canto superior esquerdo
    vertices[1][0] = cv::Point(575,350); // x = 570  y = 350 -> canto superior direito
    vertices[1][1] = cv::Point(920,540);

    const cv::Point* ppt[1] = { vertices[0] };
    int npt[] = { 4 };

    int channel_count, ignore_mask_color;

    if ( img.channels() > 2)
    {
        channel_count = img.channels(); // i.e. 3 or 4 depending on your image
        ignore_mask_color = (255) * channel_count;
    }
    else{
        ignore_mask_color = 255;
    }

    fillPoly( mask ,
              ppt  , 
              npt  ,
                1  ,
              cv::Scalar(255,0,0),
              8 );

    // essa imagem contém a porção da pista inteira 
    //imshow("Blank3", mask);

    cv::Mat maskedImage;

    cv::bitwise_and(img, mask, maskedImage);

    auto result = make_tuple(maskedImage, mask);

    return result;

}

void drawLines(cv::Mat img, vector<cv::Vec4i> lines)
{

    for( int i = 0; i < lines.size(); i++ )
    {
        cv::Vec4i l = lines[i];
        line( img, cv::Point(l[0], l[1]), cv::Point(l[2], l[3]), cv::Scalar(0,0,255), 4, cv::LINE_AA);
    }
}


cv::Mat houghLine( cv::Mat img)
{
    cv::Mat linesImg, grayImage;

    vector<cv::Vec4i> lines;

    cvtColor(img, grayImage, cv::COLOR_BGR2GRAY);

    cv::HoughLinesP(grayImage, lines, rho, theta, hough_threshold, min_line_len, max_line_gap);
    linesImg = img;
    
    drawLines(linesImg, lines);

    return linesImg;

}

cv::Mat weightedImg(cv::Mat imgModified, cv::Mat imgOriginal)
{
    cv::Mat dst;
    addWeighted( imgModified, 0.8, imgOriginal, 1, 0.0, dst);
    return dst;
}

cv::Mat morphologyOperations(cv::Mat input)
{
    
    cv::Mat element = getStructuringElement(cv::MORPH_RECT, cv::Size( morph_size+1, morph_size+1 ), cv::Point( morph_size, morph_size ) );

    cv::Mat dst;
    
    cv::erode( input, dst, element );
    
    element = getStructuringElement(cv::MORPH_RECT, cv::Size( morph_size+1, morph_size+1 ), cv::Point( morph_size, morph_size ) );
    cv::dilate( input, dst, element );
        
    return dst;
}

cv::Mat selectWhiteYellow(cv::Mat input)
{

    cv::Mat converted;
	cv::Mat mask = cv::Mat::zeros( cv::Size(input.cols, input.rows), CV_8UC3 );
	cv::Mat white_mask;
	cv::Mat yellow_mask;
    
    int low_H = 0, low_S = 200, low_V = 0;
    int high_H = 255, high_S = 255, high_V = 255; 

    cvtColor(input, converted, cv::COLOR_BGR2HLS);

    // white line
    cv::inRange(converted, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), white_mask);

    // yellow line
    low_H = 10;
    low_S = 0;
    low_V = 100;

    high_H = 40;
    high_S = 255;
    high_V = 255;

    cv::inRange(converted, cv::Scalar(low_H, low_S, low_V), cv::Scalar(high_H, high_S, high_V), yellow_mask);
	
	
    cv::bitwise_or(white_mask, yellow_mask, mask);

    cv::Mat maskedImage;
    
    cv::bitwise_and(input, input, mask, maskedImage);

    return mask;

}

int main(int argc, char** argv)
{

    cv::Mat img;
    cv::VideoCapture cap("solidWhite1.mp4");

    cap >> img;

    cv::resize(img, img, cv::Size(960, 540));

    cv::VideoWriter video("outcpp.avi", cv::VideoWriter::fourcc('M','J','P','G'), 24, cv::Size(img.size().width,img.size().height));

    cv::Mat image, grayImage, blurredImage, detectedEdges, dst, maskedImage, houghed, mask, closedImage, hslMaskImage;

	while(1)
	{
		
		cap >> image;
		
		if (image.empty())
      	break;
		
        cv::resize(image, image, cv::Size(960, 540));

        image = selectWhiteYellow(image);

        //image = weightedImg(image, hslMaskImage);
		//cv::imshow("Converted", image);

        cvtColor( image, grayImage, cv::COLOR_BGR2GRAY );

        // aplicando uma equalização de histograma
        // cv::equalizeHist( grayImage, grayImage );

        cv::GaussianBlur(grayImage, blurredImage, cv::Size(15,15), 0, 0);

        cv::Canny( blurredImage, detectedEdges, lowThresh, highThresh, 3 );
		
        closedImage = morphologyOperations(detectedEdges);

		// fazendo dst ser uma matriz de zeros, atuando como uma máscara
		dst = cv::Scalar::all(0);
		// copio a imagem e as bordas detectadas para dentro de dst
		image.copyTo( dst, closedImage);
		
		tie(maskedImage, mask) = regionOfInterest(dst);

		houghed = houghLine(maskedImage);

		cv::Mat res;

		res = weightedImg(houghed, image);

        res = weightedImg(mask, res);

		video.write(res);
		//imshow("Canny",detectedEdges);
		//imshow("Hough",houghed);
        // imshow("Histogram", grayImage);
        // imshow("Resultado",res);

		//cv::waitKey(0);
		
	}
	
	cap.release();
	video.release();
	cv::destroyAllWindows();
	
    return 0;   
}
