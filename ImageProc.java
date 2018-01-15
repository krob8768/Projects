/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package fvs;

import java.awt.image.BufferedImage;
import java.awt.image.DataBufferByte;
import java.io.ByteArrayInputStream;
import java.nio.ByteBuffer;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Iterator;
import java.util.List;
import java.util.concurrent.CountDownLatch;
import javafx.concurrent.Task;
import javafx.scene.image.Image;
import javafx.scene.image.PixelReader;
import javafx.scene.image.WritablePixelFormat;
import org.opencv.core.Core;
import org.opencv.core.CvType;
import static org.opencv.core.CvType.CV_8UC1;
import org.opencv.core.DMatch;
import org.opencv.core.KeyPoint;
import org.opencv.core.Mat;
import org.opencv.core.MatOfByte;
import org.opencv.core.MatOfDMatch;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfKeyPoint;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.features2d.DescriptorExtractor;
import org.opencv.features2d.DescriptorMatcher;
import org.opencv.features2d.FeatureDetector;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGR;
import static org.opencv.imgproc.Imgproc.COLOR_GRAY2BGRA;

/**
 *
 * @author Kieran
 */
public class ImageProc {
    private final static int GRAD_BLOCK_SIZE = 16;  // 16 x 16 block size
    private final static int BLOCK_SIZE = 5;  // 5 x 5 block size
    private Mat fpImg; 
    
    private int rowExt = 0;
    private int colExt = 0;
    
    public ImageProc(){
        fpImg = new Mat();
    }
    
    //////////////////////////////////////// OpenCV ///////////////////////////////////////////
    protected static Mat awtImg2Mat(java.awt.Image image){
        BufferedImage buffImg = (BufferedImage) image;
        byte[] pixels = ((DataBufferByte) buffImg.getRaster().getDataBuffer()).getData();
        Mat fpMat = new Mat(buffImg.getHeight(), buffImg.getWidth(), CvType.CV_8UC1);
        fpMat.put(0, 0, pixels);
        return fpMat;
    }
    
    protected static Mat fxImg2Mat(Image img){
        int width = (int) img.getWidth();
        int height = (int) img.getHeight();
        byte[] buffer = new byte[width * height * 4];

        PixelReader reader = img.getPixelReader();
        WritablePixelFormat<ByteBuffer> format = WritablePixelFormat.getByteBgraInstance();
        reader.getPixels(0, 0, width, height, format, buffer, 0, width * 4);

        Mat mat = new Mat(height, width, CvType.CV_8UC4);
        mat.put(0, 0, buffer);
        return mat;
    }
    
    /**
     * Convert a Mat object (OpenCV) in the corresponding Image for JavaFX
     *
     * @param frame
     *            the {@link Mat} representing the current frame
     * @return the {@link Image} to show
     */
    protected static Image mat2Image(Mat frame){
        // create a temporary buffer
        MatOfByte buffer = new MatOfByte();
        // encode the frame in the buffer, according to the PNG format
        Imgcodecs.imencode(".png", frame, buffer);
        // build and return an Image created from the image encoded in the
        // buffer
        return new javafx.scene.image.Image(new ByteArrayInputStream(buffer.toArray()));
    } 
    
    //////////////////////////////////// PREPROCESSING //////////////////////////////////
    protected Mat normalise(Mat sourceImg){
        Mat destImg = sourceImg.clone();
        
        //Imgproc.erode(destImg, destImg, new Mat());
        //Imgproc.GaussianBlur(destImg, destImg,new Size(5,5), 0);
        //Core.normalize(destImg, destImg, 100, 2, Core.NORM_MINMAX);
        destImg.convertTo(destImg, destImg.type(), 1, -125); // Decrease brightness
        //destImg.convertTo(destImg, destImg.type(), 1.5, 0); // Increase contrast
        //Imgproc.blur(destImg, destImg, new Size(3,3));
        
        //Imgproc.threshold(destImg, destImg, 10, 255, Imgproc.THRESH_TOZERO);
        //Imgproc.equalizeHist(destImg, destImg);
        destImg.convertTo(destImg, destImg.type(), 1.5, 0); // Increase contrast
        //Imgproc.threshold(destImg, destImg, 75, 255, Imgproc.THRESH_BINARY);
        
        return destImg;
    }
    
    ////////// ORIENTATION FIELD /////////
    protected Mat getOriField(Mat sourceImg){
        
        /// Generate grad_x and grad_y
        Mat gxMat = new Mat(new Size(), CvType.CV_32F);
        Mat gyMat = new Mat(new Size(), CvType.CV_32F);
        Mat oriMat = new Mat(sourceImg.size(), CvType.CV_32F);
        int ddepth = CvType.CV_32F;
        int kernel = -1; // Corresponds to the 3x3 Scharr filter
        
                
        for(int y = 0; y < sourceImg.rows(); y += GRAD_BLOCK_SIZE){
            for(int x = 0; x < sourceImg.cols(); x += GRAD_BLOCK_SIZE){
                Rect blockWindow = new Rect(x, y, GRAD_BLOCK_SIZE, GRAD_BLOCK_SIZE);
                Mat block = new Mat(sourceImg, blockWindow);
                
                // Gradient X - Computes horizontal gradient component (gx) for a single block
                Imgproc.Sobel(block, gxMat, ddepth, 1, 0, kernel, 1, 0);

                /*for(int j = y; j < y + blockSize; j ++){
                    for(int i = x; i < x + blockSize; i ++){
                        // Fill orientation block with single orientation value  
                        oriMat.put(j, i, gxMat.get(j, i));
                    }
                }*/
                /// Gradient Y - Computes vertical gradient component (gy) for a single block
                Imgproc.Sobel(block, gyMat, ddepth, 0, 1, kernel, 1, 0);
                
                // Block orientation estimate
                double[] orientation = {locOriEst(gxMat, gyMat)};
                for(int j = y; j < y + GRAD_BLOCK_SIZE; j ++){
                    for(int i = x; i < x + GRAD_BLOCK_SIZE; i ++){
                        // Fill orientation block with single orientation value  
                        oriMat.put(j, i, orientation);
                    }
                }
            }
        }
        
        return smoothOriField(oriMat);
    }
        
    protected Mat drawOriMap(Mat oriMat){
        
        Mat oriMap = new Mat(oriMat.size(), CvType.CV_32F);
        Imgproc.cvtColor(oriMap, oriMap, COLOR_GRAY2BGR);
        
        double rTheta, dTheta, posTheta;
        
        double a, b, c = Math.PI/2;
        double A, B = GRAD_BLOCK_SIZE/2, twoB = GRAD_BLOCK_SIZE;
        int coE;
        Point p1;
        Point p2;
        
        for(int y = 0; y < oriMap.rows(); y += GRAD_BLOCK_SIZE){
            for(int x = 0; x < oriMap.cols(); x += GRAD_BLOCK_SIZE){
                rTheta = oriMat.get(y, x)[0];
                dTheta = (180/Math.PI)*rTheta;
                //dTheta = (dTheta < 0) ? dTheta += 360 : dTheta;
                //System.out.println("x: " + x + ",   y: " + y + ",  theta: " + dTheta);
                
                posTheta = Math.abs(rTheta);
                if(posTheta < Math.PI/4){ // Less than 45 degrees
                    b = Math.PI - Math.PI/2 - posTheta;
                    a = posTheta;
                    A = (B / Math.sin(b)) * Math.sin(a);
                    coE = (rTheta > 0) ? 1 : -1;
                    p1 = new Point(x + B+(coE*A), y);
                    p2 = new Point(x + B-(coE*A), y + twoB);
                }
                else{ // Greater than 45 degrees
                    b = posTheta;
                    a = Math.PI/2 - posTheta;
                    A = (B / Math.sin(b)) * Math.sin(a);
                    coE = (rTheta > 0) ? 1 : -1;
                    p1 = new Point(x + twoB, y + B-(coE*A));
                    p2 = new Point(x, y + B+(coE*A));
                }
                
                Imgproc.line(oriMap, p1, p2, new Scalar(255,255,0), 1);
                
            }
        }
        
        return oriMap;
    }
    
    private double locOriEst(Mat gxMat, Mat gyMat){
        double dividend = 0;
        double divisor = 0;   
        for(int y = 0; y < gxMat.rows(); y++){
            for(int x = 0; x < gxMat.cols(); x++){
                dividend += (2 * gxMat.get(y, x)[0] * gyMat.get(y, x)[0]);
                divisor += ((gxMat.get(y, x)[0]*gxMat.get(y, x)[0]) - (gyMat.get(y, x)[0]*gyMat.get(y, x)[0]));
            }
        }
        double orientation = (0.5 * Math.atan2(dividend, divisor));
        if(orientation < 0){
            orientation += 2*Math.PI;
        }
        return orientation;
    }
    
    protected Mat smoothOriField(Mat oriMat){
        
        Mat phiXMat = new Mat(oriMat.size(), oriMat.type());
        Mat phiYMat = new Mat(oriMat.size(), oriMat.type());
        Mat smoothOriMat = new Mat(oriMat.size(), oriMat.type());
                
        // Convert into continuous vector field
        for (int y = 0; y < oriMat.rows(); y++){
            for (int x = 0; x < oriMat.cols(); x++){
                phiXMat.put(y, x, Math.cos(2*oriMat.get(y, x)[0]));
                phiYMat.put(y, x, Math.sin(2*oriMat.get(y, x)[0]));
            }
        }
        // Apply Gaussian low-pass filter to
        Imgproc.GaussianBlur(phiXMat, phiXMat, new Size(45,45), -1);
        Imgproc.GaussianBlur(phiYMat, phiYMat, new Size(45,45), -1);
        
        // Obtains smooth orientation field
        for (int y = 0; y < oriMat.rows(); y++){
            for (int x = 0; x < oriMat.cols(); x++){
                double phiY = phiYMat.get(y, x)[0];
                double phiX = phiXMat.get(y, x)[0];
                double[] smoothOriVal = {(0.5 * Math.atan2(phiY, phiX))};
                smoothOriMat.put(y, x, smoothOriVal);
            }
        }
        return smoothOriMat;
    }
    
    protected Mat drawGrid(Mat sourceImg){
        
        for(int x = 0; x < sourceImg.cols(); x += GRAD_BLOCK_SIZE ){
            Point p1 = new Point(x+0.5, 0);
            Point p2 = new Point(x+0.5, sourceImg.rows()-1);
            Imgproc.line(sourceImg, p1, p2, new Scalar(0,255,0), 1);
        }
        
        for(int y = 0; y < sourceImg.rows(); y += GRAD_BLOCK_SIZE ){
            Point p1 = new Point(0, y+0.5);
            Point p2 = new Point(sourceImg.cols()-1, y+0.5);
            Imgproc.line(sourceImg, p1, p2, new Scalar(0,255,0), 1);
        }
        
        return sourceImg;
    }
    ////////// RIDGE FREQUENCY ESTIMATION /////////
    private void ridgeFreqEst(Mat sourceImg, Mat oriMat){
        double angleShift = Math.PI/2;
        double orientation;
        int width;
        int height;
        
        Mat block;
        
        // Do for loop blocks  W x W
        for(int y = 0; y < sourceImg.rows(); y += GRAD_BLOCK_SIZE){
            height = GRAD_BLOCK_SIZE * 2;
            if(y == sourceImg.rows() - GRAD_BLOCK_SIZE){
                height = GRAD_BLOCK_SIZE;
            }
            for(int x = 0; x < sourceImg.cols(); x += GRAD_BLOCK_SIZE){
                // Get orientation of block and rotate by 90 degrees (orhtogonal to orientation)
                width = GRAD_BLOCK_SIZE * 2;
                if(x == sourceImg.cols() - GRAD_BLOCK_SIZE){
                    width = GRAD_BLOCK_SIZE;
                }
                Rect rect = new Rect(x, y, width, height);
                block = new Mat(sourceImg, rect);
            }
        }
        
        
    }
    
    //////////////// GABOR FILTER ////////////////
    protected Mat applyGabor(Mat sourceImg, Mat oriMat){
        Mat destImg = new Mat(sourceImg.size(), CvType.CV_32F);
        Mat newSource = sourceImg.clone();
        sourceImg.convertTo(newSource, CvType.CV_32F);
        
        //getGaborKernel(KernalSize, Sigma, Theta, Lambda,Gamma,psi);
        /*ksize – Size of the filter returned.
        sigma – Standard deviation of the gaussian envelope.
        theta – Orientation of the normal to the parallel stripes of a Gabor function.
        lambd – Wavelength of the sinusoidal factor.
        gamma – Spatial aspect ratio.
        psi – Phase offset.
        ktype – Type of filter coefficients. It can be CV_32F or CV_64F .*/
        
        for(int y = 0; y < newSource.rows(); y += GRAD_BLOCK_SIZE){
            for(int x = 0; x < newSource.cols(); x += GRAD_BLOCK_SIZE){
                Rect blockWindow = new Rect(x, y, GRAD_BLOCK_SIZE, GRAD_BLOCK_SIZE);
                Mat block = new Mat(newSource, blockWindow);
                Mat filBlock = new Mat();
                
                double orientation = oriMat.get(y, x)[0];
                Mat kernel = Imgproc.getGaborKernel(new Size(7,7), 2, (int)orientation, 8, CvType.CV_32F);
                Imgproc.filter2D(block, filBlock, CvType.CV_32F, kernel);
                
                for(int j = 0; j < GRAD_BLOCK_SIZE; j ++){
                    for(int i = 0; i < GRAD_BLOCK_SIZE; i ++){
                        double[] filVal = filBlock.get(j, i);
                        destImg.put(y+j, x+i, filVal);
                    }
                }
            }
        }
        return destImg;
    }
    
    //////////////////////////////////// THINNING //////////////////////////////////
    private int bool2Int(boolean b) { 
        return b ? 1 : 0;
    }
    
    private Mat skeletiseIter(Mat sourceImg, int step){
        //Perform a single thinning iteration, which is repeated until the skeletization is finalized
	Mat tempImg = sourceImg.clone();
	for (int i = 1; i < sourceImg.rows() - 1; i++){
            for (int j = 1; j < sourceImg.cols() - 1; j++){
                int p1 = (int)sourceImg.get(i, j)[0];
                int p2 = (int)sourceImg.get(i-1, j)[0];
                int p3 = (int)sourceImg.get(i-1, j+1)[0];
                int p4 = (int)sourceImg.get(i, j+1)[0];
                int p5 = (int)sourceImg.get(i+1, j+1)[0];
                int p6 = (int)sourceImg.get(i+1, j)[0];
                int p7 = (int)sourceImg.get(i+1, j-1)[0];
                int p8 = (int)sourceImg.get(i, j-1)[0];
                int p9 = (int)sourceImg.get(i-1, j-1)[0];
                int A =	bool2Int((p2 == 0 && p3 == 1)) + bool2Int((p3 == 0 && p4 == 1)) +
                        bool2Int((p4 == 0 && p5 == 1)) + bool2Int((p5 == 0 && p6 == 1)) +	
                        bool2Int((p6 == 0 && p7 == 1)) + bool2Int((p7 == 0 && p8 == 1)) +
                        bool2Int((p8 == 0 && p9 == 1)) + bool2Int((p9 == 0 && p2 == 1));
                int B = p2 + p3	+ p4 +	p5 + p6	+ p7 + p8 + p9;
                int m1 = step == 1 ? (p2 * p4 * p6) : (p2 * p4 * p8);
                int m2 = step == 1 ? (p4 * p6 * p8) : (p2 * p6 * p8);
                if (p1 == 1 && A == 1 && (B >= 2 && B <= 6) && m1 == 0 && m2 == 0){
                    tempImg.put(i,j,0);
                }
            }
	}
        return tempImg;
    }
    
    protected Mat thinning(Mat img){ 
        fpImg = img.clone();
        
        // Enforce the range to be in between 0 - 255 / Binarise - 0 and 1	
        Core.divide(255, fpImg, fpImg);
        
        Mat prev = Mat.zeros(img.size(), CV_8UC1);
        Mat diff = new Mat();
        
        do{
            long startTime = System.currentTimeMillis();
            
            fpImg = skeletiseIter(fpImg, 1);
            fpImg = skeletiseIter(fpImg, 2);
            Core.absdiff(fpImg, prev, diff);
            fpImg.copyTo(prev);
            long endTime = System.currentTimeMillis();
        long totalTime = endTime - startTime;
        //System.out.println(totalTime + "  milliseconds");
        }	
        while (Core.countNonZero(diff) > 0);
        
        Core.multiply(fpImg, new Scalar(255), fpImg);
        return fpImg;
    }
    
    /////////////////////////////////// MINUTISE EXTRACTION //////////////////////////////////
    protected List<List<Point>> detectMinutiae(Mat thinImg){
        Core.divide(255, thinImg, thinImg);
        List<Point> endings = new ArrayList<>();
        List<Point> forks = new ArrayList<>();
        
        for (int y = 1; y < thinImg.rows() - 1; y++){      
            //System.out.println("y:  " + y );
            for (int x = 1; x < thinImg.cols() - 1; x++){
                // Mat.get(int row, int col) - so x and y are reversed
                
                int p = (int)thinImg.get(y, x)[0];
                int p1 = (int)thinImg.get(y, x+1)[0];
                int p2 = (int)thinImg.get(y-1, x+1)[0];
                int p3 = (int)thinImg.get(y-1, x)[0];
                int p4 = (int)thinImg.get(y-1, x-1)[0];
                int p5 = (int)thinImg.get(y, x-1)[0];
                int p6 = (int)thinImg.get(y+1, x-1)[0];
                int p7 = (int)thinImg.get(y+1, x)[0];
                int p8 = (int)thinImg.get(y+1, x+1)[0];
                
                if(p == 1){ // Is ridge
                    int connections = p1+ p2 + p3 + p4 + p5 + p6 + p7 + p8;
                    
                    // Find Ridge endings
                    if(connections == 1){
                        endings.add(new Point(x,y));
                    }
                    
                    // Find Bifurcations
                    else if((p2+p4+p7 == 3) || (p4+p6+p1 == 3) || 
                            (p6+p8+p3 == 3) || (p8+p2+p5 == 3) ||
                            (p2+p5+p7 == 3) || (p4+p7+p1 == 3) || 
                            (p6+p1+p3 == 3) || (p8+p3+p5 == 3)){
                        //System.out.println("x:  " + x  + "  y:  " + y  + " -- boundaries:  " + boundaries);
                        forks.add(new Point(x,y));
                    }
                }
                
            }
                        
        }
        endings = rmvEdgePts(endings, thinImg);
        endings = rmvFalsePts(thinImg, endings, 1);
        forks = rmvEdgePts(forks, thinImg);
        
        List<List<Point>> minutiae = new ArrayList<>();
        minutiae.add(endings);
        minutiae.add(forks);
                
        Core.multiply(thinImg, new Scalar(255), thinImg);
        return minutiae;
    }
    
    private List<Point> rmvEdgePts(List<Point> minutiae, Mat sourceImg){
        
        int midCol = sourceImg.cols() / 2;
        int midRow = sourceImg.rows() / 2;
        
        List<Integer> quadrantList = getQuadrants(minutiae, midCol, midRow);        
        List<Point> pts2Rmv = new ArrayList<>();
        
        for(int i = 0; i < minutiae.size(); i ++){
            int edge, loops, xIncrement, yIncrement;
            Point startP = minutiae.get(i);
            
            int quadrant = quadrantList.get(i);
            
            // left edge
            if(quadrant == 1 || quadrant == 3){
                edge = 0;
                loops = Math.abs(edge - ((int)startP.x - 1));
                xIncrement = (-1);
                yIncrement = 0;
                if(checkForEdge(sourceImg, startP, loops, xIncrement, yIncrement)){
                    pts2Rmv.add(startP);
                }
            }
            // right edge
            if(quadrant == 2 || quadrant == 4){
                edge = sourceImg.cols() - 1;
                loops = Math.abs(edge - ((int)startP.x + 1));
                xIncrement = 1;
                yIncrement = 0;
                if(checkForEdge(sourceImg, startP, loops, xIncrement, yIncrement)){
                    pts2Rmv.add(startP);
                }
            }
            // top edge
            if(quadrant == 1 || quadrant == 2){
                edge = 0;
                loops = Math.abs(edge - ((int)startP.y - 1));
                xIncrement = 0;
                yIncrement = 1;
                if(checkForEdge(sourceImg, startP, loops, xIncrement, yIncrement)){
                    pts2Rmv.add(startP);
                }
            }
            // bottom edge
            if(quadrant == 3 || quadrant == 4){
                edge = sourceImg.rows() - 1;
                loops = Math.abs(edge - ((int)startP.y + 1));
                xIncrement = 0;
                yIncrement = 1;
                if(checkForEdge(sourceImg, startP, loops, xIncrement, yIncrement)){
                    pts2Rmv.add(startP);
                }
            }
                  
        }
        minutiae.removeAll(pts2Rmv);
            
        return minutiae;
    }
    
    private List<Integer> getQuadrants(List<Point> minutiae, int midCol, int midRow){
        List<Integer> quadrant = new ArrayList<>();
        
        for(int i = 0; i < minutiae.size(); i ++){
            //Top quadrants
            if(minutiae.get(i).y <= midRow){
                //Top-left - 1
                if(minutiae.get(i).x <= midCol){
                    quadrant.add(1);
                }
                // Top-right - 2
                else{
                    quadrant.add(2);
                }
            }
            //Bottom quadrants
            else{
                // Bottom-left - 3
                if(minutiae.get(i).x <= midCol){
                    quadrant.add(3);
                }
                // Bottom-right - 4
                else{
                    quadrant.add(4);
                }
            }
        }
        return quadrant;
    }
    
    private boolean checkForEdge(Mat sourceImg, Point startP, int loops, int xIncrement, int yIncrement){
               
        int loopCtr = 0;   
        int xCoord = (int)startP.x;
        int yCoord = (int)startP.y;
        
        while(loopCtr < loops){
            xCoord += xIncrement;
            yCoord += yIncrement;

            // If pixel is white (a ridge), end loop
            if(sourceImg.get(yCoord, xCoord)[0] == 1){
                break;
            }
            loopCtr++;
        }

        if(loopCtr == loops){
            return true;
        }
        return false;
    }
        
    protected Mat highlightMinutiae(Mat sourceImg, List<List<Point>> minutiae){        
        Mat destImg = sourceImg.clone();

        List<Point> endings = new ArrayList<>(minutiae.get(0));
        List<Point> forks = new ArrayList<>(minutiae.get(1));
        
        Imgproc.cvtColor(destImg, destImg, COLOR_GRAY2BGR);
        for(int i = 0; i < endings.size(); i ++){
            Point p1 = new Point(endings.get(i).x - 4, endings.get(i).y - 4);
            Point p2 = new Point(endings.get(i).x + 4, endings.get(i).y + 4);
            Imgproc.rectangle(destImg, p1, p2, new Scalar(0,0,255));
        }
        for(int i = 0; i < forks.size(); i ++){
            Imgproc.circle(destImg, forks.get(i), 4, new Scalar(255,127,0), 1);
        }
        return destImg;
    }
    
    protected Mat getCorners(Mat fpThin){
        Mat harris_normalised = new Mat(); 
        Mat harris_corners = Mat.zeros(fpThin.size(), CvType.CV_32FC1);
        Imgproc.cornerHarris(fpThin, harris_corners, 2, 3, 0.04, Core.BORDER_DEFAULT);
        Core.normalize(harris_corners, harris_normalised, 0, 255, Core.NORM_MINMAX, CvType.CV_32FC1, new Mat());
    
        ////next bit...
        
        float threshold = 150;
        List<KeyPoint> keypoints = new ArrayList<>();
        Mat rescaled = new Mat();
        Core.convertScaleAbs(harris_normalised, rescaled);
        Mat harris_c = new Mat(rescaled.rows(), rescaled.cols(), CvType.CV_8UC3);
        List<Mat> harrisList = new ArrayList<>();
        harrisList.add(harris_c);
        harrisList.add(harris_c);
        harrisList.add(harris_c);
        List<Mat> in = new ArrayList<>();
        in.add(rescaled);
        in.add(rescaled);
        in.add(rescaled);
        MatOfInt from_to = new MatOfInt(); 
        int[] fromToArr = {0,0, 1,1, 2,2};
        from_to.fromArray(fromToArr);
        Core.mixChannels(in, harrisList, from_to);
        for(int x = 0; x < harris_normalised.cols(); x++){
            for(int y = 0; y < harris_normalised.rows(); y++){
                if((int)harris_normalised.get(y, x)[0] > threshold){// might be x, y
                    //  Draw    or  store   the keypoint    location    here,   just    like
                    //you   decide. In  our case    we  will    store   the location    of  
                    //  the keypoint
                    Imgproc.circle(harris_c, new Point(x, y), 5, new Scalar(0,255,0), 1);
                    keypoints.add(new KeyPoint(x, y, 1));
                }
            }
        }
        return harris_c;
    }
    
    protected List<Point> rmvFalsePts(Mat thinImg, List<Point> minutiae, int type){
        Mat sourceImg = thinImg.clone();
        
        List<Point> removeList = new ArrayList<>();
        
        int smallestDist, blockSize = 23;
        int halfBlock = (blockSize-1)/2;
        int lDist, rDist, tDist, bDist;
        
        Rect rect;
        for(int i = 0; i < minutiae.size(); i ++){
            System.out.println("i: " + i);
            int xPt = (int)minutiae.get(i).x;
            int yPt = (int)minutiae.get(i).y;
            
            lDist = xPt - 0;
            rDist = thinImg.cols()-1 - xPt;
            tDist = yPt - 0;
            bDist = thinImg.rows()-1 - yPt;
            
            smallestDist = Math.min(Math.min(lDist, rDist), Math.min(tDist, bDist));
            if(smallestDist <= halfBlock){
                removeList.add(minutiae.get(i));
                continue;
            }
            int sourceX = xPt - halfBlock;
            int sourceY = yPt - halfBlock;
            rect = new Rect(sourceX, sourceY, blockSize, blockSize);
            Mat imgBlock = new Mat(sourceImg, rect);
            Mat workBlock = Mat.zeros(imgBlock.size(), CvType.CV_8U);
            double[] ptInitVal = {-1};
            workBlock.put(halfBlock, halfBlock, ptInitVal);
            
            switch(type){
                case 1:
                    if(!endingRmvl(imgBlock, workBlock, new Point(halfBlock,halfBlock))){
                        removeList.add(minutiae.get(i));
                    }
                case 3:
                    //forkRmvl();
            }
        }
        minutiae.removeAll(removeList);
        return minutiae;
    }
    
    private boolean endingRmvl(Mat imgBlock, Mat workBlock, Point start){
        //Core.divide(255, imgBlock, imgBlock);
        List<Point> connectedPts = new ArrayList<>();
        List<Point> traversed = new ArrayList<>();
        connectedPts.add(start);
        
        while(!connectedPts.isEmpty()){
            Point pt = connectedPts.get(0);
            int x = (int)pt.x, y = (int)pt.y;
            for(int j = (-1)*1; j < 2; j++){
                for(int i = (-1)*1; i < 2; i++){
                    //System.out.println("x: " + (x + i) + ",  y: " + (y + j));
                    if(y+j < 0 || y+j >= imgBlock.rows() || x+i < 0 || x+i >= imgBlock.cols()){
                        continue;
                    }
                    if(imgBlock.get(y + j, x + i)[0] == 1){
                        double[] one = {1};
                        workBlock.put(y + j, x + i, one);
                        Point p = new Point(x+i, y+j);
                        //System.out.println("Point x: " + p.x + ",  y: " + p.y);
                        if(!traversed.contains(p)){
                            connectedPts.add(p);
                            traversed.add(p);
                        }                            
                    }
                }
            }
            connectedPts.remove(0);
        }
       
        int transCtr = 0;
        for(int x = 1; x < workBlock.cols(); x ++){
            transCtr += (int)workBlock.get(0, x)[0];
        }
        for(int y = 1; y < workBlock.rows(); y ++){
            transCtr += (int)workBlock.get(y, workBlock.cols()-1)[0]; 
        }
        for(int x = workBlock.cols() - 1; x > -1; x --){
            transCtr += (int)workBlock.get(workBlock.rows()-1, x)[0];
        }
        for(int y = workBlock.rows() - 1; y > -1; y --){
            transCtr += (int)workBlock.get(y, 0)[0];
        }
        return transCtr == 1;
    }
    
    private void forkRmvl(){
        
    }
    
    /////////////////////////////////// MINUTIAE EXTRACTION //////////////////////////////////
    private void matchMinutiae(List<Minutiae> tMinutiae, List<Minutiae> qMinutiae){
        double oriShift, xShift, yShift;
    }
    
    protected List<Minutiae> pts2Minutiae(List<List<Point>> ptsList, Mat orientationMat){
        List<Minutiae> minutiae = new ArrayList<>();
        
        for(int i = 0; i < ptsList.get(0).size(); i ++){
            Point p = ptsList.get(0).get(i);
            float angle = (float)orientationMat.get((int)p.y, (int)p.x)[0];
            int type = 1;
            Minutiae m = new Minutiae(p, angle, type);
            minutiae.add(m);
        }
        for(int i = 0; i < ptsList.get(1).size(); i ++){
            Point p = ptsList.get(1).get(i);
            float angle = (float)orientationMat.get((int)p.y, (int)p.x)[0];
            int type = 3;
            Minutiae m = new Minutiae(p, angle, type);
            minutiae.add(m);
        }
        return minutiae;
    }
    
    /////////////////////////////////// SINGULARITY DETECTION //////////////////////////////////
    protected List<Singularity> detectSingularity(Mat oriMat){
        int bs = GRAD_BLOCK_SIZE;
        List<Singularity> singularities = new ArrayList<>();
        for(int y = bs*3; y < oriMat.rows() - bs*3; y += bs){
            for(int x = bs*3; x < oriMat.cols() - bs*3; x += bs){
                double p1 = oriMat.get(y, x)[0] * 180/Math.PI;
                //double p2 = oriMat.get(y-bs, x)[0] * 180/Math.PI;
                //double p3 = oriMat.get(y-bs, x+bs)[0] * 180/Math.PI;
                double p4 = oriMat.get(y, x+bs)[0] * 180/Math.PI;
                //double p5 = oriMat.get(y+bs, x+bs)[0] * 180/Math.PI;
                //double p6 = oriMat.get(y+bs, x)[0] * 180/Math.PI;
                //double p7 = oriMat.get(y+bs, x-1)[0] * 180/Math.PI;
                //double p8 = oriMat.get(y, x-bs)[0] * 180/Math.PI;
                //double p9 = oriMat.get(y-bs, x-bs)[0] * 180/Math.PI;
                //double oriBlockSum = p2+p3+p4+p5+p6+p7+p8+p9;
                double oriBlockSum = Math.abs(p1) + Math.abs(p4);
                
                if((p1 < 0 && p4 > 0) || (p4 < 0 && p1 > 0)){
                    if(oriBlockSum < 50 && oriBlockSum > 10){
                        Point pt1 = new Point(x, y);
                        System.out.println(pt1); 
                        if(pt1.x > 100 && pt1.x < 300){
                            Singularity s1 = new Singularity(pt1, (float)oriMat.get(y, x)[0]);
                            singularities.add(s1);
                        }
                    }
                }
            }
        }
        /*for(int i = 0; i < singularities.size(); i ++){
            System.out.println(singularities.get(i));
        }*/
        return singularities;
    }
    
    protected Mat highlightSingularity(Mat sourceImg, List<Singularity> singularities){
        Mat destImg = sourceImg.clone();
        Imgproc.cvtColor(destImg, destImg, COLOR_GRAY2BGR);
        //destImg = drawGrid(destImg);
        
        for(int i = 0; i < singularities.size(); i ++){
            Point p = new Point(singularities.get(i).x, singularities.get(i).y); 
            Imgproc.circle(destImg, p, 7, new Scalar(0,255,0), 2);
        }
        return destImg;
    }
    
    /////////////////////////////////// MINUTIAE MATCHING //////////////////////////////////
    protected void match(List<Minutiae> tMinutiae, List<Singularity> tSing, List<Minutiae> qMinutiae, List<Singularity> qSing){
        Point tSingP, qSingP;
        float tSingOri, qSingOri;
        double xShift, yShift;
        float rotation;
        
        double queryX, queryY;
        
        // For all template singularities
        for(int i = 0; i < tSing.size(); i ++){
            tSingP = new Point(tSing.get(i).x,tSing.get(i).y);
            tSingOri = tSing.get(i).orientation;
            //For all query singularities
            for(int j = 0; j < qSing.size(); j ++){
                qSingP = new Point(qSing.get(j).x,qSing.get(j).y);
                qSingOri = qSing.get(j).orientation;
                
                xShift = qSingP.x - tSingP.x;
                yShift = qSingP.y - tSingP.y;
                rotation = qSingOri - tSingOri;
                // For all query minutiae
                for(int k = 0; k < qMinutiae.size(); k ++){
                    //Transform
                    double qX = qMinutiae.get(k).x - xShift;
                    double qY = qMinutiae.get(k).y - yShift;
                    
                    double singDistX = qX - qSingP.x;
                    double singDistY = qY - qSingP.y;
                    
                    double newX = singDistY * Math.cos(rotation) - singDistX * (-1*Math.sin(rotation));
                    double newY = singDistY * (-1*Math.sin(rotation)) + singDistX * Math.cos(rotation);
                    
                    Point newPoint = new Point(newX, newY);
                    float newRotation = qMinutiae.get(k).orientation - rotation;
                    
                }
            }
        }
        for(int l = 0; l < tMinutiae.size(); l ++){
            System.out.println("x: " + tMinutiae.get(l).x + ",  y: " + tMinutiae.get(l).y
                    + ",  angle: " + tMinutiae.get(l).orientation * 180/Math.PI);
        }
        
    }
    
    /////////////////////////////////// ORB //////////////////////////////////
    protected MatOfKeyPoint orbExtract(Mat thinImg, List<Minutiae> minutiae){
        List<KeyPoint> keypointList = new ArrayList<>();
        for(int i = 0; i < minutiae.size(); i++){
            float x = (float)minutiae.get(i).x;
            float y = (float)minutiae.get(i).y;
            float size = 30;
            float angle = (float)minutiae.get(i).orientation + (float)Math.PI;
            KeyPoint keypoint = new KeyPoint(x, y, size, angle);
            keypointList.add(keypoint);
        }
        
        MatOfKeyPoint keypoints = new MatOfKeyPoint();
        keypoints.fromList(keypointList);
       
        return keypoints;
    }
    
    protected MatOfDMatch orbMatch(Mat img1, MatOfKeyPoint keypoints1, Mat img2, MatOfKeyPoint keypoints2){
        //Mat img1 = Highgui.imread(filename1, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
        //Mat img2 = Highgui.imread(filename1, Highgui.CV_LOAD_IMAGE_GRAYSCALE);
        
        //Definition of ORB keypoint detector and descriptor extractors
        //FeatureDetector detector = FeatureDetector.create(FeatureDetector.ORB); 
        DescriptorExtractor extractor = DescriptorExtractor.create(DescriptorExtractor.SIFT);

        //Detect keypoints
        //detector.detect(thinImg, keypoints);
        
        //Extract descriptors
        Mat descriptors1 = new Mat();
        Mat descriptors2 = new Mat();
        extractor.compute(img1, keypoints1, descriptors1);
        extractor.compute(img2, keypoints2, descriptors2);

        //Definition of descriptor matcher
        DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);
        //DescriptorMatcher matcher = DescriptorMatcher.create(DescriptorMatcher.BRUTEFORCE_HAMMING);

        //Match points of two images
        MatOfDMatch matches = new MatOfDMatch();
        matcher.match(descriptors1, descriptors2, matches);
        //matcher.
        List<DMatch> matchList = matches.toList();
        for(int i = 0; i < matchList.size(); i++){
            //System.out.println("Distnace: " + matchList.get(i).distance);
        }
        System.out.println("Match Score: " + matchList.size());
        
        return matches;
    }
    
    //////////////////////////////////////////
    protected class Minutiae{
        int x;
        int y;
        float orientation;
        int type;
        
        public Minutiae(Point pt, float angle, int mType){
            x = (int)pt.x;
            y = (int)pt.y;
            orientation = angle;
            type = mType;
        }
    }
    
    protected class Singularity{
        int x;
        int y;
        float orientation;
        
        public Singularity(Point pt, float angle){
            x = (int)pt.x;
            y = (int)pt.y;
            orientation = angle;
        }
    }
    
    /////////////////////////////////// IMAGE BLOCKS ////////////////////////////////////////////////
    private List<Mat> splitImage(Mat sourceImg, int blockSize){
                
        List<Mat> blockList = new ArrayList<>();
        int blockCtr = 0;

        for(int y = 0; y < sourceImg.rows() - 1; y += blockSize){ 
            for(int x = 0; x < sourceImg.cols() - 1; x += blockSize){
                Rect rect = new Rect(x, y, blockSize, blockSize);
                Mat block = new Mat(sourceImg, rect);
                blockList.add(block);
            }
        }
        return blockList;
    }
    
    private Mat mergeImage(List<Mat> blockList, int hBlockTotal, int vBlockTotal){

        List<Mat> hList = new ArrayList<>();
        List<Mat> vList = new ArrayList<>();

        int blockCtr = 0;
        
        for(int y = 0; y < vBlockTotal; y++){
            for(int x = 0; x < hBlockTotal; x++){
                hList.add(blockList.get(blockCtr));
                //System.out.println(blockList.get(blockCtr).get(5, 5)[0]);
                //System.out.println("inin loop: " + blockList.get(blockCtr).get(y, x)[0]);
                blockCtr++;
            }
            Mat hMat = new Mat();
            Core.hconcat(hList, hMat);
            hList.clear();
            vList.add(hMat);
            
        }
        Mat vMat = new Mat();
        Core.vconcat(vList, vMat);
        
        return vMat;
    }
    
    ///////////////////////////////////////////////////////////////////////////////////////////
    
    /**
      * Replicates the right and bottom edges of the image 
      * until the width and height is divisible by the 
      * square root of the block size
      * @param sourceImg Image channel for border to be added to
      * @return Image channel with added borders
      */
    private Mat borderExtend(Mat sourceImg){
        
        //colExt = sqrtBlockNum - (source.cols() % sqrtBlockNum); 
        //rowExt = sqrtBlockNum - (source.rows() % sqrtBlockNum);               

        colExt = sourceImg.cols() % BLOCK_SIZE;
        rowExt = sourceImg.rows() % BLOCK_SIZE;
        
        // Add border so image can be split evenly into blocks 
        Core.copyMakeBorder(sourceImg, sourceImg, 0, rowExt, 
                0, colExt, Core.BORDER_CONSTANT, Scalar.all(255));  // White border
        
        return sourceImg;
    }
    
    private Mat borderEdge(Mat sourceImg){
        // Add extra border (so edge pixels can have a full boundary)
        Core.copyMakeBorder(sourceImg, sourceImg, 1, 1, 
                1, 1, Core.BORDER_CONSTANT, Scalar.all(255));
        
        return sourceImg;
    }
        
    private class Section{
        private Mat img;
        private int id;
        
        public Section(Mat img, int id){
            this.img = img;
            this.id = id;
        }
    }
    
}
