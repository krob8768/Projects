/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package marsrover;

import java.awt.Point;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;

/**
 *
 * @author Kieran
 */
public class Plateau {
    
    private int maxX;
    private int maxY;
    private int[] maxCoord;
    
    public Plateau(){
        maxCoord = new int[] {5,5};
  }
    
    /**
     * Prints plateau as grid
     * @param roverInfo Rover information
     */
    protected void printGrid(List<String> roverInfo){
        List<Point> roverPts = new ArrayList<>();
        for(int i = 0; i < roverInfo.size(); i ++){
            String[] rover = roverInfo.get(i).split(" ");
            int x = Integer.parseInt(rover[0]);
            int y = Integer.parseInt(rover[1]);
            roverPts.add(new Point(x,y));
            
        }
        System.out.println();
        for(int y = maxY; y >= 0; y--){
            for(int x = 0; x <= maxX; x++){
                if(roverPts.contains(new Point(x,y))){
                    int occurrences = Collections.frequency(roverPts, new Point(x,y));
                    if(occurrences == 1){
                        System.out.print("  R  ");
                    }
                    else if(occurrences > 1){
                        System.out.print(" R " + "x" + occurrences);
                    }
                    
                }
                else{
                    System.out.print("  *  ");
                }
            }
            System.out.print("\n\n");
        }
    }

    /**
     * @return the maximum x-coordinate
     */
    public int getMaxX() {
        return maxX;
    }

    /**
     * @param maxX the maximum x-coordinate to set
     */
    public void setMaxX(int maxX) {
        this.maxX = maxX;
    }

    /**
     * @return the maximum y-coordinate
     */
    public int getMaxY() {
        return maxY;
    }

    /**
     * @param maxY the the maximum y-coordinate to set
     */
    public void setMaxY(int maxY) {
        this.maxY = maxY;
    }

    /**
     * @return the minimum upper-right coordinate
     */
    public int[] getMaxCoord() {
        return maxCoord;
    }

    /**
     * @param maxCoord the minimum upper-right coordinate to set
     */
    public void setMaxCoord(int[] maxCoord) {
        this.maxCoord = maxCoord;
    }
 
}
