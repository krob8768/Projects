/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package marsrover;

import java.awt.Point;
import java.util.concurrent.ThreadLocalRandom;

/**
 *
 * @author Kieran
 */
public final class Rover {
    private Point point;
    private char orientation;
    
    public Rover(){
        makeCoords();
        makeOrientation();
    }
    
    /**
     * Copy constructor
     * @param rover Rover to be copied
     */
    public Rover(Rover rover){
        this.point = rover.point;
        this.orientation = rover.orientation;                
    }
    
    /**
     * Generates random coordinates for rovers
     */
    protected void makeCoords(){
        point = new Point(ThreadLocalRandom.current().nextInt(0, 5 + 1), 
                ThreadLocalRandom.current().nextInt(0, 5 + 1));
    }

    /**
     * Generates random orientation for rovers
     */
    protected void makeOrientation(){
        int directionNum = ThreadLocalRandom.current().nextInt(1, 4 + 1);
        
        switch(directionNum){
            case 1:
                orientation = 'N';
                break;
            case 2:
                orientation = 'E';
                break;
            case 3:
                orientation = 'S';
                break;
            case 4:
                orientation = 'W';
                break;
                        
        }
    }
    
    /**
     * Changes orientation of rover using 
     * the direction instruction
     * @param newDir Direction for rover to turn
     */
    protected void changeDirection(char newDir){
        switch(newDir){
            case 'L':
                switch(orientation){
                    case 'N':
                        orientation = 'W';
                        break;
                    case 'E':
                        orientation = 'N';
                        break;
                    case 'S':
                        orientation = 'E';
                        break;
                    case 'W':
                        orientation = 'S';
                        break;
                }
                break;
            case 'R':
                switch(orientation){
                    case 'N':
                        orientation = 'E';
                        break;
                    case 'E':
                        orientation = 'S';
                        break;
                    case 'S':
                        orientation = 'W';
                        break;
                    case 'W':
                        orientation = 'N';
                        break;
                }
                break;  
        }
    }
    
    /**
     * Moves rover one grid point in the direction 
     * of its current orientation
     */
    protected void move(){
        switch(orientation){
            case 'N':
                point.setLocation(point.x, point.y + 1);
                break;
            case 'E':
                point.setLocation(point.x + 1, point.y);
                break;
            case 'S':
                point.setLocation(point.x, point.y - 1);
                break;
            case 'W':
                point.setLocation(point.x - 1, point.y);
                break;
        }
    }

    /**
     * @return the direction
     */
    public char getDirection() {
        return orientation;
    }

    /**
     * @param direction the direction to set
     */
    public void setDirection(char direction) {
        this.orientation = direction;
    }

    /**
     * @return the point
     */
    public Point getPoint() {
        return point;
    }

    /**
     * @param point the point to set
     */
    public void setPoint(Point point) {
        this.point = point;
    }
    
}
