/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package marsrover;

import java.awt.Point;
import java.util.ArrayList;
import java.util.List;

/**
 *
 * @author Kieran
 */
public class Nasa {

    private List<String> roverInfo;
    
    public Nasa(){}
    
    /**
     * Returns user inputted maximum plateau coordinate, 
     * rover information and instructions
     * @return Maximum coordinate, rover information 
     * and instructions
     */
    protected  List<String> getInstrns(){
        List<String> roverInstrns = new ArrayList<>();
        String currRover;
        String maxCoord;
        
        // Asks user to enter the upper-right coordinates of the plateau
        int[] maxCoordArr = Controller.findMaxCoord();
        String statement = "\nPlease enter the upper-right coordinates of the plateau, e.g. 5 5 or 7 10\n\t"
                + "(Min. x: " + maxCoordArr[0] + ", Min. y: " + maxCoordArr[1] + ")";
        String input = Controller.getValidInput(statement, Controller.FORMAT_CODE.MAXCOORDINATE);      
        maxCoord = input;
        roverInstrns.add(input);
        
        boolean nextInstrn = true;
        while(nextInstrn){

            // User input rover information (position and orientation)
            statement = "\nPlease identify a rover to move by entering its current"
                    + " position and orientation\n\t(e.g. 1 2 N)"; 
            
            // Asks user to select a rover until a valid one has been entered
            boolean isValidRover = true;
            while(true){
                if(!isValidRover){
                    System.out.println("THERE IS NO ROVER IN THIS LOCATION! PLEASE TRY AGAIN!");
                }
                input = Controller.getValidInput(statement, Controller.FORMAT_CODE.ROVER_INFO);
                currRover = input;
                isValidRover = checkRoverInfo(input);
                
                if(isValidRover){
                    break;
                }
                else{
                    isValidRover = false;
                }
            }                
            roverInstrns.add(input);
            
            // Asks user to enter an instruction until a valid one has been entered
            boolean isValid = false;
            while(!isValid){
                statement = "\nPlease enter movement instructions for the identified"
                        + " rover\n\t(e.g. MMRMMRMRRM)";  
                input = Controller.getValidInput(statement, Controller.FORMAT_CODE.INSTRUCTION);
                /*isValid = validateInstrn(maxCoord, currRover, input);
                if(!isValid){
                    System.out.println("THIS INSTRUCTION PUTS THE ROVER OUT OF BOUNDS! PLEASE TRY AGAIN!");
                }*/
            }
            roverInstrns.add(input);
            
            // Asks user whether or not to input another rover instruction
            statement = "\nWould you like to send an instruction for another rover?"
                    + "\n\t(Type 1 for yes or 2 for no)";
            input = Controller.getValidInput(statement, Controller.FORMAT_CODE.CONFIRMATION);  
            
            int confirmation = Integer.parseInt(input);
            nextInstrn = (confirmation == 1);
        }
        
        return roverInstrns;
    }

    /**
     * Checks if user input rover information 
     * matches that of a deployed rover
     * @param rover User input rover information
     * @return True if user input rover matches a deployed rover
     */
    private boolean checkRoverInfo(String rover){
        for(int i = 0; i < roverInfo.size(); i++){
            if(rover.equals(roverInfo.get(i))){
                return true;
            }
        }
        return false;
    }
                
    /**
     * Checks if instruction puts rover out of bounds
     * @param maxCoord Maximum coordinate of plateau
     * @param roverInfo Rover information
     * @param instrn Instruction
     * @return True if instruction is valid
     */
    private boolean validateInstrn(String maxCoord, String roverInfo, String instrn){
        Rover rover = new Rover();
        String[] roverInfoArr = roverInfo.split(" ");
        int x = Integer.parseInt(roverInfoArr[0]);
        int y = Integer.parseInt(roverInfoArr[1]);
        char dir = roverInfoArr[2].charAt(0);
        rover.setPoint(new Point(x,y));
        rover.setDirection(dir);
        
        rover = Controller.moveRover(rover, instrn);
        
        String[] maxCoordArr = maxCoord.split(" ");
        int maxX = Integer.parseInt(maxCoordArr[0]);
        int maxY = Integer.parseInt(maxCoordArr[1]);
        
        // Return false if instruction puts rover out-of-bounds
        return !(rover.getPoint().x < 0 || rover.getPoint().x > maxX || 
                rover.getPoint().y < 0 || rover.getPoint().y > maxY);
    }
    
    /**
     * @return the roverInfo
     */
    public List<String> getRoverInfo() {
        return roverInfo;
    }

    /**
     * @param roverInfo the roverInfo to set
     */
    public void setRoverInfo(List<String> roverInfo) {
        this.roverInfo = roverInfo;
    }
    
}
