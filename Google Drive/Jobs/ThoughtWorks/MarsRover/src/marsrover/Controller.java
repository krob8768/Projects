/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package marsrover;

import java.awt.Point;
import java.util.ArrayList;
import java.util.List;
import java.util.Scanner;

/**
 *
 * @author Kieran
 */
public class Controller {
    
    /**
     * Types of user input to be checked for validity
     */
    public enum FORMAT_CODE{
        MAXCOORDINATE, ROVER_INFO, INSTRUCTION, CONFIRMATION;
    }
    
    private static Plateau plateau = new Plateau();
    private static Nasa nasa = new Nasa();
    private static List<Rover> squad = new ArrayList<>();  
    private static List<Integer> outputOrder = new ArrayList<>();
    private static String dupMoveInfo = "";
    
    private static List<Point> beacons = new ArrayList<>();
   
    public static void main(String[] args) {
        // Creates 4 Rover objects with random position and orientation
        initRovers();       
        
        boolean repeat = true;
        while(repeat){        
            // Sends location of each rover to Nasa as a list of strings
            nasa.setRoverInfo(getRoverInfo()); 
            
            // Prints rover information to screen
            showRoverInfo();

            plateau.setMaxCoord(findMaxCoord());
            
            // Gets NASA input and splits it into strings
            List<String> input = nasa.getInstrns();

            // Gets and sets maximum coordinate of the plateau
            String[] maxCoordArr = input.get(0).split(" ");
            int maxX = Integer.parseInt(maxCoordArr[0]);
            int maxY = Integer.parseInt(maxCoordArr[1]);   
            plateau.setMaxX(maxX);
            plateau.setMaxY(maxY);

            // Moves rovers as per instructions
            for(int i = 1; i < input.size(); i++){
                int roverIdx = getRoverIdx(input.get(i));
                outputOrder.add(roverIdx);
                Rover rover = new Rover(squad.get(roverIdx));
                squad.set(roverIdx, moveRover(rover, input.get(i+1)));
                i++;
            }
           
            // Changes a rovers orientation if it is identical to another
            moveDupRover();
            
            // Prints grid and formatted input and output to screen
            plateau.printGrid(getRoverInfo());
            showFormattedInput(input);
            showFormattedOuput();
            
            // Shows changes made to rovers with identical orientation and position 
            System.out.println(dupMoveInfo);
            dupMoveInfo = "";
            
            outputOrder.clear();
            showRoverInfo();
            
            // Asks user to continue or end the appllication
            String statement = "\nType 1 to send more instructions\nType 2 to exit";
            String input2 = getValidInput(statement, FORMAT_CODE.CONFIRMATION);
            int confirmation = Integer.parseInt(input2);
            repeat = (confirmation == 1);
        }
    }
    
    /**
     * Changes the orientation of a rover if two are identical 
     */
    private static void moveDupRover(){
        boolean moved = false;
        for(int i = 0; i < squad.size(); i++){
            for(int j = 0; j < squad.size(); j++){
                if(i == j){
                    continue;
                }
                // While two rovers are the same position and orientaion
                while(squad.get(i).getPoint().equals(squad.get(j).getPoint()) && 
                        squad.get(i).getDirection() == squad.get(j).getDirection()){
                    
                    squad.get(i).changeDirection('L');
                    //squad.get(i).setMoved(true);
                    moved = true;
                    System.out.println();
                }
                if(moved){
                    dupMoveInfo += "\nThe orientaion of Rover " + (i+1) + " has been changed "
                            + "so that it is no longer identical to Rover " + (j+1) + "\n";
                    moved = false;
                }
            }
        }
    }
    
    /**
     * Returns index in the rover list of a rover to be moved
     * @param roverInfo Position and orientation of rover as a string
     * @return Rover index
     */
    private static int getRoverIdx(String roverInfo){
        String[] roverInfoArr = roverInfo.split(" ");    
        int x = Integer.parseInt(roverInfoArr[0]);
        int y = Integer.parseInt(roverInfoArr[1]);
        Point pt = new Point(x,y);
        char dir = roverInfoArr[2].charAt(0);
        int roverIdx = -1;
        for(int i = 0; i < squad.size(); i++){
            if(squad.get(i).getPoint().equals(pt) && squad.get(i).getDirection() == dir){
                roverIdx = i;
                break;
            }
        }
        return roverIdx;
    }
    
    /**
     * Moves rover according to the supplied instructions
     * @param rover The rover to be moved
     * @param instrn The movement instruction
     * @return Moved rover
     */
    protected static Rover moveRover(Rover rover, String instrn){
        
        char[] movements = instrn.toCharArray();
        for(int i = 0; i < movements.length; i++){
            if('L' == movements[i] || 'R' == movements[i]){
                rover.changeDirection(movements[i]);
            }
            else{
                
                
                Point currPosition = rover.getPoint();
                rover.move();
                if(!isOnPlateau(rover)){
                    beacons.add(currPosition);
                }
                
            }
        }
        return rover;
    }
    
    private static boolean isOnPlateau(Rover rover){
        
        return !(rover.getPoint().x < 0 || rover.getPoint().x > plateau.getMaxX() || 
                rover.getPoint().y < 0 || rover.getPoint().y > plateau.getMaxY());
    }
    
    /**
     * Initialises rovers
     */
    private static void initRovers(){
        Rover rover;
        for(int i = 0; i < 4; i++){
            rover = new Rover();   
            
            for(int j = 0; j < squad.size(); j++){
                while(rover.getPoint() == squad.get(j).getPoint() && 
                        rover.getDirection() == squad.get(j).getDirection()){
                    
                    rover = new Rover();
                    j = -1;
                }
            }
            squad.add(rover);                      
        }
        System.out.println("Deploying Rovers...");
    }
    
    /**
     * Returns a list of formatted rover information
     * @return Rover information
     */
    private static List<String> getRoverInfo(){
        List<String> roverInfo = new ArrayList<>(); 
        for(int i = 0; i < squad.size(); i++){
            int x = squad.get(i).getPoint().x;
            int y = squad.get(i).getPoint().y;
            char dir = squad.get(i).getDirection();
            roverInfo.add( x + " " + y + " " + dir);   
        }
        return roverInfo;
    }
    
    /**
     * Prints rover information
     */
    protected static void showRoverInfo(){
        for(int i = 0; i < squad.size(); i++){
            int x = squad.get(i).getPoint().x;
            int y = squad.get(i).getPoint().y;
            char dir = squad.get(i).getDirection();
            System.out.println("Rover " + (i+1) + ":\t" + x + " " + y + " " + dir); 
        }
    }
    
    /**
     * Returns user input after checking it for validity
     * @param statement Context of what the user must enter
     * @param code The type of user input
     * @return Valid user input
     */
    protected static String getValidInput(String statement, FORMAT_CODE code){
        Scanner scanner = new Scanner(System.in); 
        String input;
        boolean err = false;
        while(true){
            if(err){
               System.out.println("INVALID INPUT! PLEASE TRY AGAIN!"); 
            }
            System.out.println(statement);
            input = scanner.nextLine();    
            
            // Removes trailing/leading whitespace and splits string into and array
            input = input.trim();
            String[] info = input.split(" ");
            
            // Confirmation, max. coordinate and rover info format check
            if((code == FORMAT_CODE.CONFIRMATION && info.length == 1) || 
                    (code == FORMAT_CODE.MAXCOORDINATE && info.length == 2) || 
                    (code == FORMAT_CODE.ROVER_INFO && info.length == 3)){
                try{
                    Integer.parseInt(info[0]);
                    if(code != FORMAT_CODE.CONFIRMATION){
                        Integer.parseInt(info[1]);
                    }
                }catch(Exception ex){
                    err = true;
                    continue;
                }
                if(code == FORMAT_CODE.MAXCOORDINATE){
                    // Checks if the max. coordinates are below the minimum allowed
                    if(Integer.parseInt(info[0]) < plateau.getMaxCoord()[0] || 
                            Integer.parseInt(info[1]) < plateau.getMaxCoord()[1]){
                        err = true;
                        continue;
                    }
                }
                if(code == FORMAT_CODE.CONFIRMATION){
                    // Checks if the input is a 1 or 2
                    if(Integer.parseInt(info[0]) < 1 || Integer.parseInt(info[0]) > 2){
                        err = true;
                        continue;
                    }
                }
                else if(code == FORMAT_CODE.ROVER_INFO){
                    // Checks if the rover orientation is valid
                    char[] dir = info[2].toCharArray();
                    if(dir.length != 1 || ('N' != dir[0] && 'E' != dir[0] 
                            && 'S' != dir[0] && 'W' != dir[0])){
                        err = true;
                        continue;
                    }
                }
            }
            // Instruction format check
            else if(code == FORMAT_CODE.INSTRUCTION){
                boolean instrnErr = false;
                char[] instrnArr = input.toCharArray();
                for(int i = 0; i < instrnArr.length; i ++){
                    if(instrnArr[i] != 'M' && instrnArr[i] != 'L' && instrnArr[i] != 'R'){
                        instrnErr = true;
                    }
                }
                if(instrnErr){
                    err = true;
                    continue;
                }
            }
            else{
                err = true;
                continue;
            }
            break;           
        }
        return input;
    }
    
    /**
     * Prints formatted user input
     * @param input User input (Max. coordinate, rovers and instructions)
     */
    private static void showFormattedInput(List<String> input){
        System.out.println("\nInput:\n");
        for(int i = 0; i < input.size(); i++){
            System.out.println(input.get(i) + "\n");
        }
    }
    
    /**
     * Prints formatted output
     */
    private static void showFormattedOuput(){
        System.out.println("\nOuput:\n");
        for(int i = 0; i < outputOrder.size(); i++){
            int x = squad.get(outputOrder.get(i)).getPoint().x;
            int y = squad.get(outputOrder.get(i)).getPoint().y;
            char dir = squad.get(outputOrder.get(i)).getDirection();
            System.out.println(x + " " + y +  " " + dir + "\n");
        }
    }
   
    /**
     * Finds the largest x and y coordinates of the rovers
     * to determine minimum largest coordinate
     * @return Minimum x and y values for the upper-right 
     * coordinate of the plateau
     */
    protected static int[] findMaxCoord(){
        int[] maxCoord = {5, 5};
        
        for(int i = 0; i < squad.size(); i++){
            if(squad.get(i).getPoint().x > maxCoord[0]){
                maxCoord[0] = squad.get(i).getPoint().x;
            }
            if(squad.get(i).getPoint().y > maxCoord[1]){
                maxCoord[1] = squad.get(i).getPoint().y;
            }
        }
        return maxCoord;
    }
    
}
