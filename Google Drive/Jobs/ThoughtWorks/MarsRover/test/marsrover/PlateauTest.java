/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */
package marsrover;

import java.util.List;
import org.junit.After;
import org.junit.AfterClass;
import org.junit.Before;
import org.junit.BeforeClass;
import org.junit.Test;
import static org.junit.Assert.*;

/**
 *
 * @author Kieran
 */
public class PlateauTest {
    
    public PlateauTest() {
    }
    
    @BeforeClass
    public static void setUpClass() {
    }
    
    @AfterClass
    public static void tearDownClass() {
    }
    
    @Before
    public void setUp() {
    }
    
    @After
    public void tearDown() {
    }

    /**
     * Test of printGrid method, of class Plateau.
     */
    @Test
    public void testPrintGrid() {
        System.out.println("printGrid");
        List<String> roverInfo = null;
        Plateau instance = new Plateau();
        instance.printGrid(roverInfo);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of getMaxX method, of class Plateau.
     */
    @Test
    public void testGetMaxX() {
        System.out.println("getMaxX");
        Plateau instance = new Plateau();
        int expResult = 0;
        int result = instance.getMaxX();
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of setMaxX method, of class Plateau.
     */
    @Test
    public void testSetMaxX() {
        System.out.println("setMaxX");
        int maxX = 0;
        Plateau instance = new Plateau();
        instance.setMaxX(maxX);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of getMaxY method, of class Plateau.
     */
    @Test
    public void testGetMaxY() {
        System.out.println("getMaxY");
        Plateau instance = new Plateau();
        int expResult = 0;
        int result = instance.getMaxY();
        assertEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of setMaxY method, of class Plateau.
     */
    @Test
    public void testSetMaxY() {
        System.out.println("setMaxY");
        int maxY = 0;
        Plateau instance = new Plateau();
        instance.setMaxY(maxY);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of getMaxCoord method, of class Plateau.
     */
    @Test
    public void testGetMaxCoord() {
        System.out.println("getMaxCoord");
        Plateau instance = new Plateau();
        int[] expResult = null;
        int[] result = instance.getMaxCoord();
        assertArrayEquals(expResult, result);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }

    /**
     * Test of setMaxCoord method, of class Plateau.
     */
    @Test
    public void testSetMaxCoord() {
        System.out.println("setMaxCoord");
        int[] maxCoord = null;
        Plateau instance = new Plateau();
        instance.setMaxCoord(maxCoord);
        // TODO review the generated test code and remove the default call to fail.
        fail("The test case is a prototype.");
    }
    
}
