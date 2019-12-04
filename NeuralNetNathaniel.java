

/**
 * @author Nathaniel E. Russell
 * NeuralNetNathaniel
 * 2019-2-23
 */

public class NeuralNetNathaniel
{
    
    // Constants once initialized
    private final int[] LAYERAGE;
    private final double LEARN_RATE;
    private final int EPOCHS;
    private final double[][] TRAINING_DATA;
    
    
    // Neural Network Pieces
    private double[][] NODES;
    private double[][] BIASES;
    private double[][][] WEIGHTS;
    
    // Error used in  back propagation
    private double[][] ERROR;
    
    
    /**
     * Constructor:
     * Sets constant variables
     * Initializes NODES , WEIGHTS , and BIASES
     * Randomizes WEIGHTS and BIASES
     * calls train() in order to train network
     * 
     * @param layerage       (An array with the number of nodes for each layer) 
     * (e.i. {2,5,3} is 2-input_nodes , 5-hidden_nodes , 3-output_nodes)
     * 
     * @param learningRate   (Decimal value for LEARN_RATE)
     * 
     * @param epochs         (Number of times the network will train itself)
     * 
     * @param trainingData   (The data you want to train the neural network on)
     * (2D array, if LAYERAGE is {2,5,3} then each column of training data should look like:)
     *       { input , input , desired_out , desired_out , desired_out }
     */
    public NeuralNetNathaniel(int[] layerage , double learningRate , int epochs , double[][] trainingData)
    {
        LAYERAGE = layerage;
        int max = max(layerage);
        
        LEARN_RATE = learningRate;
        EPOCHS = epochs;
        TRAINING_DATA = trainingData;
        
        
        NODES  = new double[LAYERAGE.length][max];
        BIASES = new double[LAYERAGE.length][max];
        ERROR  = new double[LAYERAGE.length][max];
        reset(NODES);
        reset(ERROR);
        randomizeBiases();
        
        
        WEIGHTS = new double[LAYERAGE.length - 1][max][max];
        randomizeWeights();
        
        
        train();
        
    }
    
    
    /**
     * Constructor:
     * Sets constant variables
     * Initializes NODES , WEIGHTS , and BIASES
     * Randomizes WEIGHTS and BIASES
     * 
     * @param layerage       (An array with the number of nodes for each layer) 
     * (e.i. {2,5,3} is 2-input_nodes , 5-hidden_nodes , 3-output_nodes)
     * 
     * @param learningRate   (Decimal value for LEARN_RATE)
     */
    public NeuralNetNathaniel(int[] layerage , double learningRate)
    {
        LAYERAGE = layerage;
        int max = max(layerage);
        
        LEARN_RATE = learningRate;
        
        TRAINING_DATA = null;
        EPOCHS = 0;
        
        
        NODES  = new double[LAYERAGE.length][max];
        BIASES = new double[LAYERAGE.length][max];
        ERROR  = new double[LAYERAGE.length][max];
        reset(NODES);
        reset(ERROR);
        randomizeBiases();
        
        
        WEIGHTS = new double[LAYERAGE.length - 1][max][max];
        randomizeWeights();
        
    }
    
    
    /**
     * After having been trained use output() to forward propagate on fully trained network
     * @param data   (data for input nodes to let run through the neural network)
     * @param round  (if true the output nodes are rounded to 0.0 or 1.0)
     */
    public void output(double[][] data , boolean round)
    {
        for (int i = 0; i < data.length; i++)
        {
            forwardPropagate(data[i]);
            
            System.out.println("INPUTS\n-----------------------------");
            for (int j = 0; j < LAYERAGE[0]; j++)
            {
                System.out.println( (!round) ? NODES[0][j] : Math.round(NODES[0][j])  );
            }
            System.out.println("\nOUTPUTS\n-----------------------------");
            for (int j = 0; j < LAYERAGE[LAYERAGE.length - 1]; j++)
            {
                System.out.println( (!round) ? NODES[NODES.length - 1][j] : Math.round(NODES[NODES.length - 1][j]) );
            }
            System.out.println("\n\n");
        }
    }
    
    
    /**
     * Train the neural network on TRAINING_DATA, for EPOCHS number of repetitions
     */
    public void train()
    {
        System.out.println("[-TRAINING-]");
        
        for (int i = 0; i < EPOCHS; i++)
        {
            // forward propagate then back propagate
            forwardPropagate(TRAINING_DATA[i % TRAINING_DATA.length]);
            backPropagate(TRAINING_DATA[i % TRAINING_DATA.length]);
            
            
            // Print out a loading bar for every 5% of progress
            if (i % (EPOCHS / 20) == 0)
            {
                System.out.print("[" );
                for (int j = 0; j < i / (EPOCHS / 20); j++)
                {
                    System.out.print("=");
                }
                for (int j = i / (EPOCHS / 20); j < 20; j++)
                {
                    System.out.print(".");
                }
                System.out.println("] " + (5 * i / (EPOCHS / 20)) + "%" );
            }
        }
        System.out.println("[====================] 100%");
        System.out.println("[------COMPLETE------]");
    }
    
    
    /**
     * One individual training epoch with data passed in
     * @param data (Training data)
     */
    public void trainIndividual( double data[] )
    {
        forwardPropagate(data);
        backPropagate(data);
    }
    
    
    /**
     * Set input NODES to input data
     * Forward propagate through until all of NODES are filled with data
     * @param data ( can be either training data or new testing data )
     */
    public void forwardPropagate(double[] data)
    {
        // set all nodes to 0
        reset(NODES);
        int weight = 0;
        
        
        // fill inputs
        for (int i = 0; i < LAYERAGE[0]; i++)
        {
            NODES[0][i] = data[i];
        }
        
        
        // propagate remaining layers
        for (int layer = 1; layer < LAYERAGE.length; layer++)
        {
            weight = layer - 1;
            for ( int front = 0; front < LAYERAGE[layer]; front++)
            {
                for (int back = 0; back < LAYERAGE[layer - 1]; back++)
                {
                    // this_NODE += (Each node one layer back) * (The weight connecting it to this_NODE)
                    NODES[layer][front] += NODES[layer - 1][back] * WEIGHTS[weight][back][front];
                }
                // Once this_NODE is filled, add the weight and run it through the sigmoid function
                NODES[layer][front] = sigmoid(NODES[layer][front] + BIASES[layer][front]);
                
            } // end single node change
        } // end changing layer of nodes
         
        
    } // end forwardPropagate
    
    
    /**
     * Generate's each nodes error starting with output layer and going backwards
     * Then adjusts the weights and biases accordingly
     * @param data ( this is the training data, used to derive both inputs and desired outputs )
     */
    private void backPropagate(double[] data)
    {
        reset(ERROR);
        int weight = 0;
        
        
        // Output Error
        for (int i = 0; i < LAYERAGE[LAYERAGE.length - 1]; i++)
        {
            ERROR[ERROR.length - 1][i] = 
                    (data[data.length - LAYERAGE[LAYERAGE.length - 1] + i] - NODES[LAYERAGE.length - 1][i]) 
                    * sigmoidDerivative(NODES[LAYERAGE.length - 1][i]);
        }
        
        // Remaining Layer's Error
        for (int layer = LAYERAGE.length - 2; layer > 0; layer--)
        {
            weight = layer - 1;
            for ( int back = 0; back < LAYERAGE[layer]; back++)
            {
                for (int front = 0; front < LAYERAGE[layer + 1]; front++)
                {
                    ERROR[layer][back] += ERROR[layer + 1][front] * WEIGHTS[weight][back][front];
                }
                ERROR[layer][back] *= sigmoidDerivative(NODES[layer][back]);
            }// end each node's error
        }// end each layer
        
        // Update Weights
        for (int layer = 0; layer < LAYERAGE.length - 1; layer++)
        {
            for (int back = 0; back < LAYERAGE[layer]; back++)
            {
                for (int front = 0; front < LAYERAGE[layer + 1]; front++)
                {
                    WEIGHTS[layer][back][front] += (LEARN_RATE * ERROR[layer + 1][front] * NODES[layer][back]);
                }
            }
        }
        
        // Update Biases
        for (int layer = 0; layer < LAYERAGE.length - 1; layer++)
        {
            for (int node = 0; node < LAYERAGE[layer]; node++)
            {
                BIASES[layer][node] += LEARN_RATE * ERROR[layer][node];
            }
        }
        
    }
    
    
    /**
     * Get the results of the output nodes for one swing through of the input nodes
     * @param data (inputs for the input nodes)
     * @return {array} (the layer of output nodes after propagating the data)
     */
    public double[] getResults( double data[] )
    {
        forwardPropagate(data);
        return NODES[NODES.length - 1];
    }
    
    
    /**
     * Used to find the largest number of nodes in a single layer
     * @param array (layerage is passed in)
     * @return {number} (the maximum value of the array)
     */
    private int max( int[] array )
    {
        int max = array[0];
        for (int i = 0; i < array.length; i++)
        {
            if (max < array[i])
            {
                max = array[i];
            }
        }
        return max;
    }
    
    
    /**
     * Used to set all values of an array to 0
     * To be used to reset ERROR and NODES
     * @param array (all values are 0)
     */
    private void reset( double[][] array )
    {
        for (int i = 0; i < array.length; i++)
        {
            for (int j = 0; j < array[i].length; j++)
            {
                array[i][j] = 0;
            }
        }
    }
    
    
    /**
     * When originally creating weights this randomizes them, [0-1)
     */
    private void randomizeWeights()
    {
        for (int i = 0; i < WEIGHTS.length; i++)
        {
            for (int j = 0; j < LAYERAGE[i]; j++)
            {
                for (int k = 0; k < LAYERAGE[i + 1]; k++)
                {
                    WEIGHTS[i][j][k] = Math.random();
                }
            }
        }
    }
    
    
    /**
     * When originally creating biases this randomizes them, [0-1)
     */
    private void randomizeBiases()
    {
        for (int i = 0; i < LAYERAGE.length; i++)
        {
            for (int j = 0; j < LAYERAGE[i]; j++)
            {
                BIASES[i][j] = Math.random();
            }
        }
    }
    
    
    /**
     * Activation function for forward propagation
     * @param z
     * @return {number} (1 / (1 + e^-z))
     */
    private double sigmoid(double z)
    {
        return 1.0 / ( 1.0 + Math.pow( 2.718 , -1.0 * z));
    }
    
    
    /**
     * Function used when back propagating
     * @param z
     * @return {number} (z * (1 - z))
     */
    private double sigmoidDerivative( double z )
    {
        return (z * (1.0 - z));
    }
    
    
    
    
    
    
    

}
