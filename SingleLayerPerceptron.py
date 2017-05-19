from numpy.random import random_sample

class Perceptron(object):

    def __init__(self, no_of_inputs):
        """
        Initialises the weights with random values
        Sets the learning rate / adaptatin rate
        """
        self.w = random_sample(no_of_inputs + 1) # X,Y + bias
        self.lr = 0.001
        self.bias = float(1)

    def weight_adjustment(self, inputs, error):
        """
        Adjusts the weights in self.w
        @param inputs a list of the input values used
        @param error the difference between desired and calculated
        """
        for x in range(len(inputs)):
            # Adjust the input weights
            self.w[x] = self.w[x] + (self.lr * inputs[x] * error)

        # Adjust the bias weight (the last weight in self.w)
        self.w[-1] = self.w[-1] + (self.lr * error)

    def result(self, inputs):
        """
        @param inputs one set of data
        @returns the the sum of inputs multiplied by their weights
        """
        value = 0.0
        for x in range(len(inputs)):
            # Add the value of the inputs
            value += inputs[x] * self.w[x]

        # Add the value of bias
        value += self.bias * self.w[-1]

        return value

    def recall(self, inputs):
        res = self.result(inputs)
        if res > 0.5:return 1
        else: 
            return 0
    
def normalise(data):
    #to make the number as float
    temp_list = []
    for entry in data:
        entry_list = []
        for value in entry[0]:
            entry_list.append(float(value))
        temp_list.append([entry_list, entry[1]])

    return temp_list

LAE = 0.001
Eps = 0.02

def main(data):
    
    training_data = normalise(data)
    # Create the perceptron
    p = Perceptron(len(data[0][0]))

    # Number of full iterations
    iteration_no = 0

    # Instantiate ave for the loop
    ave =999

    while (abs(ave-LAE) > Eps):

        error = 0

        # For each set in the training_data

        for value in training_data:

            # Calculate the result
            output = p.result(value[0])

            # Calculate the error
            iter_error = value[1] - output

            # Add the error to the epoch error
            error += iter_error

            # Adjust the weights based on inputs and the error
            p.weight_adjustment(value[0], iter_error)

        # Calculate the AE - epoch error / number of sets
        ave = float(error/len(training_data))

        # Print the AE for each iteration
        print ("Average Error of %d iter's is %.10f and Weights: 0: %.10f - 1: %.10f - Bias: %.10f" % (iteration_no, ave,p.w[0], p.w[1], p.w[2]))

        # Increment the epoch number
        iteration_no += 1

    return p


data = [#((X, Y), CLASSIFICATION)
        [[1,5], 0],
        [[3,3], 0],
        [[3,1], 0],
        [[2,2], 0],
        [[1,10], 0],
        [[1,1], 0],
        [[1,2], 0],
        [[2,8], 0],
        [[2,4], 0],
        [[6,0], 1],
        [[7,1], 1],
        [[8,1], 1],
        [[8,2], 1],
        [[9,2], 1],
        [[10,3],1],
        [[10,1],1],
        [[10,4],1],
        [[7,2],1]
    ]


main(data)
