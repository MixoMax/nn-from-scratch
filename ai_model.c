/*
@Author: Linus Horn
@Contact: <linus@linush.org> || <linus.horn@uni-rostock.de>
@Created on: 2024-11-28
*/

/*
ai_model.c

A very simple implementation of a neural network with Linear layers.
*/

/*
Python code:
import random
import json
import time

class NeuralNetwork:
    def __init__(self, input_size: int = 784, hidden_layers = [512, 512], output_size: int = 10):
        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size
        self.weights = []
        self.biases = []

        # Input to hidden Layers Network
        
        # self.weights.append(0.01 * np.random.randn(input_size, hidden_layers[0]))
        self.weights.append([[0.01 * random.gauss(0, 1) for _ in range(hidden_layers[0])] for _ in range(input_size)])


        # self.biases.append(np.zeros(output_size))
        self.biases.append([0 for _ in range(output_size)])

    
    def forward(self, inputs: list[float]):
        layers = [inputs]

        for i in range(len(self.weights)):
            # Dot product

            # layers.append(np.dot(layers[-1], self.weights[i]) + self.biases[i])
            layers.append([sum(layers[-1][j] * self.weights[i][j][k] for j in range(len(layers[-1]))) + self.biases[i][k] for k in range(len(self.biases[i]))])
        
        return layers[-1]

    def _from_pyt_state_dict(self, json_path: str):
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        self.weights = []
        self.biases = []

        # we need to rotate the weights
        # a pytorch layedr Linear(a,b) has len(data[key]) = b and len(data[key][0]) = a
        # but in our case, we need to have len(data[key]) = a and len(data[key][0]) = b
        
        # json is like this: {"model.0.weight": [[], [], ...], "model.0.bias": [], "model.1.weight": [], "model.1.bias": [], ...}
        for key in data:
            if 'weight' in key:
                d = data[key]
                # rotate the matrix
                # d = np.array(d).T.tolist()
                d = list(map(list, zip(*d)))
                self.weights.append(d)
            elif 'bias' in key:
                d = data[key]
                self.biases.append(d)
            else:
                raise ValueError('Invalid key')
        
        # reset the input size, hidden layers, and output size to the ones in the model
        self.input_size = len(self.weights[0])
        self.hidden_layers = [len(self.weights[i]) for i in range(1, len(self.weights)-1)]
        self.output_size = len(self.weights[-1][0])
        
        
        return self
    
    def __str__(self):
        return f'NeuralNetwork(input_size={self.input_size}, hidden_layers={self.hidden_layers}, output_size={self.output_size})'
    
    def summary(self):
        n_total_params = 0
        for i in range(len(self.weights)):
            n_total_params += len(self.weights[i]) + len(self.biases[i])
        
        print(f'Neural Network Summary\n{"-"*20}\n')
        print(f'Input Size: {self.input_size}')
        print("Layer Shapes:")
        for i in range(len(self.weights)):
            print(f'Layer {i+1}: {len(self.weights[i])}x{len(self.weights[i][0])}')
        print(f'Output Size: {self.output_size}')

model = NeuralNetwork()._from_pyt_state_dict('./model.json')
print(model)
model.summary()

sample_X = [random.random() for _ in range(784)]
t_start = time.time()
print(model.forward(sample_X))
print(f'Time: {time.time() - t_start}')
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>
#include <jansson.h>

// Dynamic 2D array structure
typedef struct {
    double** data;
    int rows;
    int cols;
} Matrix;

typedef struct {
    int input_size;
    int* hidden_layers;
    int hidden_layer_count;
    int output_size;
    Matrix* weights;
    Matrix* biases;
} NeuralNetwork;

// Function prototypes
Matrix create_matrix(int rows, int cols);
void free_matrix(Matrix* matrix);
Matrix transpose_matrix(Matrix m);
NeuralNetwork create_neural_network(int input_size, int* hidden_layers, int hidden_layer_count, int output_size);
void free_neural_network(NeuralNetwork* nn);
Matrix forward_pass(NeuralNetwork* nn, Matrix inputs);
NeuralNetwork load_from_pytorch_json(const char* json_path);
void print_neural_network_summary(NeuralNetwork* nn);

// Matrix creation and manipulation
Matrix create_matrix(int rows, int cols) {
    Matrix m;
    m.rows = rows;
    m.cols = cols;
    m.data = malloc(rows * sizeof(double*));
    for (int i = 0; i < rows; i++) {
        m.data[i] = calloc(cols, sizeof(double));
    }
    return m;
}

void free_matrix(Matrix* matrix) {
    for (int i = 0; i < matrix->rows; i++) {
        free(matrix->data[i]);
    }
    free(matrix->data);
}

Matrix transpose_matrix(Matrix m) {
    Matrix transposed = create_matrix(m.cols, m.rows);
    for (int i = 0; i < m.rows; i++) {
        for (int j = 0; j < m.cols; j++) {
            transposed.data[j][i] = m.data[i][j];
        }
    }
    return transposed;
}

// Neural Network Initialization
NeuralNetwork create_neural_network(int input_size, int* hidden_layers, int hidden_layer_count, int output_size) {
    NeuralNetwork nn;
    nn.input_size = input_size;
    nn.hidden_layers = malloc(hidden_layer_count * sizeof(int));
    memcpy(nn.hidden_layers, hidden_layers, hidden_layer_count * sizeof(int));
    nn.hidden_layer_count = hidden_layer_count;
    nn.output_size = output_size;

    // Allocate weights and biases
    nn.weights = malloc((hidden_layer_count + 1) * sizeof(Matrix));
    nn.biases = malloc((hidden_layer_count + 1) * sizeof(Matrix));

    // Initialize first layer
    nn.weights[0] = create_matrix(input_size, hidden_layers[0]);
    nn.biases[0] = create_matrix(1, hidden_layers[0]);

    // Initialize random weights
    for (int i = 0; i < nn.weights[0].rows; i++) {
        for (int j = 0; j < nn.weights[0].cols; j++) {
            nn.weights[0].data[i][j] = 0.01 * (2.0 * rand() / RAND_MAX - 1.0);
        }
    }

    return nn;
}

void free_neural_network(NeuralNetwork* nn) {
    free(nn->hidden_layers);
    for (int i = 0; i <= nn->hidden_layer_count; i++) {
        free_matrix(&nn->weights[i]);
        free_matrix(&nn->biases[i]);
    }
    free(nn->weights);
    free(nn->biases);
}

// Forward Pass
Matrix forward_pass(NeuralNetwork* nn, Matrix inputs) {
    Matrix layers[nn->hidden_layer_count + 2];
    layers[0] = inputs;

    for (int i = 0; i < nn->hidden_layer_count + 1; i++) {
        // Create output layer
        int out_rows = layers[i].rows;
        int out_cols = nn->weights[i].cols;
        layers[i+1] = create_matrix(out_rows, out_cols);

        // Compute matrix multiplication with bias
        for (int r = 0; r < out_rows; r++) {
            for (int c = 0; c < out_cols; c++) {
                double val = 0.0;
                for (int k = 0; k < layers[i].cols; k++) {
                    val += layers[i].data[r][k] * nn->weights[i].data[k][c];
                }
                layers[i+1].data[r][c] = val + nn->biases[i].data[0][c];
            }
        }
    }

    return layers[nn->hidden_layer_count + 1];
}


// Function to read the JSON file content
char* read_file(const char* filename) {
    FILE* file = fopen(filename, "r");
    if (!file) return NULL;

    fseek(file, 0, SEEK_END);
    long length = ftell(file);
    fseek(file, 0, SEEK_SET);

    char* buffer = malloc(length + 1);
    if (!buffer) {
        fclose(file);
        return NULL;
    }

    fread(buffer, 1, length, file);
    buffer[length] = '\0';
    fclose(file);
    return buffer;
}

// Function to load neural network parameters from PyTorch JSON state_dict
NeuralNetwork load_from_pytorch_json(const char* json_path) {
    char* json_content = read_file(json_path);
    if (!json_content) {
        fprintf(stderr, "Could not read JSON file\n");
        exit(1);
    }

    // Parse JSON content
    json_error_t error;
    json_t* root = json_loads(json_content, 0, &error);
    if (!root) {
        fprintf(stderr, "Error parsing JSON: %s\n", error.text);
        free(json_content);
        exit(1);
    }

    // Initialize weights and biases arrays
    size_t layer_count = 0;
    const char* key;
    json_t* value;
    json_object_foreach(root, key, value) {
        if (strstr(key, "weight")) {
            layer_count++;
        }
    }

    NeuralNetwork nn;
    nn.weights = malloc(layer_count * sizeof(Matrix));
    nn.biases = malloc(layer_count * sizeof(Matrix));
    nn.hidden_layers = malloc((layer_count - 1) * sizeof(int));
    nn.hidden_layer_count = (int)(layer_count - 1);

    nn.hidden_layer_count = (int)(layer_count - 1);
    nn.weights = malloc(layer_count * sizeof(Matrix));
    nn.biases = malloc(layer_count * sizeof(Matrix));
    nn.hidden_layers = malloc(nn.hidden_layer_count * sizeof(int));


    // Iterate over keys and load weights and biases
    size_t layer_index = 0;
    json_object_foreach(root, key, value) {
        if (strstr(key, "weight")) {
            // Load weight matrix and transpose it
            json_t* weight_array = value;
            size_t rows = json_array_size(weight_array);
            size_t cols = 0;
            if (rows > 0) {
                json_t* first_row = json_array_get(weight_array, 0);
                cols = json_array_size(first_row);
            }

            // Create matrix with transposed dimensions
            Matrix weight_matrix = create_matrix((int)cols, (int)rows);
            for (size_t i = 0; i < rows; i++) {
                json_t* row = json_array_get(weight_array, i);
                for (size_t j = 0; j < cols; j++) {
                    json_t* item = json_array_get(row, j);
                    weight_matrix.data[j][i] = json_number_value(item);
                }
            }
            nn.weights[layer_index] = weight_matrix;
        } else if (strstr(key, "bias")) {
            // Load bias vector
            json_t* bias_array = value;
            size_t size = json_array_size(bias_array);
            Matrix bias_matrix = create_matrix(1, (int)size);
            for (size_t i = 0; i < size; i++) {
                json_t* item = json_array_get(bias_array, i);
                bias_matrix.data[0][i] = json_number_value(item);
            }
            nn.biases[layer_index] = bias_matrix;
            layer_index++;
        } else {
            fprintf(stderr, "Invalid key in JSON: %s\n", key);
            json_decref(root);
            free(json_content);
            exit(1);
        }
    }

    // Set input size, hidden layers, and output size
    nn.input_size = nn.weights[0].rows;
    nn.output_size = nn.weights[layer_count - 1].cols;

    if (nn.hidden_layer_count > 0) {
        for (int i = 0; i < nn.hidden_layer_count; i++) {
            nn.hidden_layers[i] = nn.weights[i + 1].rows;
        }
    }

    json_decref(root);
    free(json_content);
    return nn;
}

void print_neural_network_summary(NeuralNetwork* nn) {
    int total_params = 0;
    printf("Neural Network Summary\n");
    printf("--------------------\n");
    printf("Input Size: %d\n", nn->input_size);
    
    printf("Layer Shapes:\n");
    for (int i = 0; i <= nn->hidden_layer_count; i++) {
        printf("Layer %d: %dx%d\n", i+1, nn->weights[i].rows, nn->weights[i].cols);
        total_params += nn->weights[i].rows * nn->weights[i].cols + nn->biases[i].cols;
    }
    
    printf("Output Size: %d\n", nn->output_size);
}

int prediction(NeuralNetwork nn, double* input_data) {
    Matrix input = create_matrix(1, nn.input_size);
    for (int i = 0; i < nn.input_size; i++) {
        input.data[0][i] = input_data[i];
    }

    Matrix output = forward_pass(&nn, input);
    int max_index = 0;
    double max_value = output.data[0][0];
    for (int i = 1; i < nn.output_size; i++) {
        if (output.data[0][i] > max_value) {
            max_value = output.data[0][i];
            max_index = i;
        }
    }

    free_matrix(&input);
    free_matrix(&output);
    return max_index;
}

int main() {
    // Seed random number generator
    srand(time(NULL));

    // Load neural network from PyTorch JSON
    NeuralNetwork nn = load_from_pytorch_json("model.json");
    print_neural_network_summary(&nn);

    
    // load test_data.json
    // test_data.json: {"X": [[double, double, ...], [double, double, ...], ...], "y": [int, int, ...]}
    char* json_content = read_file("test_data.json");

    // Parse JSON content
    json_error_t error;
    json_t* root = json_loads(json_content, 0, &error);
    if (!root) {
        fprintf(stderr, "Error parsing JSON: %s\n", error.text);
        free(json_content);
        exit(1);
    }

    // Load input data
    json_t* X = json_object_get(root, "X");
    json_t* y = json_object_get(root, "y");

    // Check if X and y are arrays
    if (!json_is_array(X) || !json_is_array(y)) {
        fprintf(stderr, "X and y must be arrays\n");
        json_decref(root);
        free(json_content);
        exit(1);
    }

    // Check if X and y have the same length
    size_t n_samples = json_array_size(X);
    if (n_samples != json_array_size(y)) {
        fprintf(stderr, "X and y must have the same length\n");
        json_decref(root);
        free(json_content);
        exit(1);
    }

    printf("Loaded %zu samples\n", n_samples);

    // Iterate over samples and make predictions
    int correct = 0;
    int total = 0;
    double t_start = (double)clock() / CLOCKS_PER_SEC;

    for (size_t i = 0; i < n_samples; i++) {
        printf("Sample %zu / %zu\n", i + 1, n_samples);
        json_t* sample = json_array_get(X, i);
        if (!json_is_array(sample)) {
            fprintf(stderr, "Sample must be an array\n");
            json_decref(root);
            free(json_content);
            exit(1);
        }

        size_t n_features = json_array_size(sample);
        double* input_data = malloc(n_features * sizeof(double));
        for (size_t j = 0; j < n_features; j++) {
            json_t* item = json_array_get(sample, j);
            input_data[j] = json_number_value(item);
        }

        int label = json_integer_value(json_array_get(y, i));
        int prediction_result = prediction(nn, input_data);
        if (prediction_result == label) {
            correct++;
        }
        total++;

        free(input_data);
    }

    double elapsed = (double)clock() / CLOCKS_PER_SEC - t_start;
    printf("Accuracy: %.2f%%\n", (double)correct / total * 100);
    printf("Time: %.2f seconds\n", elapsed);

    double predictions_per_sec = (double)total / elapsed;
    printf("Predictions per second: %.2f\n", predictions_per_sec);

    return 0;
}