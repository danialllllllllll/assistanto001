# NeuralGenetic - ANN Optimization with Genetic Algorithm

## Overview
This project implements a genetic algorithm to optimize artificial neural network (ANN) parameters for classification tasks. It uses the Fruits360 dataset to classify images of fruits (Apple Braeburn, Lemon Meyer, Mango, and Raspberry) using a neural network trained via genetic algorithm optimization.

The implementation is built from scratch using NumPy and demonstrates how genetic algorithms can be used to train neural networks without traditional backpropagation.

## Project Structure
- `Example_GA_ANN.py` - Main script that runs the genetic algorithm optimization
- `ANN.py` - Neural network implementation with forward pass and activation functions
- `ga.py` - Genetic algorithm operations (selection, crossover, mutation)
- `dataset_features.pkl` - Preprocessed feature vectors from Fruits360 dataset (360-bin histogram reduced to 102 features)
- `outputs.pkl` - Class labels for the dataset samples
- `pyproject.toml` - Python project configuration and dependencies

## Dependencies
- Python 3.12+
- NumPy 2.3.3+
- Matplotlib 3.10.6+

## How It Works
1. **Feature Extraction**: Uses 360-bin histogram of the Hue channel, reduced to 102 features via standard deviation filtering
2. **Neural Network Architecture**:
   - Input layer: 102 neurons (feature vector size)
   - Hidden layer 1: 150 neurons
   - Hidden layer 2: 60 neurons
   - Output layer: 4 neurons (4 fruit classes)
3. **Genetic Algorithm**:
   - Population size: 8 solutions
   - Parent selection: 4 best solutions
   - Generations: 20
   - Mutation rate: 10%
   - Activation function: Sigmoid

## Running the Project
Execute the main script to train the neural network:
```bash
python Example_GA_ANN.py
```

The script will:
- Load the preprocessed dataset features and labels
- Initialize random weight populations
- Run genetic algorithm for 20 generations
- Output fitness values for each generation
- Save the final weights to a pickle file
- Generate a fitness plot saved as `fitness_plot.png`

## Output Files
- `weights_<generations>_iterations_<mutation>%_mutation.pkl` - Trained network weights
- `fitness_plot.png` - Visualization of fitness progression across generations

## Recent Changes
- **2025-10-08**: Fixed NumPy 2.x compatibility issues by adding `dtype=object` to array creation
- **2025-10-08**: Configured matplotlib to use non-interactive 'Agg' backend for Replit environment
- **2025-10-08**: Updated pyproject.toml to properly handle flat layout Python modules

## References
Based on the work and tutorials by Ahmed Fawzy Gad:
- Book: "Practical Computer Vision Applications Using Deep Learning with CNNs" (Apress, 2018)
- GitHub: https://github.com/ahmedfgad/NumPyANN
- LinkedIn tutorials on ANN and GA implementation
