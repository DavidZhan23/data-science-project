import subprocess
import os

def main():
    # Define the paths
    data_file = '../../cleaned_dataset/data4training.csv'
    ann_script = 'neural_network.py'
    visualize_script = 'visualize_nn_results.py'
    correlation_dir = 'correlation_matrix'
    
    # Check if the data file exists
    if not os.path.exists(data_file):
        print(f"Data file not found: {data_file}")
        return
    
    # Ensure the correlation_matrix directory exists
    if not os.path.exists(correlation_dir):
        os.makedirs(correlation_dir)

    # Run the neural network training script
    print("Running neural_network.py to train the neural network models...")
    subprocess.run(["python", ann_script])

    # Run the visualization script
    print("\nRunning visualize_nn_results.py to generate and save the confusion matrices...")
    subprocess.run(["python", visualize_script])

    print("\nAll processes completed successfully.")

if __name__ == "__main__":
    main()
