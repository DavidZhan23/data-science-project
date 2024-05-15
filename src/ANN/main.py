import subprocess

def main():
    print("Running tune.py to find the best hyperparameters...")
    subprocess.run(["python", "tune.py"])
    
    print("\nRunning random_forest.py to train and evaluate the Random Forest models...")
    subprocess.run(["python", "random_forest.py"])

if __name__ == "__main__":
    main()
