# WineQualityPrediction
The provided Java-based Spark application demonstrates the implementation of a Decision Tree classifier using Spark MLlib for training and testing. Below are the steps to follow for running the code:

# Prerequisites:
Apache Spark: Ensure that Apache Spark is installed and configured. Download Spark from here.

Java Development Kit (JDK): Make sure you have Java installed. Download the JDK from Oracle.

# Project Structure:
src/: Contains Java source code files.

DecisionTreeSpark.java: Code for training a Decision Tree model and evaluating it on a test dataset.
TestDecisionTreeModel.java: Code for loading a pre-trained Decision Tree model and testing it on a validation dataset.
data/: Placeholder for dataset files.

TrainingDataset.csv: Training dataset in CSV format.
ValidationDataset.csv: Validation dataset in CSV format.
model/: Placeholder for storing the trained Decision Tree model.

# Running the Code:
Compile the Java code:

bash
Copy code
javac -cp "path/to/spark/jars/*" src/*.java
Run the training and testing code:

bash
Copy code
spark-submit --class DecisionTreeSpark --master local[*] --driver-class-path "path/to/spark/jars/*" src/DecisionTreeSpark.java
Run the pre-trained model testing code:

bash
Copy code
spark-submit --class TestDecisionTreeModel --master local[*] --driver-class-path "path/to/spark/jars/*" src/TestDecisionTreeModel.java


Update dataset paths in the code if your datasets are in a different directory.

Update the model path in TestDecisionTreeModel.java to point to the directory where the trained model is saved.

Adjust column names and configurations based on your actual dataset structure.
