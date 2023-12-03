# Use an OpenJDK runtime as the base image
FROM openjdk:8-jre

# Set the working directory
WORKDIR /app

# Copy the Spark job files into the container
COPY ParallelModelTraining.java /app
COPY PredictionApplication.java /app

# Set up Spark environment variables
ENV SPARK_HOME /path/to/spark
ENV PATH $SPARK_HOME/bin:$PATH

# Build the Spark job files
RUN javac -classpath "$SPARK_HOME/jars/*" ParallelModelTraining.java
RUN javac -classpath "$SPARK_HOME/jars/*" PredictionApplication.java

# Command to run the Spark job for parallel model training
CMD ["java", "-classpath", "$SPARK_HOME/jars/*:.","ParallelModelTraining"]

