import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaRDD;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class DecisionTreeSpark {
    public static void main(String[] args) {
        // Set up Spark
        SparkConf conf = new SparkConf().setAppName("DecisionTreeSpark").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("DecisionTreeSpark").getOrCreate();

        // Load the dataset
        String datasetPath = "app/TrainingDataset.csv";
        Dataset<Row> data = spark.read().option("header", "true").option("delimiter", ";").csv(datasetPath);

        // Convert categorical labels into numerical labels
        StringIndexer labelIndexer = new StringIndexer().setInputCol("quality").setOutputCol("label");
        data = labelIndexer.fit(data).transform(data);

        // Combine feature columns into a single vector column
        String[] featureColumns = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"};
        VectorAssembler assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features");
        data = assembler.transform(data);

        // Split the data into training and testing sets
        Dataset<Row>[] splits = data.randomSplit(new double[]{0.8, 0.2}, 42);
        Dataset<Row> trainingData = splits[0];
        Dataset<Row> testData = splits[1];

        // Create a Decision Tree classifier
        DecisionTreeClassifier dt = new DecisionTreeClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features");

        // Train the model
        DecisionTreeClassificationModel model = dt.fit(trainingData);

        // Make predictions on the test set
        Dataset<Row> predictions = model.transform(testData);

        // Evaluate the model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score: " + f1);

        // Stop Spark
        spark.stop();
    }
}
