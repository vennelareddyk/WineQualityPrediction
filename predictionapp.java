import org.apache.spark.SparkConf;
import org.apache.spark.api.java.JavaSparkContext;
import org.apache.spark.ml.classification.DecisionTreeClassificationModel;
import org.apache.spark.ml.classification.DecisionTreeClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.StringIndexer;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class TestDecisionTreeModel {
    public static void main(String[] args) {
        // Set up Spark
        SparkConf conf = new SparkConf().setAppName("TestDecisionTreeModel").setMaster("local[*]");
        JavaSparkContext sc = new JavaSparkContext(conf);
        SparkSession spark = SparkSession.builder().appName("TestDecisionTreeModel").getOrCreate();

        // Load the validation dataset
        String validationDatasetPath = "app/ValidationDataset.csv";
        Dataset<Row> validationData = spark.read().option("header", "true").option("delimiter", ";").csv(validationDatasetPath);

        // Convert categorical labels into numerical labels
        StringIndexer labelIndexer = new StringIndexer().setInputCol("quality").setOutputCol("label");
        validationData = labelIndexer.fit(validationData).transform(validationData);

        // Combine feature columns into a single vector column
        String[] featureColumns = {"fixed acidity", "volatile acidity", "citric acid", "residual sugar",
                "chlorides", "free sulfur dioxide", "total sulfur dioxide", "density", "pH", "sulphates", "alcohol"};
        VectorAssembler assembler = new VectorAssembler().setInputCols(featureColumns).setOutputCol("features");
        validationData = assembler.transform(validationData);

        // Load the pre-trained Decision Tree model
        DecisionTreeClassificationModel model = DecisionTreeClassificationModel.load("path_to_your_model_directory");

        // Make predictions on the validation set
        Dataset<Row> predictions = model.transform(validationData);

        // Evaluate the model
        MulticlassClassificationEvaluator evaluator = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = evaluator.evaluate(predictions);
        System.out.println("F1 Score on Validation Dataset: " + f1);

        // Stop Spark
        spark.stop();
    }
}
