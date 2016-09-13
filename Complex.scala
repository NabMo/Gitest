//spark-shell --packages com.databricks:spark-csv_2.11:1.2.0
import org.apache.spark.mllib.linalg.Vectors
import org.apache.spark.mllib.clustering.KMeans
import org.apache.spark.sql.functions._


val data = sc.textFile("ComplexDataTop1000.csv")
val header = data.first
val rows = data.filter(l => l != header)

case class CC1(FACTResultID: String, MakeID: Int, AgeID: Int, MileageID: Int, CylinderCapacityID: Int, Result: Int)

val allSplit = rows.map(line => line.split(","))

val allData = allSplit.map( p => CC1( p(0).toString, p(1).toInt, p(2).trim.toInt, p(3).trim.toInt, p(4).trim.toInt, p(5).trim.toInt))

// convert rdd to dataframe
val allDF = allData.toDF()

// convert back to rdd and cache the data
val rowsRDD = allDF.rdd.map(r => (r.getString(0), r.getInt(1), r.getInt(2), r.getInt(3), r.getInt(4), r.getInt(5)))
rowsRDD.cache()


// convert data to RDD which will be passed to KMeans and cache the data
val vectors = allDF.rdd.map(r => Vectors.dense( r.getInt(1), r.getInt(2), r.getInt(3), r.getInt(4), r.getInt(5)))
vectors.cache()


//KMeans model with 3 clusters and 20 iterations
val kMeansModel = KMeans.train(vectors, 3, 20)

//Print the center of each cluster
kMeansModel.clusterCenters.foreach(println)

// Get the prediction from the model with the ID so we can link them back to other information
val predictions = rowsRDD.map{r => (r._1, kMeansModel.predict(Vectors.dense(r._2, r._3, r._4, r._5, r._6) ))}

// convert the rdd to a dataframe
val predDF = predictions.toDF("FACTResultID", "CLUSTER")


//predDF.show()
//allDF.show()

// join the dataframes on ID (spark 1.4.1)
val t = allDF.join(predDF, "FACTResultID")


//Export
t.write.format("com.databricks.spark.csv").save("clusteringResults.csv")


// review a subset of each cluster
t.filter("CLUSTER = 0").show()
t.filter("CLUSTER = 1").show()
t.filter("CLUSTER = 2").show()

// get descriptive statistics for each cluster
t.filter("CLUSTER = 0").describe().show()
t.filter("CLUSTER = 1").describe().show()