// Your sbt build file. Guides on how to write one can be found at
// http://www.scala-sbt.org/0.13/docs/index.html

scalaVersion := "2.10.6"

sparkVersion := "1.6.1"

crossScalaVersions := Seq("2.10.6", "2.11.8")

spName := "yu-iskw/spark-parallelized-sgd"

// Don't forget to set the version
version := "0.0.2"

// All Spark Packages need a license
licenses := Seq("Apache-2.0" -> url("http://opensource.org/licenses/Apache-2.0"))

sparkComponents ++= Seq("mllib", "streaming", "sql", "graphx")

spAppendScalaVersion := true

spIncludeMaven := true

spIgnoreProvided := true

val testSparkVersion = settingKey[String]("The version of Spark to test against.")

testSparkVersion := sys.props.getOrElse("spark.testVersion", sparkVersion.value)

// Can't parallelly execute in test
parallelExecution in Test := false

fork in Test := true

javaOptions ++= Seq("-Xmx2G", "-XX:MaxPermSize=256m")

libraryDependencies ++= Seq(
  "org.scalatest" %% "scalatest" % "2.1.5" % "test",
  "org.apache.spark" %% "spark-core" % testSparkVersion.value,
  "org.apache.spark" %% "spark-yarn" % testSparkVersion.value,
  "org.apache.spark" %% "spark-mllib" % testSparkVersion.value,
  "org.apache.spark" %% "spark-streaming" % testSparkVersion.value,
  "org.apache.spark" %% "spark-streaming-kafka" % testSparkVersion.value,
  "org.apache.spark" %% "spark-sql" % testSparkVersion.value,
  "org.apache.spark" %% "spark-catalyst" % testSparkVersion.value,
  "org.apache.spark" %% "spark-hive" % testSparkVersion.value,
  "org.apache.spark" %% "spark-graphx" % testSparkVersion.value,
  "org.apache.spark" %% "spark-repl" % testSparkVersion.value
)

resolvers ++= Seq(
  "JBoss Repository" at "http://repository.jboss.org/nexus/content/repositories/releases/",
  "Spray Repository" at "http://repo.spray.cc/",
  "Cloudera Repository" at "https://repository.cloudera.com/artifactory/cloudera-repos/",
  "Akka Repository" at "http://repo.akka.io/releases/",
  "Twitter4J Repository" at "http://twitter4j.org/maven2/",
  "Apache HBase" at "https://repository.apache.org/content/repositories/releases",
  "Twitter Maven Repo" at "http://maven.twttr.com/",
  "scala-tools" at "https://oss.sonatype.org/content/groups/scala-tools",
  "Typesafe repository" at "http://repo.typesafe.com/typesafe/releases/",
  "Second Typesafe repo" at "http://repo.typesafe.com/typesafe/maven-releases/",
  "Mesosphere Public Repository" at "http://downloads.mesosphere.io/maven"
)

mergeStrategy in assembly := {
  case m if m.toLowerCase.endsWith("manifest.mf") => MergeStrategy.discard
  case m if m.toLowerCase.matches("meta-inf.*\\.sf$") => MergeStrategy.discard
  case "log4j.properties" => MergeStrategy.discard
  case m if m.toLowerCase.startsWith("meta-inf/services/") => MergeStrategy.filterDistinctLines
  case "reference.conf" => MergeStrategy.concat
  case _ => MergeStrategy.first
}

ScoverageSbtPlugin.ScoverageKeys.coverageHighlighting := {
  if (scalaBinaryVersion.value == "2.10") false
  else true
}
