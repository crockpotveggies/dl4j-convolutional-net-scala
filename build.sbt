name := "deeplearning4j-examples-scala"

version := "1.0"

scalaVersion := "2.10.4"

lazy val root = project.in(file("."))

libraryDependencies ++= Seq(
  "commons-io" % "commons-io" % "2.4",
  "com.google.guava" % "guava" % "19.0",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.4-rc3.8",
  "org.deeplearning4j" % "deeplearning4j-nlp" % "0.4-rc3.8",
  "org.deeplearning4j" % "deeplearning4j-ui"  % "0.4-rc3.8",
  "org.nd4j"          % "nd4j-x86"            % "0.4-rc3.8",
  "org.nd4j" % "canova-nd4j-codec" % "0.0.0.14",
  "org.nd4j" % "canova-nd4j-image" % "0.0.0.14",
  "org.jblas" % "jblas" % "1.2.4",
  "com.twelvemonkeys.imageio" % "imageio-core" % "3.1.2",
  "com.twelvemonkeys.common" % "common-lang" % "3.1.2"
)


resolvers += "Sonatype OSS Snapshots" at "https://oss.sonatype.org/content/repositories/snapshots"

resolvers += "Sonatype release Repository" at "http://oss.sonatype.org/service/local/staging/deploy/maven2/"

resolvers += "Local Maven Repository" at "file:///"+Path.userHome.absolutePath+"/.m2/repository/"
