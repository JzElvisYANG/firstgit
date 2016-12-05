from pyspark import SparkContext, SparkConf
conf = SparkConf().setAppName("Myapp").setMaster("local")
sc = SparkContext(conf = conf)
import math

#read the wikipedia articles from the provided file, producing an RDD.
doc1 = sc.textFile("a.txt")
doc2 = sc.textFile("b.txt")

#calculate TF scores from each document
words1 = doc1.flatMap(lambda l: l.split(" "))
filteredWords1 = words1.filter(lambda x: x != '' and not x.startswith('\\'))
wordCount1 = filteredWords1.map(lambda x: (x,1)).reduceByKey(lambda x, y: x+y)

words2 = doc2.flatMap(lambda l: l.split(" "))
filteredWords2 = words2.filter(lambda x: x != '' and not x.startswith('\\'))
wordCount2 = filteredWords2.map(lambda x: (x,1)).reduceByKey(lambda x, y: x+y)

#count the number of documents each term appears in
unionSet = wordCount1.union(wordCount2)
termCount = unionSet.reduceByKey(lambda x, y: x+y)
print(termCount.collect())
#print(wordCount1.collect())
#print(wordCount2.collect())

#calculate DF(t,D)
keysSet = wordCount1.keys().union(wordCount2.keys())
keysCount = keysSet.map(lambda x: (x, 1)).reduceByKey(lambda x, y: x + y)
print("-------------DF(t,D)---------------")
print(keysCount.collect())

#calculate IDF(t,D)
iDf = keysCount.map(lambda x:(x[0], math.log((2 + 1) / (x[1] + 1))))
print("-------------IDF(t,D)--------------")
print(iDf.collect())

#calculate TFIDF(t,D)
tFIDFSet1 = wordCount1.union(iDf)
tFIDF1 = tFIDFSet1.reduceByKey(lambda x, y: x * y)

tFIDFSet2 = wordCount2.union(iDf)
tFIDF2 = tFIDFSet2.reduceByKey(lambda x, y: x * y)

print("-------------TFIDF(t,D)------------")
print(tFIDF1.collect())
print(tFIDF2.collect())
