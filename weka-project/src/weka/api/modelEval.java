package weka.api;

import weka.core.Instance;
import weka.core.Instances;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.filters.supervised.attribute.AttributeSelection;

import weka.filters.Filter;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.core.SerializationHelper;
import weka.filters.supervised.instance.StratifiedRemoveFolds;
import weka.classifiers.Evaluation;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;


public class modelEval {
	public static void main(String args[]) throws Exception {
		
		//Defining the path for loading the dataset and path to save the model
		final String BasePath = "/home/suvanshchawla/vulnerable-methods-dataset.arff";
		final String ModelPath = "/home/suvanshchawla/classification.model";
		
		// Fetching the data from the source
		DataSource source = new DataSource(BasePath);
		Instances dataset = source.getDataSet();

		// Removing the attributes that don't contribute to the classification
		AttributeSelection Attrfilter = new AttributeSelection();
		CfsSubsetEval Cfseval = new CfsSubsetEval();
		GreedyStepwise search = new GreedyStepwise();

		search.setSearchBackwards(true);

		Attrfilter.setEvaluator(Cfseval);
		Attrfilter.setSearch(search);

		Attrfilter.setInputFormat(dataset);

		Instances newData = Filter.useFilter(dataset, Attrfilter);

		// Shuffling the order of instances
		Random random = new Random();
		newData.randomize(random);

		// set class index to the last attribute
		newData.setClassIndex(newData.numAttributes() - 1);

		// Creating the split for training and testing
		StratifiedRemoveFolds filter = new StratifiedRemoveFolds();
		String[] options = new String[6];

		options[0] = "-N"; // indicate we want to set the number of folds
		options[1] = Integer.toString(5); // split the data into five random folds
		options[2] = "-F"; // indicate we want to select a specific fold
		options[3] = Integer.toString(1); // select the first fold
		options[4] = "-S"; // indicate we want to set the random seed
		options[5] = Integer.toString(1); // set the random seed to 1

		filter.setOptions(options); // set the filter options
		filter.setInputFormat(newData); // prepare the filter for the data format
		filter.setInvertSelection(false); // do not invert the selection

		// apply filter for test data here
		Instances test = Filter.useFilter(newData, filter);

		// prepare and apply filter for training data here
		filter.setInvertSelection(true); // invert the selection to get other data
		Instances train = Filter.useFilter(newData, filter);

		// create and build the classifier - We can use different types of classifiers
		J48 tree = new J48(); // It is an implementation of C4.5 algorithm in Java
		tree.buildClassifier(train);

		SMO svm = new SMO();
		svm.buildClassifier(train);

		NaiveBayes nb = new NaiveBayes();
		nb.buildClassifier(train);

		RandomForest rf = new RandomForest();
		rf.buildClassifier(train);

		Evaluation eval = new Evaluation(train);
		Random rand = new Random(1);
		int folds = 10;


		eval.evaluateModel(rf, test);
		System.out.println("Correct % = " + eval.pctCorrect());
		System.out.println("Incorrect % = " + eval.pctIncorrect());
		System.out.println("AUC = " + eval.areaUnderROC(1));
		System.out.println("kappa = " + eval.kappa());
		System.out.println("MAE = " + eval.meanAbsoluteError());
		System.out.println("RMSE = " + eval.rootMeanSquaredError());
		System.out.println("RAE = " + eval.relativeAbsoluteError());
		System.out.println("RRSE = " + eval.rootRelativeSquaredError());
		System.out.println("Precision = " + eval.precision(1));
		System.out.println("Recall = " + eval.recall(1));
		System.out.println("fMeasure = " + eval.fMeasure(1));
		System.out.println("Error Rate = " + eval.errorRate());

		eval.crossValidateModel(rf, test, folds, rand);
		System.out.println(eval.toSummaryString("Evaluation results:\n", false));
		System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
		
		weka.core.SerializationHelper.write(ModelPath, rf);
	}

}