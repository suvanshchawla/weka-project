package weka.api;

import weka.core.Instances;

import weka.attributeSelection.CfsSubsetEval;
import weka.attributeSelection.GreedyStepwise;
import weka.filters.supervised.attribute.AttributeSelection;

import weka.filters.Filter;
import java.util.Random;
import weka.core.converters.ConverterUtils.DataSource;
import weka.classifiers.Evaluation;
import weka.filters.unsupervised.instance.RemovePercentage;

import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.functions.SMO;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.J48;


public class modelEval_folds {
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
		RemovePercentage filter = new RemovePercentage();



		filter.setPercentage(70); // Removing 70% of the data
		filter.setInputFormat(newData); // prepare the filter for the data format
		filter.setInvertSelection(false); // do not invert the selection

		// apply filter for test data here
		Instances test = Filter.useFilter(newData, filter);

		// prepare and apply filter for training data here
		filter.setInputFormat(newData); // prepare the filter for the data format
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
		
		Evaluation eval_model = new Evaluation(train);
		
		eval_model.evaluateModel(rf, test);
		System.out.println("Correct % = " + eval_model.pctCorrect());
		System.out.println("Incorrect % = " + eval_model.pctIncorrect());
		System.out.println("AUC = " + eval_model.areaUnderROC(1));
		System.out.println("kappa = " + eval_model.kappa());
		System.out.println("MAE = " + eval_model.meanAbsoluteError());
		System.out.println("RMSE = " + eval_model.rootMeanSquaredError());
		System.out.println("RAE = " + eval_model.relativeAbsoluteError());
		System.out.println("RRSE = " + eval_model.rootRelativeSquaredError());
		System.out.println("Precision = " + eval_model.precision(1));
		System.out.println("Recall = " + eval_model.recall(1));
		System.out.println("fMeasure = " + eval_model.fMeasure(1));
		System.out.println("Error Rate = " + eval_model.errorRate());

		
		int seed = 1;
		int folds = 10;
		
		// randomize data
		Random rand = new Random(seed);
		//create random dataset
		Instances randData = new Instances(train);
		randData.randomize(rand);
		
		//stratify	    
		if (randData.classAttribute().isNominal())
			randData.stratify(folds);
		
		// perform cross-validation	    	    
		for (int n = 0; n < folds; n++) {
			Evaluation eval = new Evaluation(randData);
			//get the folds	      
			Instances train_set = randData.trainCV(folds, n);
			Instances test_set = randData.testCV(folds, n);	      
			// build and evaluate classifier	     
			rf.buildClassifier(train_set);
			eval.evaluateModel(nb, test_set);

			// output evaluation
			System.out.println("\n \n \n");
//			*********Multiple Metrics can be shown, they just need to be uncommented***********
			System.out.println(eval.toMatrixString("=== Confusion matrix for fold " + (n+1) + "/" + folds + " ===\n"));
//			System.out.println("Correct % = "+eval.pctCorrect());
//			System.out.println("Incorrect % = "+eval.pctIncorrect());
//			System.out.println("AUC = "+eval.areaUnderROC(1));
//			System.out.println("kappa = "+eval.kappa());
//			System.out.println("MAE = "+eval.meanAbsoluteError());
//			System.out.println("RMSE = "+eval.rootMeanSquaredError());
//			System.out.println("RAE = "+eval.relativeAbsoluteError());
//			System.out.println("RRSE = "+eval.rootRelativeSquaredError());
//			System.out.println("Precision = "+eval.precision(1));
//			System.out.println("Recall = "+eval.recall(1));
			System.out.println("fMeasure = "+eval.fMeasure(1));
//			System.out.println("Error Rate = "+eval.errorRate());
			//the confusion matrix
			//System.out.println(eval.toMatrixString("=== Overall Confusion Matrix ===\n"));
		}
		//Code for exporting the Classifier as a .model file
//		weka.core.SerializationHelper.write(ModelPath, rf);
	}

}
