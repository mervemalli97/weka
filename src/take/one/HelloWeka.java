package take.one;
import java.io.File;

import weka.classifiers.Classifier;
import weka.classifiers.Evaluation;
import weka.classifiers.bayes.NaiveBayes;
import weka.classifiers.meta.AdaBoostM1;
import weka.classifiers.meta.Bagging;
import weka.classifiers.meta.Stacking;
import weka.classifiers.meta.Vote;
import weka.classifiers.trees.J48;
import weka.classifiers.trees.RandomForest;
import weka.classifiers.trees.RandomTree;
import weka.core.Instances;
import weka.core.converters.ConverterUtils.DataSource;

public class HelloWeka {
	public static void main(String[] args) {
		DataSource source, tsource;
		try {
			source = new DataSource("data/iris.arff");
			tsource = new DataSource("data/test.arff");

			Instances dataset = source.getDataSet();
			Instances datatest = tsource.getDataSet();
			
			if (dataset.classIndex() == -1) {
				dataset.setClassIndex(dataset.numAttributes() - 1);
			}
			if (datatest.classIndex() == -1) {
				datatest.setClassIndex(datatest.numAttributes() - 1);
			}
			
			
			AdaBoostM1 adaboost = new AdaBoostM1();
			adaboost.setClassifier(new NaiveBayes());
			adaboost.setNumIterations(20);
			adaboost.buildClassifier(dataset);
			
			Evaluation eval1 = new Evaluation(dataset);
			eval1.evaluateModel(adaboost, datatest);
			System.out.println(eval1.toSummaryString("Results for Adaboost : ", false));
			
			Bagging bag = new Bagging();
			bag.setClassifier(new RandomTree());
			bag.setNumIterations(20);
			bag.buildClassifier(dataset);


			Evaluation eval2 = new Evaluation(dataset);
			eval2.evaluateModel(bag, datatest);
			System.out.println(eval2.toSummaryString("Results for Bagging : ", false));
			
			Stacking stack = new Stacking();
			stack.setMetaClassifier(new J48());
			Classifier[] classifiers = {
					new J48(),
					new RandomForest(),
					new NaiveBayes()
			};
			stack.setClassifiers(classifiers);
			stack.buildClassifier(dataset);

			Evaluation eval3 = new Evaluation(dataset);			
			eval3.evaluateModel(stack, datatest);
			System.out.println(eval3.toSummaryString("Results for Stack : ", false));
			
			Vote voter = new Vote();
			voter.setClassifiers(classifiers);
			voter.buildClassifier(dataset);

			Evaluation eval4 = new Evaluation(dataset);
			eval4.evaluateModel(voter, datatest);
			System.out.println(eval4.toSummaryString("Results for Voter : ", false));
			
			System.out.println(adaboost.getCapabilities().toString());
			System.out.println(bag.getCapabilities().toString());
			System.out.println(stack.getCapabilities().toString());
			System.out.println(voter.getCapabilities().toString());
			
			
			
		} catch (Exception e) {
			// TODO Auto-generated catch block
			e.printStackTrace();
		}
	}
}
