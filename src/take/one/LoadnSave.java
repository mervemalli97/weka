package take.one;

import weka.core.Instances;
import weka.core.converters.ArffSaver;
import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.Remove;

import java.io.File;

import weka.core.converters.ConverterUtils.DataSource;
import weka.filters.unsupervised.attribute.Remove;
public class LoadnSave{
	public static void main(String args[]) throws Exception{
		DataSource source = new DataSource("data/iris.arff");
		Instances dataset = source.getDataSet();
		
		System.out.println(dataset.toSummaryString());
		
		String[] opts = new String[]{ "-R", "1"};
		// this takes two arguments, -R (range) is the option and 1 is the value to process
		Remove remove = new Remove(); // new instance of filter
		remove.setOptions(opts);
		remove.setInputFormat(dataset);
		Instances newData = Filter.useFilter(dataset, remove);

		ArffSaver saver = new ArffSaver();
		saver.setInstances(newData);
		saver.setFile(new File("data/new.arff"));
		saver.writeBatch();
	}
}