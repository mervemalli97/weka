package take.one;

import weka.core.Instances;
import weka.core.converters.ArffSaver;

import java.io.File;

import weka.core.converters.ConverterUtils.DataSource;
public class LoadnSave{
	public static void main(String args[]) throws Exception{
		DataSource source = new DataSource("data/iris.arff");
		Instances dataset = source.getDataSet();
		
		System.out.println(dataset.toSummaryString());
		
		ArffSaver saver = new ArffSaver();
		saver.setInstances(dataset);
		saver.setFile(new File("data/new.arff"));
		saver.writeBatch();
	}
}