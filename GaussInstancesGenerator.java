import java.io.FileWriter;
import java.io.IOException;


public class GaussInstancesGenerator {

	final static String filename = "T:\\path\\to\\the\\Gauss\\file.arff";
	
	public static void main(String[] args) throws IOException {
		
		class Task {
			
			class Point {
				
				public double x;
				public double y;
				
				public Point (double XC, double YC, double sigma) {
					this.x = sigma * Math.sqrt((-2) * Math.log10(Math.random())) * Math.sin(2 * Math.PI * Math.random()) + XC;
					this.y = sigma * Math.sqrt((-2) * Math.log10(Math.random())) * Math.sin(2 * Math.PI * Math.random()) + YC;
				}
				
				
				@Override
				public String toString() {
					return this.x + "," + this.y;
				}
				
			}
			
			protected double XC = 0.0;
			protected double YC = 0.0;
			protected double sigma = 0.0;
			protected int numPoints = 0;
			
			protected Point[] points = null;
			
			public Task(double XC, double YC, double sigma, int numPoints) {
				this.XC = XC;
				this.YC = YC;
				this.sigma = sigma;
				this.numPoints = numPoints;
				this.points = new Point[numPoints];
				this.calculate();
			}
			
			
			public void calculate() {
				for (int i = 0; i < this.numPoints; i++) {
					this.points[i] = new Point(this.XC, this.YC, this.sigma);
				}
			}
			
			
			@Override
			public String toString() {
				String text = "";
				for (int i = 0; i < this.numPoints; i++) {
					text = text + this.points[i].toString() + "\n";
				}
				return text;
			}
			
		}
		
		System.out.print("Please wait....");
		StringBuffer text = new StringBuffer();
		text.append("@relation Gauss\n\n");
		text.append("@attribute x numeric\n@attribute y numeric\n\n");
		text.append("@data\n");
		text.append(new Task(0, 0, 2.5, 100));
		text.append(new Task(0, 8, 2.5, 100));
		text.append(new Task(8, 0, 2.5, 100));
		text.append(new Task(8, 8, 2.5, 100));
		FileWriter fileWriter = new FileWriter(filename);
		fileWriter.write(text.toString());
		fileWriter.close();
		System.out.println("Done!");
	}
	
}
