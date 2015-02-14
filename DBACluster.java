package weka.clusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Random;
import java.util.Vector;

import weka.clusterers.RandomizableClusterer;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.core.Utils;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * <!-- globalinfo-start -->
 * Clusterer using virtual ants to cluster {@linkplain Instances} data. The ants walk
 * directly on the {@linkplain Instance} objects of Instances without using a
 * grid. For more information about ant clustering see:
 * <br/><br/>
 * Lumer&#47;Faieta 1994:
 * 	Lumer, Erik D.; Faieta, Baldo:
 * 	Diversity and Adaptation in Populations of Clustering Ants.
 * 	In: Cliff, David; Husbands, Phil; Meyer, Jean-Arcady; Wilson, Stewart W. (Eds.):
 * 	From Animals to Animats 3 - Proceedings of the Third International Conference on Simulation of Adaptive Behavior.
 *  Pages 501-508.
 * 	Complex adaptive systems.
 * 	MIT Press, Cambridge (Massachusetts), 1994.
 * <br/><br/>
 * and for more information about the background:
 * <br/><br/>
 * Deneubourg et al. 1991:
 *  Deneubourg, Jean Louis; Goss, Simon; Franks, Nigel R.; Sendova-Franks, Ana B.; Detrain, Claire; Chr&#233;tien, Ludovic:
 *  The Dynamics of Collective Sorting - Robot-Like Ants and Ant-Like Robots.
 *  In: Meyer, Jean-Arcady; Wilson, Stewart W. (Eds.):
 *  From Animals to Animats - Proceedings of the First International Conference on Simulation of Adaptive Behavior.
 *  Pages 356â€“365.
 *  MIT Press, Cambridge (Massachusetts), 1991.
 * <p/>
 * <!-- globalinfo-end -->
 * 
 * <!-- technical-bibtex-start -->
 * BibTeX:
 * <pre>
 * &#64;inbook{Lumer&#47;Faieta1994,
 *    author = {Erik D. Lumer and Baldo Faieta},
 *    title = {Diversity and Adaptation in Populations of Clustering Ants},
 *    editor = {David Cliff, Phil Husbands, Jean-Arcady Meyer and Stewart W. Wilson},
 *    booktitle = {From Animals to Animats 3 - Proceedings of the Third International Conference on Simulation of Adaptive Behavior},
 *    series = {Complex adaptive systems},
 *    publisher = {MIT Press, Cambridge (Massachusetts)},
 *    pages = {501-508},
 *    year = {1994}
 * }
 * </pre>
 * <p/>
 * <!-- technical-bibtex-end -->
 * 
 * <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -a &lt;num&gt;
 *  Alpha, coefficient for cluster similarity.</pre>
 * 
 * <pre> -s &lt;num&gt;
 *  Size of the neighborhood, respectively the ants view range. Relative to the
 *  longest distance, but values greater than 1 are also possible for extremely
 *  large neighborhoods, e.g. when running a clustering task for very sparse
 *  instances. All instances within the range are regarded as neighbors of the
 *  instance. The greater the value is, the bigger is the neighborhood.</pre>
 * 
 * <pre> -rt &lt;num&gt;
 *  Tolerated raise of the local similarity during the clustering process. If the
 *  difference between the local similarity values of two instances is not
 *  greater than this value, the difference is not regarded as significant and
 *  the two instances can belong to the same cluster.</pre>
 * 
 * <pre> -nt &lt;num&gt;
 *  Threshold for marking instances as noise. Instances with a local similarity
 *  value equal or below this threshold value should be marked as noise. Set to
 *  -1 to not seek for noise instances.</pre>
 * 
 * <pre> -an &lt;num&gt;
 *  Number of ants in the ant colony. How many ants are clustering instances.</pre>
 * 
 * <pre> -ic &lt;num&gt;
 *  How many ants are called per ant cycle. Each ant that is called can work and
 *  walk.</pre>
 * 
 * <pre> -i &lt;num&gt;
 *  Maximum number of ant cycles to be executed. When all ant cycles passed, the
 *  algorithm terminates. If all ants shut down before all ant cycles passed, the
 *  algorithm can also terminate earlier.</pre>
 * 
 * <pre> -aag &lt;num&gt;
 *  When an ant observes something for this number of ant calls in series, it
 *  assumes that the observation is a global phenomenon.</pre>
 * 
 * <pre> -dist &lt;classname and options&gt;
 *  Distance function that is used for instance comparison according to the
 *  attributes of the instances.
 *  (default = weka.core.EuclideanDistance)</pre>
 * 
 * <pre> -cm &lt;num&gt;
 *  The cluster number of the result can be limited to this number of clusters.
 *  After clustering ended the clusters are joined to match this criteria. If
 *  there should not be a limitation set this value to -1.</pre>
 * 
 * <pre> -m
 *  Replace missing values.</pre>
 * 
 * <!-- options-end -->
 * 
 * @version 0.9
 * @author Christoph
 * @see RandomizableClusterer
 */
public class DBACluster extends RandomizableClusterer implements TechnicalInformationHandler {
	
	/** For serialization */
	private static final long serialVersionUID = 7917723307771847875L;
	
	/** Alpha interval lower bound. */
	static final double alphaMinValue = 0.0;
	
	/** Alpha interval upper bound. */
	static final double alphaMaxValue = 9999999.0;
	
	/** The cluster ID for not assigned clusters. */
	static final int unassignedClusterID = -1;
	
	/** Colony similarity coefficient alpha. The larger, the more similar the colonies must be. */
	protected double optn_alpha = 0.37; //-a >= 0 , 0.37
	
	/** Size of the neighborhood. All instances within the range are regarded as neighbors of an instance. */
	protected double optn_s = 0.25; //-s >= 0 , 0.125 , 0.25
	
	/** A raise of the local similarity value between two instances is not regarded as significant when not greater than this value. */
	protected double optn_foiRaiseTolerance = 0.012; //-rt >= 0 , 0.0125 , 0.12
	
	/** When the local similarity of an instance is below this value it is regarded as noise and not clustered. Set to -1 to not seek for noise instances. */
	protected double optn_foiNoiseThreshold = -1.0; //-nt >= 0 || -1
	
	/** Number of ants. */
	protected int optn_antsNum = 10; //-an > 0, 10
	
	/** How many ants are called per ant cycle. */
	protected int optn_antsCallPerAntCycle = 10000; //-ic > 0, 10000
	
	/** How many ant cycles should be executed maximum. */
	protected int optn_antsMaxAntCycles = 50; //-i >= 0, 50
	
	/** Ants assume that their observation is a global phenomenon when they saw it for this number of calls in series. */
	protected int optn_antsAssumeGlobalAfterNumCalls = 1000; //-aag > 0
	
	/** This is a modifier to handle the distance calculation of difficult datasets in a different way. The distance is divided by the maximum distance in a neighborhood when set. */
	protected boolean optn_distanceFunctionAlign = false; //-d
	
	/** The distance function used for determining the distance between instances. */
	protected DistanceFunction optn_distanceFunction = new EuclideanDistance(); //-dist
	
	/** The maximum number of clusters in the result. Set to -1 to not limit the maximum number of clusters. */
	protected int optn_maxClusterNum = -1; //-cm > 0 || -1
	
	/** How many neighbor Instances are evaluated in the neighborhood maximum. It can speed up the algorithm a bit. Set to -1 to always evaluate the whole neighborhood. */
	protected int optn_maxNumNeighborEvaluation = 37; //-nne //Not yet implemented. A way to make the algorithm faster.
	
	/** Replace missing values globally? */
	protected boolean optn_replaceMissing = true; //-m
	
	/** Instances data to be clustered. */
	protected Instances data = null; //It must not be altered after the instances were read! Classes refer to it, it is like a global variable/knowledge, and must be initialized first!
	
	/** Ant colony of ants. */
	protected AntHill antHill = null;
	
	/** Assignments of the most recently clustered gridInstances to clusters. */
	protected int[] out_clusterAssignments = null;
	
	/** How many ant cycles were executed. */
	protected int out_antCycles = -1;
	
	/** Were as many ant cycles executed as allowed or less? */
	protected boolean out_atAntCycleExecutionLimit = false;
	
	/** Size of the largest neighborhood during last execution. */
	protected int out_maxNeighborhoodSize = -1;
	
	/** Debug: Amount of foi values below or equal optn_foiNoiseThreshold in last clustering execution. */
//	protected int out_foiCountBelowOrEqualNoiseThreshold = -1;
	
	/** Debug: Minimum foi value found above optn_foiNoiseThreshold in last execution. */
//	protected double out_minimumFoiAboveNoiseThreshold = -1;
	
	/** Debug: Average foi value found in last execution. */
//	protected double out_averageFoi = -1;
	
	/** Debug: Maximum foi value found in last execution. */
//	protected double out_maximumFoi = -1;
	
	/** How many instances remained unclustered. */
	protected int out_numUnclustered = -1;
	
	/** Replace missing values filter. */
	protected ReplaceMissingValues replaceMissingValuesFilter;
	
	/** Random number generator. */
	protected Random rand = null; //It is required globally, and can not be instantiated every time it is used, because the generator starts with the same seed then again -> always same numbers are generated.
	
	/** The default constructor. */
	public DBACluster() {
		super();
		this.rand = new Random(getSeed());
	}
	
	
	/**
	 * Returns a description of this clusterer.
	 * 
	 * @return a brief text describing this clusterer.
	 */
	public String globalInfo() {
		return "Clusterer using artificial ants (agents) to cluster instances. "
				+ "The ants walk in the instance attribute space and group instances that indicate their affiliation by their"
				+ "local similarity values into clusters. Instances are assigned to other instances that have a higher similarity"
				+ "value, so if the instance to be clustered would be moved it would be moved the shortest distance when moved towards"
				+ "an instance with higher similarity value.";
	}
	
	
	/**
	 * Gives more information about the technical article of this class.
	 * 
	 * @return A TechnicalInformation object, with references to
	 *         the article used for building this algorithm.
	 * @see TechnicalInformationHandler
	 */
	@Override
	public TechnicalInformation getTechnicalInformation() {
		TechnicalInformation ti;
		ti = new TechnicalInformation(Type.ARTICLE);
		ti.setValue(Field.AUTHOR, "Lumer, Erik D.; Faieta, Baldo");
		ti.setValue(Field.TITLE, "Diversity and Adaptation in Populations of Clustering Ants");
		ti.setValue(Field.EDITOR, "Cliff, David; Husbands, Phil; Meyer, Jean-Arcady; Wilson, Stewart W.");
		ti.setValue(Field.BOOKTITLE, "From Animals to Animats 3 - Proceedings of the Third International Conference on Simulation of Adaptive Behavior");
		ti.setValue(Field.SERIES, "Complex adaptive systems");
		ti.setValue(Field.PUBLISHER, "MIT Press");
		ti.setValue(Field.YEAR, "1994");
		ti.setValue(Field.PAGES, "499-508");
		ti.setValue(Field.ISBN13, "9780262531221");
		return ti;
	}
	
	
	/**
	 * Returns default capabilities of the clusterer.
	 * 
	 * @return the capabilities of this clusterer
	 */
	@Override
	public Capabilities getCapabilities() {
		Capabilities result = super.getCapabilities();
		result.disableAll();
		result.enable(Capability.NO_CLASS); //This is a clusterer, so no class required.
		result.enable(Capability.NOMINAL_ATTRIBUTES); //Attribute capabilities..
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		return result;
	}
	
	
	/**
	 * Tip text provider for the alpha value setting.
	 * 
	 * @return Text that briefly describes the functionality of the alpha value
	 *         setting. 
	 */
	public String alphaTipText() {
		return "Colony similarity coefficient Alpha for adjusting the similarity of clusters.";
	}
	
	
	/**
	 * Sets the alpha value.
	 * 
	 * @param value the alpha value to be used
	 * @throws IllegalArgumentException if {@code value} is below or equal
	 *         {@value #alphaMinValue} or above {@value #alphaMaxValue}.
	 */
	public void setAlpha(double value) throws IllegalArgumentException {
		if (value <= alphaMinValue || value > alphaMaxValue) { //Must not be null, because otherwise division with 0 can occur later in calculating foi.
			if (m_Debug) {
				throw new IllegalArgumentException("The colony similarity coefficient must be in the interval between " + alphaMinValue + " and " + alphaMaxValue + ".");
			}
			else {
				value = alphaMaxValue;
			}
		}
		optn_alpha = value;
	}
	
	
	/**
	 * Tells the currently set alpha value.
	 * 
	 * @return the current alpha value.
	 */
	public double getAlpha() {
		return this.optn_alpha;
	}
	
	
	/**
	 * Tip text provider for the neighborhood size setting.
	 * 
	 * @return Text that briefly describes the functionality of the
	 *         pick up threshold.
	 */
	public String neighborhoodSizeTipText() {
		return "Size of the neighborhood around an instance or ant relative to the maximum distance in the dataset (when using normalized distance).";
	}
	
	
	/**
	 * Sets the neighborhood size. How far ants can see when they are at a
	 * instance.
	 * 
	 * @param value size of the neighborhood in percent. Usually 1.0 is the size
	 *        of the largest neighborhood, but in sparse data sets oversized
	 *        neighborhoods also can be helpful.
	 * @throws IllegalArgumentException if in debug mode and {@code value} < 0.
	 */
	public void setNeighborhoodSize(double value) throws IllegalArgumentException {
		if (value < 0) {
			if (m_Debug) {
				throw new IllegalArgumentException("The neighborhood size must be at least 0.");
			}
			else {
				value = 0.0;
			}
		}
		this.optn_s = value;
	}
	
	/**
	 * Tells the size of the neighborhood respectively the ants view range.
	 * 
	 * @return the size of the neighborhood.
	 */
	public double getNeighborhoodSize() {
		return this.optn_s;
	}
	
	
	/**
	 * Tip text provider for the local similarity raise tolerance setting.
	 * 
	 * @return Text that briefly describes the local similarity raise tolerance
	 *         setting.
	 */
	public String localSimilarityRaiseToleranceTipText() {
		return "Tolerated raise of the local similarity between two instances.";
	}
	
	
	/**
	 * Sets the local similarity raise tolerance.
	 * 
	 * @param value new local similarity raise tolerance threshold.
	 * @throws IllegalArgumentException if in debug mode and when {@code value} < 0.
	 */
	public void setLocalSimilarityRaiseTolerance(double value) throws IllegalArgumentException {
		if (value < 0) {
			if (m_Debug) {
				throw new IllegalArgumentException("The tolerance can only be 0 or greater.");
			}
			else {
				value = 0.0;
			}
		}
		this.optn_foiRaiseTolerance = value;
	}
	
	
	/**
	 * Tells the local similarity raise tolerance.
	 * 
	 * @return the currently set local similarity raise tolerance threshold.
	 */
	public double getLocalSimilarityRaiseTolerance() {
		return this.optn_foiRaiseTolerance;
	}
	
	
	/**
	 * Tip text provider for the local similarity noise threshold setting.
	 * 
	 * @return Text that briefly describes the local similarity noise threshold
	 *         setting.
	 */
	public String localSimilarityNoiseThresholdTipText() {
		return "Instances with a local similarity value below this threshold are regarded as noise or set to -1 no not seek for noise instances.";
	}
	
	
	/**
	 * Sets the local similarity noise threshold.
	 * 
	 * @param value all instances with a local similarity below this value are
	 *              regarded as noise and not clustered. Set the value -1 to not
	 *              search for noise instances.
	 * @throws IllegalArgumentException if in debug mode and when {@code value} is < 0
	 *         (except -1).
	 */
	public void setLocalSimilarityNoiseThreshold(double value) throws IllegalArgumentException {
		if (value < 0 && value != -1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The noise threshold can only be 0 or greater. Set to -1 to not mark noise instaces.");
			}
			else {
				value = 0.0;
			}
		}
		this.optn_foiNoiseThreshold = value;
	}
	
	
	/**
	 * Tells the local similarity noise threshold.
	 * 
	 * @return threshold for finding noise instances.
	 */
	public double getLocalSimilarityNoiseThreshold() {
		return this.optn_foiNoiseThreshold;
	}
	
	
	/**
	 * Tip text provider for the number of ants setting.
	 * 
	 * @return Text that briefly describes the number of ants setting.
	 */
	public String antsNumTipText() {
		return "Number of ants available.";
	}
	
	
	/**
	 * Sets the number of available ants.
	 * 
	 * @param value how many ants should work on the clustering task.
	 * @throws IllegalArgumentException if {@code value} is smaller than 1 and in
	 *         debug mode.
	 */
	public void setAntsNum(int value) {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The number of ants must be a positive integer value. Also at least one ant must be available, otherwise nothing can be clustered.");
			}
			else {
				value = 1;
			}
		}
		this.optn_antsNum = value;
	}
	
	
	/**
	 * Tells how many ants must work on the clustering task.
	 * 
	 * @return ant count.
	 */
	public int getAntsNum() {
		return this.optn_antsNum;
	}
	
	
	/**
	 * Tip text provider for the ant calls per ant cycle setting.
	 * 
	 * @return Text that briefly describes the ant calls per ant cycle setting.
	 */
	public String antsCallPerAntCycleTipText() {
		return "How many ants are called per ant cycle.";
	}
	
	
	/**
	 * Sets the amount of ants to be called per ant cycle.
	 * 
	 * @param value ant calls per ant cycle.
	 * @throws IllegalArgumentException if {@code value} is smaller than 1 and
	 *         debug mode is activated.
	 */
	public void setAntsCallPerAntCycle(int value) {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The number of ants to call per ant cycle must be a positive integer value. Also at least one ant must be called per ant cycle, otherwise nothing can be clustered.");
			}
			else {
				value = 1;
			}
		}
		this.optn_antsCallPerAntCycle = value;
	}
	
	
	/**
	 * Tells how many ants currently should be called in one ant cycle. 
	 * 
	 * @return ants per ant cycle call count.
	 */
	public int getAntsCallPerAntCycle() {
		return this.optn_antsCallPerAntCycle;
	}
	
	
	/**
	 * Tip text provider for the number of ant cycles to execute maximum setting.
	 * 
	 * @return Text that briefly describes the maximum ant cycles setting.
	 */
	public String antsMaxAntCyclesTipText() {
		return "Maximum number ant cycles (iterations).";
	}
	
	
	/**
	 * Set the maximum number of ant cycles to be executed.
	 * 
	 * @param value the maximum number of iterations.
	 * @throws IllegalArgumentException if maximum number of ant cycles is
	 *         smaller than 1.
	 */
	public void setAntsMaxAntCycles(int value) throws IllegalArgumentException {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("Maximum number of ant cycles must be positive integer > 1.");
			}
			else {
				value = 12;
			}
		}
		this.optn_antsMaxAntCycles = value;
	}
	
	
	/**
	 * Tells the currently set number of ant cycles to execute maximum.
	 * 
	 * @return maximum number of ant cycles to execute.
	 */
	public int getAntsMaxAntCycles() {
		return this.optn_antsMaxAntCycles;
	}
	
	
	/**
	 * Tip text provider for the assume global setting.
	 * 
	 * @return Text that briefly describes the assume global setting.
	 */
	public String antsAssumeGlobalAfterNumCallsTipText() {
		return "Ants assume that an observation is a global phenomenon when they observed it this amount of calls in series.";
	}
	
	
	/**
	 * Set after how many ant calls an ant regards a phenomenon as global when
	 * the ant observed that for this number of ant cycles in series.
	 * 
	 * @param value ant calls to mind before assuming something observed is
	 *              global.
	 * @throws IllegalArgumentException if in debug mode and {@code value} < 1.
	 */
	public void setAntsAssumeGlobalAfterNumCalls(int value) throws IllegalArgumentException {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The number of observation calls must be at least 1 or greater.");
			}
			else {
				value = 1000;
			}
		}
		this.optn_antsAssumeGlobalAfterNumCalls = value;
	}
	
	
	/**
	 * Tell after how many ant calls an ant assumes some observed phenomenon is
	 * global when the ant observed that for this number of calls.
	 * 
	 * @return ant calls to pass before something is regarded as global.
	 */
	public int getAntsAssumeGlobalAfterNumCalls() {
		return this.optn_antsAssumeGlobalAfterNumCalls;
	}
	
	
	/**
	 * Tip text provider for the distance function align setting.
	 * 
	 * @return Text that briefly describes the distance function align setting.
	 */
	public String distanceFunctionAlignTipText() {
		return "As a special modifier usable for difficult data sets to align each distance in a neighborhood to the maximum distance in this neighborhood.";
	}
	
	
	/**
	 * Set to true if the distance of a neighborhood should be aligned to the
	 * maximum distance in each neighborhood. This is usually not necessary.
	 * 
	 * @param value set to true to use extra calculation of the neighborhood
	 *        distance.
	 */
	public void setDistanceFunctionAlign(boolean value) {
		this.optn_distanceFunctionAlign = value;
	}
	
	
	/**
	 * Tell if the distance in the neighborhoods is calculated with the extra
	 * calculation.
	 * 
	 * @return true, if the extra calculation should be used.
	 */
	public boolean getDistanceFunctionAlign() {
		return this.optn_distanceFunctionAlign;
	}
	
	
	/**
	 * Tip text provider for the distance function setting.
	 * 
	 * @return Text that briefly describes the functionality of the
	 *         distance function.
	 */
	public String distanceFunctionTipText() {
		return "Distance function to use for instance comparison.";
	}
	
	
	/**
	 * Sets the distance function.
	 * 
	 * @param value the distance function to use, as a {@code DinstanceFunction} object.
	 */
	public void setDistanceFunction(DistanceFunction value) {
		this.optn_distanceFunction = value;
	}
	
	
	/**
	 * Tells the currently set distance function.
	 * 
	 * @return an instance of the current distance function. Its class is
	 *         {@code DinstanceFunction}.
	 */
	public DistanceFunction getDistanceFunction() {
		return this.optn_distanceFunction;
	}
	
	
	/**
	 * Tip text provider for the maximum cluster number setting.
	 * 
	 * @return Text that briefly describes the maximum cluster number setting.
	 */
	public String maxClusterNumTipText() {
		return "The maximum number of clusters can be limited in the result.";
	}
	
	
	/**
	 * Sets an upper bound for the cluster count in the result. Set to -1 to not
	 * postprocess the clusters.
	 * 
	 * @param value how many clusters should be in the result maximum.
	 * @throws IllegalArgumentException if in debug mode and {@code value} < 1 and not
	 *                                  -1.
	 */
	public void setMaxClusterNum(int value) throws IllegalArgumentException {
		if (value < 1 && value != -1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The number of clusters in the result must be 1 or greater. Set to -1 to not limit the number of clusters in the result.");
			}
			else {
				value = -1;
			}
		}
		this.optn_maxClusterNum = value;
	}
	
	
	/**
	 * Tells how many clusters are allowed in the result.
	 * 
	 * @return number of clusters allowed in the result, or -1 if there is no
	 *         limit.
	 */
	public int getMaxClusterNum() {
		return this.optn_maxClusterNum;
	}
	
	
	/**
	 * Tip text provider for the replace missing values setting.
	 * 
	 * @return Text that briefly describes the missing values setting.
	 */
	public String replaceMissingTipText() {
		return "Replace missing values globally.";
	}
	
	
	/**
	 * Sets the replace missing value option.
	 * 
	 * @param value true, if the clusterer should try to calculate values for
	 *        missing values in the dataset.
	 */
	public void setReplaceMissing(boolean value) {
		optn_replaceMissing = value;
	}
	
	
	/**
	 * Tells if missing values should be replaced.
	 * 
	 * @return true, if missing values should be replaced.
	 */
	public boolean getReplaceMissing() {
		return optn_replaceMissing;
	}
	
	
	/**
	 * Provides information about the available options for this clusterer.
	 * 
	 * @return An {@linkplain Enumeration} holding the descriptions of available {@linkplain Option}s.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();
		result.addElement(new Option("\tAlpha, colony similarity coefficient.\n\tParameter for cluster similarity.", "a", 1, "-a <num>")); 
		result.addElement(new Option("\tSize of the neighborhood or view range of each ant.\n\tThe radius defines which area around an ant is regarded as the ant's neighborhood. The greater the value is, the bigger is the neighborhood. Its is specified realatively to the longest distance, but values greater than 1 are also possible for extremely large neighborhoods, e.g. when running a clustering task for very sparse instances. It is recommended to keep normalization activated in the options of the distance function.", "s", 1, "-s <num>"));
		result.addElement(new Option("\tTolerated raise of the local similarity during the clustering process.\n\tIf the difference between the local similarity values of two instances is not greater than this value, the difference is not regarded as significant and the two instances can belong to the same cluster.", "rt", 1, "-rt <num>"));
		result.addElement(new Option("\tThreshold for marking instances as noise.\n\tInstances with a local similarity value equal or below this threshold value should be marked as noise. Set to -1 to not seek for noise instances.", "nt", 1, "-nt <num>"));
		result.addElement(new Option("\tNumber of ants in the ant colony.\n\tHow many ants are clustering instances.", "an", 1, "-an <num>"));
		result.addElement(new Option("\tHow many ants are called per ant cycle.\n\tEach ant that is called can work and walk.", "ic", 1, "-ic <num>"));
		result.addElement(new Option("\tMaximum number of ant cycles to be executed.\n\tEach ant cycle is composed of a fixed number of ant calls. When all ant cycles passed, the algorithm terminates. If all ants shut down before all ant cycles passed, the algorithm can also terminate earlier.", "i", 1, "-i <num>"));
		result.addElement(new Option("\tAfter which number of calls an ant assumes an observation is a global phenomenon.\n\tWhen an ant observes something for this number of ant calls in series, it assumes that the observation is a global phenomenon.", "aag", 1, "-aag <num>"));
		result.addElement(new Option("\tUse extra calculation for the distance in a neighborhood.\n\tWhen set to true, the distances in a neighborhood are divided by the maximum distance of this neighborhood, too. This is usually not necessary.", "d", 0, "-d <num>"));
		result.addElement(new Option("\tDistance function to use for instance comparison.\n\tThis distance function is used to determine the distance between two instances according to their attributes.\n\t(default = weka.core.EuclideanDistance)", "dist", 1, "-dist <classname and options>"));
		result.addElement(new Option("\tMaximum number of clusters in the result.\n\tThe cluster number of the result can be limited to this number of clusters. After clustering ended the clusters are joined to match this criteria. If there should not be a limitation set this value to -1.", "cm", 1, "-cm <num>"));
		result.addElement(new Option("\tReplace missing values.\n\tReplace missing values globally.\n\t(default = true)", "m", 0, "-m"));
		return result.elements();
	}
	
	
	/**
	 * Sets the options given as a string.
	 * 
	 * @param options a string describing the options to be set.
	 * @throws Exception if setting an option fails.
	 */
	@Override
	public void setOptions(String[] options) throws Exception {
		
		String temp;
		
		temp = Utils.getOption("a", options);
		if (temp.length() > 0) {
			this.setAlpha(Double.parseDouble(temp));
		}
		
		temp = Utils.getOption("s", options);
		if (temp.length() > 0) {
			this.setNeighborhoodSize(Double.parseDouble(temp));
		}
		
		temp = Utils.getOption("rt", options);
		if (temp.length() > 0) {
			this.setLocalSimilarityRaiseTolerance(Double.parseDouble(temp));
		}
		
		temp = Utils.getOption("nt", options);
		if (temp.length() > 0) {
			this.setLocalSimilarityNoiseThreshold(Double.parseDouble(temp));
		}
		
		temp = Utils.getOption("an", options);
		if (temp.length() > 0) {
			this.setAntsNum(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("ic", options);
		if (temp.length() > 0) {
			this.setAntsCallPerAntCycle(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("i", options);
		if (temp.length() > 0) {
			this.setAntsMaxAntCycles(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("aag", options);
		if (temp.length() > 0) {
			this.setAntsMaxAntCycles(Integer.parseInt(temp));
		}

		this.setReplaceMissing(Utils.getFlag("d", options));
		
		temp = Utils.getOption("dist", options);
		if (temp.length() > 0) {
			String[] classSpec = Utils.splitOptions(temp);
			if (classSpec.length == 0) {
				throw new Exception("Invalid DistanceFunction specification string.");
			}
			String className = classSpec[0];
			classSpec[0] = "";
			this.setDistanceFunction((DistanceFunction) Utils.forName(DistanceFunction.class, className, classSpec));
		}
		else {
			this.setDistanceFunction(new EuclideanDistance());
		}
		
		temp = Utils.getOption("cm", options);
		if (temp.length() > 0) {
			this.setMaxClusterNum(Integer.parseInt(temp));
		}
		
		this.setReplaceMissing(Utils.getFlag("m", options));
		
		super.setOptions(options);
		Utils.checkForRemainingOptions(options);
		
	}
	
	
	/**
	 * Retrieves the current set options and their values as a string.
	 * 
	 * @return options and values written in a string.
	 */
	@Override
	public String[] getOptions() {
		
		Vector<String> result = new Vector<String>();
		
		result.add("-a");
		result.add("" + this.getAlpha());
		
		result.add("-s");
		result.add("" + this.getNeighborhoodSize());
		
		result.add("-rt");
		result.add("" + this.getLocalSimilarityRaiseTolerance());
		
		result.add("-nt");
		result.add("" + this.getLocalSimilarityNoiseThreshold());
		
		result.add("-an");
		result.add("" + this.getAntsNum());
		
		result.add("-ic");
		result.add("" + this.getAntsCallPerAntCycle());
		
		result.add("-i");
		result.add("" + this.getAntsMaxAntCycles());
		
		result.add("-aag");
		result.add("" + this.getAntsAssumeGlobalAfterNumCalls());
		
		if (optn_replaceMissing) {
			result.add("-d");
		}
		
		result.add("-dist");
		result.add((this.optn_distanceFunction.getClass().getName()
				+ " " + Utils.joinOptions(this.optn_distanceFunction.getOptions())).trim());
		
		result.add("-cm");
		result.add("" + this.getMaxClusterNum());
		
		if (optn_replaceMissing) {
			result.add("-m");
		}
		
		Collections.addAll(result, super.getOptions());
		return result.toArray(new String[result.size()]);
		
	}
	
	
	/**
	 * The AntHill contains all the ants. The clustering process is executed
	 * here.
	 * <p>
	 * The AntHill also can perform simple management tasks on the ants, like 
	 * addressing and calling them, but of course it must not manage the ants in
	 * a way that this class becomes a global control unit for the ants, e.g. it
	 * must not tell the ants where to go to.
	 */
	protected class AntHill {
		
		/**
		 * Management class for {@linkplain Cluster}s. It provides a more
		 * comfortable way to address and manage Cluster objects.
		 */
		protected class Clusters implements Iterable<Cluster> {
			
			/**
			 * The task of the ClusterNumGenerator is to provide ID numbers for
			 * new clusters, so that each cluster has an unique ID. Therefore it
			 * should be involved when constructing a new Cluster.
			 */
			protected class ClusterNumGenerator {
				
				/** ID for the next cluster. */
				private int counter = -1;
				
				/** Default constructor. */
				public ClusterNumGenerator() {
					this.counter = -1;
				}
				
				/**
				 * Returns a yet unused ID that can be assigned to a new
				 * cluster.
				 * 
				 * @return an ID for a new cluster.
				 */
				public synchronized int next() {
					do {
						this.counter++;
					}
					while (this.counter == unassignedClusterID);
					return this.counter;
				}
				
				
				/**
				 * Resets this cluster number generator and start over.
				 * <p>
				 * Warning! Once a ClusterNumGenerator is reset it may return
				 * cluster IDs that are already in use! So reset a
				 * ClusterNumGenerator before or after a clustering task only!
				 */
				public void reset() {
					this.counter = -1;
				}
				
			}
			
			/** List of available {@linkplain Cluster} objects. */
			protected ArrayList<Cluster> clusters;
			
			/** A link to the noise cluster, that is a special cluster holding the instances regarded as noise. */
			protected Cluster noiseCluster;
			
			/** The ClusterNumGenerator object of this Clusters instance. */
			protected ClusterNumGenerator clusterNumGenerator;
			
			/** The default constructor. */
			public Clusters() {
				this.clusters = new ArrayList<Cluster>();
				this.noiseCluster = new Cluster(unassignedClusterID);
				this.clusterNumGenerator = new ClusterNumGenerator();
				this.clusterNumGenerator.reset();
			}
			
			
			/**
			 * Tells the Clusters object to make a new {@linkplain Cluster}.
			 * 
			 * @return the new Cluster object.
			 */
			public Cluster makeNewCluster() {
				Cluster cluster = new Cluster(this.clusterNumGenerator.next());
				this.clusters.add(cluster);
				return cluster;
			}
			
			
			/**
			 * Tell how many clusters are currently managed by this Clusters
			 * object.
			 * 
			 * @return number of clusters currently available.
			 */
			public int size() {
				return this.clusters.size();
			}
			
			
			/**
			 * Retrieves the {@linkplain Cluster} with the given {@code ID}.
			 * 
			 * @param num index of the cluster to get.
			 * @return Cluster with the given index or null if nothing found.
			 */
			public Cluster get(int num) {
				return this.clusters.get(num);
			}
			
			
			/**
			 * Returns the special noise {@linkplain Cluster}.
			 * 
			 * @return the noise cluster.
			 * @see #noiseCluster
			 */
			public Cluster getNoiseCluster() {
				return this.noiseCluster;
			}
			
			
			/**
			 * Iterates over all {@linkplain Cluster} objects managed by this object.
			 * 
			 * @return an {@linkplain Iterator} over Cluster objects.
			 */
			@Override
			public Iterator<Cluster> iterator() {
				return new Iterator<Cluster>() {
					private int position = 0;
					@Override
					public boolean hasNext() {
						return position < clusters.size();
					}
					@Override
					public Cluster next() {
						Cluster cluster = clusters.get(position);
						position++;
						return cluster;
					}
				};
			}
			
		}
		
		
		/**
		 * This is the cluster that can be assigned to the instances in the
		 * dataset.
		 * <p>
		 * It must not perform actions by itself, but is only used by other
		 * classes.
		 */
		protected class Cluster implements Iterable<InstancePlaceholder> {
			
			/** ID of this Cluster, unique among all Cluster objects. */
			protected int ID = -1;
			
			/**
			 * {@linkplain InstancePlaceholder} this cluster started with. It is usually the
			 * most different InstancePlaceholer to InstancePlaceholder objects
			 * of other clusters.
			 */
			protected InstancePlaceholder start;
			
			/** All {@linkplain InstancePlaceholder} members of this cluster. */
			protected ArrayList<InstancePlaceholder> members;
			
			/**
			 * The default constructor.
			 *  
			 * @param ID ID of the new cluster.
			 */
			public Cluster(int ID) {
				this.ID = ID;
				this.members = new ArrayList<InstancePlaceholder>();
			}
			
			
			/**
			 * Changes the ID of this cluster to the given {@code ID}.
			 * 
			 * @param ID new ID of this cluster.
			 */
			public void setID(int ID) {
				this.ID = ID;
			}
			
			
			/**
			 * Tells the ID of this cluster.
			 * 
			 * @return ID of this cluster.
			 */
			public int getID() {
				return this.ID;
			}
			
			
			/**
			 * Tells if this cluster is the noise cluster, therefore containing
			 * all {@linkplain InstancePlaceholder} objects regarded as noise.
			 * 
			 * @return true, if this cluster is the noise cluster, false
			 *         otherwise.
			 */
			public boolean isNoiseCluster() {
				return this.ID == unassignedClusterID;
			}
			
			
			/**
			 * Sets another starting point {@linkplain InstancePlaceholder} for this cluster.
			 * 
			 * @param start starting point to set for this cluster.
			 */
			public void setStart(InstancePlaceholder start) {
				this.start = start;
			}
			
			
			/**
			 * Tells the {@linkplain InstancePlaceholder} this cluster started with.
			 * 
			 * @return starting point of this cluster.
			 */
			public InstancePlaceholder getStart() {
				if (m_Debug && !(this.start instanceof InstancePlaceholder)) {
					throw new RuntimeException("Start of cluster is missing or not instance of InstancePlaceholder. Every cluster must have a start instance placeholder.");
				}
				return start;
			}
			
			
			/**
			 * Tells the size of this cluster, namely how many
			 * {@linkplain InstancePlaceholder} objects this cluster contains.
			 * 
			 * @return member count of this cluster.
			 */
			public int size() {
				return this.members.size();
			}
			
			
			/**
			 * Adds a new {@linkplain InstancePlaceholder} to this cluster.
			 * 
			 * @param mem the InstancePlaceholder to add.
			 */
			public void add(InstancePlaceholder mem) {
				if (!(this.start instanceof InstancePlaceholder)) {
					this.start = mem;
				}
				this.members.add(mem);
			}
			
			
			/**
			 * Removes a member from this cluster.
			 * 
			 * @param mem the {@linkplain InstancePlaceholder} to remove from this cluster.
			 */
			public void remove(InstancePlaceholder mem) {
				this.members.remove(mem);
			}
			
			
			/**
			 * Tells if this cluster contains a specific {@linkplain InstancePlaceholder}.
			 * 
			 * @param mem InstancePlaceholder to request membership information
			 *            for.
			 * @return true if this cluster contains {@code mem}, false otherwise.
			 */
			public boolean contains(InstancePlaceholder mem) {
				return this.members.contains(mem);
			}
			
			
			/**
			 * Tells if this object equals the given object, by customized criteria.
			 * 
			 * @param obj the object to compare with.
			 * @return true, if this object equals the compared {@code obj} object.
			 * @see java.lang.Object#equals
			 */
			@Override
			public boolean equals(Object obj) {
				if (!(obj instanceof Cluster)) {
					return false;
				}
				return this.ID == ((Cluster) obj).ID;
			}
			
			
			/**
			 * Returns a string representation of this object.
			 * 
			 * @return a string representing this object.
			 * @see java.lang.Object#toString()
			 */
			@Override
			public String toString() {
				return "[c:" + this.ID + "," + this.members.size() + "]";
			}
			
			
			/**
			 * Iterates over all {@linkplain InstancePlaceholder} objects managed by this object.
			 * 
			 * @return an {@linkplain Iterator} over InstancePlaceholder objects.
			 */
			@Override
			public Iterator<InstancePlaceholder> iterator() {
				return new Iterator<InstancePlaceholder>() {
					private int position = 0;
					@SuppressWarnings("unchecked")
					private ArrayList<InstancePlaceholder> list = (ArrayList<InstancePlaceholder>) members.clone();
					@Override
					public boolean hasNext() {
						return position < list.size();
					}
					@Override
					public InstancePlaceholder next() {
						InstancePlaceholder member = list.get(position);
						position++;
						return member;
					}
				};
			}
			
		}
		
		
		/**
		 * Management class for {@linkplain InstancePlaceholder} objects.
		 * <p>
		 * It provides a more comfortable way to address and manage
		 * InstancePlaceholder objects. It also does the setup for the
		 * {@link weka.core.Instance} objects to InstancePlaceholder association.
		 */
		protected class InstancePlaceholders implements Iterable<InstancePlaceholder> { //Similar to GridInstance/s.
			
			/** Holds a {@link weka.core.Instances} object, containing the original {@link weka.core.Instance} objects. */
			protected Instances data;
			
			/** The {@linkplain InstancePlaceholder} objects managed by this class. */
			protected InstancePlaceholder[] instancePlaceholders;
			
			/** All {@linkplain InstancePlaceholder} objects, that do not know their neighbors yet, are stored here again. */
			protected ArrayList<InstancePlaceholder> neighborsUnexploredList;
			
			/**
			 * The default constructor of this class.
			 * 
			 * @param data the {@link weka.core.Instances} this object should rely on.
			 */
			public InstancePlaceholders(Instances data) {
				int size = data.size();
				this.data = data;
				this.instancePlaceholders = new InstancePlaceholder[size];
				this.neighborsUnexploredList = new ArrayList<InstancePlaceholder>();
				for (int i = 0; i < size; i++) {
					InstancePlaceholder ip = new InstancePlaceholder(i, this.data.get(i), this);
					this.instancePlaceholders[i] = ip;
					this.neighborsUnexploredList.add(ip);
				}
				this.neighborsUnexploredList.trimToSize();
			}
			
			
			/**
			 * Gets the {@linkplain InstancePlaceholder} with the given ID.
			 * 
			 * @param ID ID of the InstancePlaceholder to retrieve.
			 * @return InstancePlaceholder with the index {@code ID}.
			 */
			public InstancePlaceholder get(int ID) {
				return this.instancePlaceholders[ID];
			}
			
			
			/**
			 * Returns a random {@linkplain InstancePlaceholder} from the set of managed
			 * InstancePlaceholder objects.
			 * 
			 * @return a randomly selected InstancePlaceholder.
			 */
			public InstancePlaceholder getRandomInstancePlaceholder() {
				return this.instancePlaceholders[rand.nextInt(this.instancePlaceholders.length)];
			}
			
			
			/**
			 * Size of this InstancePlaceholders object, namely how many
			 * {@linkplain InstancePlaceholder} objects this object contains.
			 * 
			 * @return size of this InstancePlaceholders object.
			 */
			public int size() {
				return this.instancePlaceholders.length;
			}
			
			
			/**
			 * Returns a list of all {@linkplain InstancePlaceholder} objects, that do not
			 * know their neighbors yet.
			 * <p>
			 * This list can be used to calculate the neighbors of an
			 * InstancePlaceholder (hereafter named InstancePlaceholder one) in
			 * a more efficient way. In the course of executing the algorithm the
			 * neighbors of all InstancePlaceholder objects in the {@link #data} dataset
			 * must be known. Once the ant wants to calculate the neighbors of
			 * an InstancePlaceholder it can iterate over all
			 * InstancePlaceholder objects to calculate their distance and then
			 * it has to decide whether it is a neighbor or not. But if the
			 * neighbors are already calculated for an InstancePlaceholder (two)
			 * there is no need to include that InstancePlaceholder two in the
			 * calculation for the neighbors for InstancePlaceholder one,
			 * because one was already regarded in calculating the neighbors for
			 * two and the distance was determined. As symmetric distances
			 * between InstancePlaceholder objects are assumed, two can be
			 * excluded in calculating the neighbors for one. The list returned
			 * by this method therefore only contains InstancePlaceholder
			 * objects, that do not fully know their neighbors yet. This way of
			 * calculating the neighbors does not alter the algorithm itself, it
			 * would still be the same if the ant iterates over all other
			 * InstancePlaceholder objects when calculating the neighbors.
			 * 
			 * @return an {@linkplain ArrayList} of {@linkplain InstancePlaceholder} objects, that do not
			 * know their neighbors.
			 */
			public ArrayList<InstancePlaceholder> getUnexploredNeighborhoodList() {
				return this.neighborsUnexploredList;
			}
			
			
			/**
			 * Notifies this object, that the {@linkplain InstancePlaceholder} {@code ip} does know its
			 * neighbors now. So {@code ip} can be excluded in calculating the
			 * neighbors for other InstancePlaceholder objects.
			 * 
			 * @param ip InstancePlaceholder that does know its neighbors.
			 * @see #getUnexploredNeighborhoodList()
			 */
			public void notifyNeighborKnown(InstancePlaceholder ip) {
				this.neighborsUnexploredList.remove(ip);
			}
			
			
			/**
			 * Returns a string representation of this object.
			 * 
			 * @return a string representing this object.
			 * @see java.lang.Object#toString()
			 */
			@Override
			public String toString() {
				return "[ips:" + this.instancePlaceholders.length + "," + this.neighborsUnexploredList.size() + "]";
			}
			
			
			/**
			 * Iterates over all {@linkplain InstancePlaceholder} objects managed by this object.
			 * 
			 * @return an {@linkplain Iterator} over InstancePlaceholder objects.
			 */
			@Override
			public Iterator<InstancePlaceholder> iterator() {
				return new Iterator<InstancePlaceholder>() {
					private int position = 0;
					@Override
					public boolean hasNext() {
						return position < instancePlaceholders.length;
					}
					@Override
					public InstancePlaceholder next() {
						InstancePlaceholder ip = instancePlaceholders[position];
						position++;
						return ip;
					}
				};
			}
			
		}
		
		
		/**
		 * Placeholder for a {@link weka.core.Instance} object.
		 * <p>
		 * InstancePlaceholder objects are agents for {@code weka.core.Instance}
		 * objects, that provide functionality required by the classes of this
		 * clusterer. An InstancePlaceholder object represents exactly one
		 * Instance, so the association is unambiguous.
		 */
		protected class InstancePlaceholder implements Iterable<InstancePlaceholder> { //Similar to GridInstance.
			
			/** Index of the represented {@link weka.core.Instance}. */
			private int ID;
			
			/** Link to the {@linkplain InstancePlaceholders} object that manages this object. */
			protected InstancePlaceholders instancePlaceholders;
			
			/** Link to the represented {@link weka.core.Instance}. */
			protected Instance instance;
			
			/** Cluster assigned to this object. */
			protected Cluster cluster;
			
			/** List of all neighbors. */
			protected ArrayList<InstancePlaceholderNeighborTag> neighbors;
			
			/** Indicates if this InstancePlaceholder object already knows its neighbors. */
			protected boolean neighborsAreKnown = false;
			
			/**
			 * Local similarity value for this InstancePlaceholder object.
			 * <p>
			 * The calculated local similarity is also stored in this object,
			 * although ants do the calculation. It is stored here, to avoid
			 * several calculations of the same value. The result is independent
			 * from the ant doing the calculation. So if another ant goes to
			 * this InstancePlaceholder, it can access the calculation result
			 * done by another ant. The algorithm logic does not change if each
			 * ant has to do its own calculation.
			 */
			protected double foi = 0.0;
			
			/** Indicates if the local similarity was already calculated for this InstancePlaceholder. */
			protected boolean foiIsCalculated = false;
			
			/**
			 * The default constructor for this class.
			 * 
			 * @param ID index of the represented {@linkplain Instance} in the given data
			 *        {@linkplain Instances}.
			 * @param instance the {@linkplain Instance} this InstancePlaceholder should
			 *        represent.
			 * @param instancePlaceholders the {@linkplain InstancePlaceholders} that
			 *        manage this InstancePlaceholder.
			 */
			public InstancePlaceholder(int ID, Instance instance, InstancePlaceholders instancePlaceholders) {
				if (ID < 0) {
					throw new IllegalArgumentException("The instancePlaceholder can not be constructed. The given instance ID is not a valid ID.");
				}
				this.ID = ID;
				this.instancePlaceholders = instancePlaceholders;
				this.instance = instance;
				this.cluster = null;
				this.neighbors = new ArrayList<InstancePlaceholderNeighborTag>();
				this.neighborsAreKnown = false;
				this.foi = 0.0;
				this.foiIsCalculated = false;
			}
			
			
			/**
			 * Tells the index of the represented {@linkplain Instance} in the given data
			 * {@linkplain Instances}.
			 * 
			 * @return index of the represented Instance.
			 */
			public int getID() {
				return this.ID;
			}
			
			
			/**
			 * Returns the {@linkplain Instance} this InstancePlaceholder represents.
			 * 
			 * @return Instance represented by this InstancePlaceholder.
			 */
			public Instance getInstance() {
				return this.instance;
			}
			
			
			/**
			 * Tells if to this InstancePlaceholder already a cluster was
			 * assigned.
			 * 
			 * @return true, if this InstancePlaceholder is already clustered,
			 *         false if not.
			 */
			public boolean hasCluster() {
				return this.cluster instanceof Cluster;
			}
			
			
			/**
			 * Tells if this InstancePlaceholder is clustered in the noise
			 * cluster, that means it was recognized as noise.
			 * 
			 * @return true, if this InstancePlaceholder in the noise cluster,
			 *         false if it is not. The value is also false, if it is not
			 *         clustered.
			 */
			public boolean hasNoiseCluster() {
				return this.cluster instanceof Cluster && this.cluster.isNoiseCluster();
			}
			
			
			/**
			 * Sets the {@linkplain Cluster} to which this InstancePlaceholder belongs.
			 * 
			 * @param cluster the cluster to which this InstancePlaceholder
			 *        should belong to. The null value is also possible, but it
			 *        is not recommended to use it here.
			 * @see #deleteCluster()
			 */
			public void setCluster(Cluster cluster) {
				this.cluster = cluster;
			}
			
			
			/**
			 * Tells the {@linkplain Cluster} to which this InstancePlaceholder belongs.
			 * 
			 * @return the cluster of this InstancePlaceholder or null if it
			 *         is not clustered.
			 */
			public Cluster getCluster() {
				return this.cluster;
			}
			
			
			/**
			 * Deletes the cluster association for this InstancePlaceholder and
			 * make it unclustered.
			 */
			public void deleteCluster() {
				this.cluster = null;
			}
			
			
			/**
			 * Notifies this InstancePlaceholder about a neighbor it has.
			 * <p>
			 * This InstancePlaceholder can also be notified about a neighbor,
			 * even it does not search for neighbors at the moment. Usually it
			 * gets informed about some neighbors before it searches for
			 * neighbors itself. When it already did this search, it is assumed
			 * that it found all neighbors it has, so this method does not
			 * provide any new information. As symmetric neighborhood relations
			 * are assumed not every InstancePlaceholder must run through all
			 * other InstancePlaceholder objects (decended from {@linkplain Instance} objects
			 * in {@linkplain Instances}). For more information the annotations
			 * for {@linkplain InstancePlaceholders#getUnexploredNeighborhoodList()}.
			 * 
			 * @param neighbor a neighborhood relation found for this
			 *        InstancePlaceholder.
			 * @see InstancePlaceholders#getUnexploredNeighborhoodList()
			 */
			public void preTellNeighbor(InstancePlaceholderNeighborTag neighbor) {
				if (this.neighborsAreKnown) {
					return;
				}
				if (this.neighbors.contains(neighbor)) {
					return;
				}
				this.neighbors.add(neighbor);
			}
			
			
			/**
			 * Notifies this InstancePlaceholder about all its neighbors. The
			 * final neighborhood list, containing all neighbors must be
			 * provided. Then it is assumed that all neighbors are known.
			 * 
			 * @param neighbors complete list of all neighborhood relations.
			 */
			public void setNeighbors(ArrayList<InstancePlaceholderNeighborTag> neighbors) {
				this.neighbors = neighbors;
				this.neighborsAreKnown = true;
			}
			
			
			/**
			 * Tells all currently known neighborhood relations of this
			 * InstancePlaceholder.
			 * 
			 * @return An {@linkplain ArrayList} containing all currently known neighborhood
			 * relations.
			 */
			public ArrayList<InstancePlaceholderNeighborTag> getNeighbors() {
				return this.neighbors;
			}
			
			
			/**
			 * Tells the size of the known neighborhood, namely how many
			 * neighbors this InstancePlaceholder currently has.
			 * 
			 * @return neighbor count of this InstancePlaceholder.
			 */
			public int getNeighborhoodSize() {
				return this.neighbors.size();
			}
			
			
			/**
			 * Tells if this InstancePlaceholder assumes that it knows all of
			 * its neighbors.
			 * 
			 * @return true if this InstancePlaceholder knows all its neighbors,
			 *         false otherwise.
			 */
			public boolean neighborsAreKnown() {
				return this.neighborsAreKnown;
			}
			
			
			/**
			 * Sets the local similarity value of this InstancePlaceholder. It
			 * is usually calculated by an {@linkplain Ant}.
			 * 
			 * @param foi local similarity value for this InstancePlaceholder.
			 * @see #foi
			 */
			public void setFoi(double foi) {
				this.foi = foi;
				this.foiIsCalculated = true;
			}
			
			
			/**
			 * Tells the local similarity value of this InstancePlaceholder. The
			 * initial value, if the setter was not invoked yet, is 0.0.
			 * 
			 * @return local similarity value of this InstancePlaceholder.
			 * @see #foi
			 */
			public double getFoi() {
				return this.foi;
			}
			
			
			/**
			 * Tells if the local similarity value of this InstancePlaceholder
			 * is known.
			 * 
			 * @return true, if the local similarity was set before, false if
			 *         the setter was not invoked yet.
			 * @see #foi
			 */
			public boolean foiIsCalculated() {
				return this.foiIsCalculated;
			}
			
			
			/**
			 * Tells if this object equals the given object, by customized criteria.
			 * 
			 * @param obj the object to compare with.
			 * @return true, if this object equals the compared {@code obj} object.
			 * @see java.lang.Object#equals
			 */
			@Override
			public boolean equals(Object obj) {
				if (!(obj instanceof InstancePlaceholder)) {
					return false;
				}
				return this.ID == ((InstancePlaceholder) obj).ID;
			}
			
			
			/**
			 * Returns a string representation of this object.
			 * 
			 * @return a string representing this object.
			 * @see java.lang.Object#toString()
			 */
			@Override
			public String toString() {
				return "[ip:" + this.ID + "]";
			}
			
			
			/**
			 * Iterates over all {@linkplain InstancePlaceholder} neighbors of this object.
			 * 
			 * @return an {@linkplain Iterator} over InstancePlaceholder neighbor objects.
			 */
			@Override
			public Iterator<InstancePlaceholder> iterator() {
				return new Iterator<InstancePlaceholder>() {
					private int position = 0;
					@SuppressWarnings("unchecked")
					private ArrayList<InstancePlaceholderNeighborTag> list = (ArrayList<InstancePlaceholderNeighborTag>) neighbors.clone();
					@Override
					public boolean hasNext() {
						return position < list.size();
					}
					@Override
					public InstancePlaceholder next() {
						InstancePlaceholder ip = list.get(position).instancePlaceholder;
						position++;
						return ip;
					}
				};
			}
			
		}
		
		
		/**
		 * Class for storing a neighborhood relation. A neighborhood relation
		 * consists of an {@linkplain InstancePlaceholder} (the neighbor) and a double value
		 * representing the distance to that neighbor.
		 * <p>
		 * This is a passive class, that means this class is used by other
		 * classes, but does not act on its own.
		 */
		protected final class InstancePlaceholderNeighborTag {
			
			/** Neighbor of the relation. */
			public InstancePlaceholder instancePlaceholder;
			
			/** Distance to the neighbor {@linkplain #instancePlaceholder}. */
			public double distance;
			
			
			/**
			 * The default constructor of this class.
			 * 
			 * @param ip neighbor of the neighborhood relation.
			 * @param distance distance to that neighbor {@code ip}.
			 */
			public InstancePlaceholderNeighborTag(InstancePlaceholder ip, double distance) {
				this.instancePlaceholder = ip;
				this.distance = distance;
			}
			
			
			//			public InstancePlaceholder getInstancePlaceholder() {
			//				return this.instancePlaceholder;
			//			}
			
			
			//			public double getDistance() {
			//				return this.distance;
			//			}
			
			
			/**
			 * Tells if this object equals the given object, by customized criteria.
			 * 
			 * @param obj the object to compare with.
			 * @return true, if this object equals the compared {@code obj} object.
			 * @see java.lang.Object#equals
			 */
			@Override
			public boolean equals(Object obj) {
				if (!(obj instanceof InstancePlaceholderNeighborTag)) {
					return false;
				}
				return this.instancePlaceholder.getID() == ((InstancePlaceholderNeighborTag) obj).instancePlaceholder.getID();
			}
			
			
			/**
			 * Returns a string representation of this object.
			 * 
			 * @return a string representing this object.
			 * @see java.lang.Object#toString()
			 */
			@Override
			public String toString() {
				return "[nt:" + this.instancePlaceholder + "<->" + this.distance + "]";
			}
			
		}
		
		
		/**
		 * Ants (agents) that perform the clustering task.
		 * <p>
		 * Ants walk over the {@linkplain InstancePlaceholder} objects and also
		 * calculate their neighbors, local similarity values and assign
		 * {@linkplain Cluster}s to them.
		 */
		protected class Ant {
			
			/**
			 * Compares two neighborhood relations by their distance.
			 */
			protected class InstancePlaceholderNeighborTagDistanceComparator implements Comparator<InstancePlaceholderNeighborTag> {
				
				/**
				 * Performs the comparison task of two neighborhood relations.
				 * 
				 * @param one first neighborhood relation of the comparison
				 * @param two second neighborhood relation of the comparison
				 * @return 1, if the distance of {@code one} is greater than the
				 *         distance of {@code two}, -1 if the distance of {@code two} is greater
				 *         than the distance of {@code one} and 0 if both distances
				 *         are the same.
				 */
				@Override
				public int compare(InstancePlaceholderNeighborTag one, InstancePlaceholderNeighborTag two) {
					if (one.distance > two.distance) { return 1; }
					else if (one.distance < two.distance) { return -1; }
					else { return 0; }
				}
				
			}
			
			/** Link to the {@linkplain AntHill} that manages this Ant. */
			protected AntHill antHill = null;
			
			/** Current position of this Ant. */
			protected InstancePlaceholder position = null;
			
			/** Destination to which this ant wants to go, if any. */
			protected InstancePlaceholder destination = null;
			
			/** View range of this ant. */
			protected double viewRange = 0.0;
			
			/** Tolerance of this ant for a raise of the local similarity value between two {@linkplain InstancePlaceholder}s. */
			protected double foiRaiseTolerance = 0.0;
			
			/** All instances with a calculated local similarity value belwow this threshold are regarded as noise {@linkplain InstancePlaceholder}s by this ant. */
			protected double foiNoiseThreshold = 0.0;
			
			/** If the ant could not work for this amount of calls in series, because it walked on a already clustered {@linkplain InstancePlaceholder}, it assumes the clustering task is done and shuts down. */
			protected int autoShutdownAfterNumIdleCalls = -1;
			
			/** How many times the ant was on a already clustered {@linkplain InstancePlaceholder} in series. Once the ant finds a InstancePlaceholder, that is not clustered this value resets to 0. */
			protected int idleCalls = 0;
			
			/** Indicate whether this ant is active or shut down. */
			protected boolean isActive = false;
			
			/**
			 * The remembered {@linkplain InstancePlaceholder} of this ant.
			 * <p>
			 * Ants can only remember InstancePlaceholder objects and relate to
			 * them like this, but not pick them up and carry them around,
			 * because they walk in the same space where the {@linkplain Instance} objects
			 * are located and every picking and dropping action would alter the
			 * {@linkplain Instances}. Picking {@linkplain Instance} objects up would mean to
			 * remove them temporarily from Instances and dropping would mean to
			 * change their attributes in order to relocate them in the space.
			 * Of course this is not acceptable. This value corresponds to a
			 * carry capacity of an ant.
			 */
			private InstancePlaceholder remember = null;
			
			
			/**
			 * The default constructor of this class.
			 * 
			 * @param antHill {@linkplain AntHill} that manages this ant
			 * @param viewRange relative view range distance for this ant
			 * @param foiRaiseTolerance tolerance of this ant for a raise of the
			 *        local similarity between two InstancePlaceholder objects.
			 * @param foiNoiseThreshold threshold for recognizing
			 *        InstancePlaceholder objects as noise. Set to -1 to not
			 *        seek for noise InstancePlaceholder objects.
			 * @param autoShutdownAfterNumIdleCalls after how many steps on
			 *        clustered InstancePlaceholder objects in series this ant
			 *        has to shut down. If no valid value is given it defaults
			 *        to 100.
			 * @see #antHill
			 * @see #viewRange
			 * @see #foiRaiseTolerance
			 * @see #foiNoiseThreshold
			 * @see #autoShutdownAfterNumIdleCalls
			 */
			public Ant(AntHill antHill, double viewRange, double foiRaiseTolerance, double foiNoiseThreshold, int autoShutdownAfterNumIdleCalls) {
				this.antHill = antHill;
				this.position = null;
				this.viewRange = viewRange > 0 ? viewRange : 0.12;
				this.foiRaiseTolerance = foiRaiseTolerance >= 0 ? foiRaiseTolerance : 0;
				this.foiNoiseThreshold = foiNoiseThreshold >= 0 ? foiNoiseThreshold : -1;
				this.walk(); //Important to give the ant an initial position.
				this.autoShutdownAfterNumIdleCalls = autoShutdownAfterNumIdleCalls > 0 ? autoShutdownAfterNumIdleCalls : 100;
				this.idleCalls = 0;
				this.isActive = true;
			}
			
			
			/**
			 * Tells if this ant is active.
			 * 
			 * @return true, if this ant is active, false if not, e.g. it was
			 *         shut down.
			 */
			public boolean isActive() {
				return this.isActive;
			}
			
			
			/**
			 * Calls this ant to do something (to work and to walk).
			 */
			public void call() {
				if (!this.isActive) {
					this.release();
					return;
				}
				try {
					this.work();
					this.walk();
				}
				catch (Exception exception) {
					this.release();
					throw exception;
				}
			}
			
			
			/**
			 * Clusters {@linkplain InstancePlaceholder} objects. If the
			 * neighbors are not known or the local similarity is not calculated
			 * for this or any neighbor InstancePlaceholder this task is done at
			 * first. Only if any of these tasks is already done and the
			 * environment of the current InstancePlaceholder is explored, the
			 * clustering task can start. If the local similarity does not
			 * differ more than {@linkplain #foiRaiseTolerance} from a close cluster,
			 * clusters must be merged.
			 */
			protected void work() {
				if (!this.position.neighborsAreKnown() || !this.position.foiIsCalculated()) {
					this.release();
					this.take(this.position);
					if (!this.getRememberedInstancePlaceholder().neighborsAreKnown()) {
						this.calculateNeighborsAtPosition();
					}
					if (!this.getRememberedInstancePlaceholder().foiIsCalculated()) {
						this.calculateFoiAtPosition();
					}
					this.release();
				}
				InstancePlaceholder testExplore = this.testExplorePosition(this.position);
				if (testExplore instanceof InstancePlaceholder) {
					this.destination = testExplore; //Destination is not really necessary, but is used to avoid seeking testExplorePosition twice (later in walk()).
					this.release();
					return; //Exploration has higher priority than calculating the cluster.
				} //No explore priority.
				if (this.foiNoiseThreshold >= 0 && this.position.getFoi() <= this.foiNoiseThreshold) {
					this.position.setCluster(this.antHill.clusters.getNoiseCluster());
					this.antHill.clusters.getNoiseCluster().add(this.position);
					return;
				}
				if (!this.holdsInstancePlaceholder()) {
					this.take(this.position); //Usual take.
					if (this.position.hasCluster()) {
						this.release();
					}
					return;
				} //this.holdsInstancePlaceholder()
				this.idleCalls = 0; // The ant can work.
				if (!this.position.hasCluster()) {
					this.release(); //Treat the current position first (new art top).
					this.take(this.position);
					Cluster cluster = this.antHill.getClusters().makeNewCluster();
					cluster.setStart(this.getRememberedInstancePlaceholder());
					cluster.add(this.getRememberedInstancePlaceholder());
					this.getRememberedInstancePlaceholder().setCluster(cluster);
					this.release();
					return;
				}
				this.getRememberedInstancePlaceholder().setCluster(this.position.getCluster());
				this.position.getCluster().add(this.getRememberedInstancePlaceholder());
				if (this.foiRaiseTolerance > 0) {
					for (InstancePlaceholder neighbor : this.getRememberedInstancePlaceholder()) {
						if (!neighbor.hasCluster()) {
							if (neighbor.getFoi() < this.getRememberedInstancePlaceholder().getFoi()) {
								break;
							}
							else {
								continue;
							}
						}
						if (neighbor.hasNoiseCluster()) {
							continue;
						}
						if (neighbor.getCluster().equals(this.getRememberedInstancePlaceholder().getCluster())) {
							continue;
						} //neighbor.hasCluster() && !neighbor.getCluster().equals(this.getRememberedInstancePlaceholder().getCluster())
						InstancePlaceholder aStart = ((InstancePlaceholder) neighbor.getCluster().getStart());
						InstancePlaceholder bStart = ((InstancePlaceholder) this.getRememberedInstancePlaceholder().getCluster().getStart());
						if (aStart.getFoi() > bStart.getFoi() && bStart.getFoi() <= this.getRememberedInstancePlaceholder().getFoi() + this.foiRaiseTolerance) {
							this.changeCluster(bStart.getCluster(), aStart.getCluster());
						}
						else if (aStart.getFoi() <= bStart.getFoi() && aStart.getFoi() <= this.getRememberedInstancePlaceholder().getFoi() + this.foiRaiseTolerance) {
							this.changeCluster(aStart.getCluster(), bStart.getCluster());
						}
						else { }
						break; //Done, do not seek for another cluster to merge.
					}
				}
				this.release();
				return;
			}
			
			
			/**
			 * Moves the ant to another GridInstance.
			 * <p>
			 * The walk direction depends on 3 factors:
			 * <ul>
			 * <li>if the ant has a destination it goes there</li>
			 * <li>if the ant holds an {@linkplain InstancePlaceholder} it searches for
			 * clusters and stops when reaching one or could not find any in a
			 * dense environment</li>
			 * <li>if the ant does not hold an InstancePlaceholder it walks
			 * randomly.</li>
			 * 
			 * @see #destination
			 * @see #holdsInstancePlaceholder()
			 */
			protected void walk() { //Avoid take or release.
				if (!(this.position instanceof InstancePlaceholder)) {
					this.position = this.antHill.getInstancePlaceholders().getRandomInstancePlaceholder();
					return;
				}
				if (this.destination instanceof InstancePlaceholder) {
					this.position = this.destination;
					this.destination = null;
				}
				this.destination = null; //Walking does not have a destination when not found above.
				if (!this.position.neighborsAreKnown() || !this.position.foiIsCalculated()) {
					return;
				} //So far exploration done here.
				InstancePlaceholder exploreTest = this.testExplorePosition(this.position);
				if (exploreTest instanceof InstancePlaceholder) {
					this.position = exploreTest;
					return;
				}
				if (this.holdsInstancePlaceholder()) {
					boolean walk = true; //seek
					while (walk) {
						walk = false;
						if (this.position.hasCluster() && !this.position.hasNoiseCluster()) {
							return;
						}
						for (InstancePlaceholder neighbor : this.position) {
							if (neighbor.getFoi() > this.position.getFoi()) {
								this.position = neighbor;
								walk = true;
								break; //Continue with neighbor.
							}
						}
					}
				}
				else {
					if (this.position.hasCluster() && this.idleCalls < this.autoShutdownAfterNumIdleCalls) {
						this.position = this.antHill.getInstancePlaceholders().getRandomInstancePlaceholder();
						this.idleCalls++;
					}
					if (this.idleCalls >= this.autoShutdownAfterNumIdleCalls) {
						this.idleCalls = 0;
						this.shutdown();
						return;
					}
//					this.idleCalls = 0; //Faster approach (this.idleCalls not necessary in work() then):
//					while (this.position.hasCluster() && this.idleCalls < this.autoShutdownAfterNumIdleCalls) {
//						this.position = this.antHill.getInstancePlaceholders().getRandomInstancePlaceholder();
//						this.idleCalls++;
//					}
//					if (this.idleCalls >= this.autoShutdownAfterNumIdleCalls) { //While ended for this reason?
//						this.idleCalls = 0;
//						this.shutdown();
//						return;
//					}
//					this.idleCalls = 0;
				}
			}
			
			
			/**
			 * Direct the ant to merge two {@linkplain Cluster}s into one.
			 * <p>
			 * In this implementation the ant receives information from the
			 * cluster about its members. When the ant explores the environment
			 * itself, it runs slower, but with less information. In another
			 * implementation the ant could check all neighbors from a starting
			 * {@linkplain InstancePlaceholder} for example, and add them to a internal todo
			 * list if they have the cluster {@code from}. Then change the
			 * cluster and do like this with the other InstancePlaceholder
			 * objects on the todo list.
			 * 
			 * @param from the starting cluster to be merged into {@code to}. It
			 *        dissolves in the {@code changeCluster} process and its
			 *        InstancePlaceholder members are moved to the cluster {@code to}.
			 * @param to the target cluster to which the cluster {@code from} is merged.
			 */
			protected void changeCluster(Cluster from, Cluster to) {
				if (from.isNoiseCluster() || to.isNoiseCluster()) {
					throw new RuntimeException("One of the clusters to change is a noise cluster. Can not change from or to a noise cluster.");
				}
				if (from.equals(to)) {
					if (m_Debug) {
						System.out.println("# ! Can not change clusters, because the two clusters are regarded as the same.");
					}
					return;
				}
				InstancePlaceholder root = this.position;
				for (InstancePlaceholder pos : from) {
					this.position = pos; //Also possible without this.position change, but this simulates walking of the ant.
					to.add(pos);
					pos.setCluster(to);
					from.remove(pos);
				}
				this.position = root; //Go back to starting position.
			}
			
			
			/**
			 * Tells the next {@linkplain InstancePlaceholder} position that should be
			 * explored before clustering the current position {@code position}. So the
			 * returned InstancePlaceholder can be prioritized over {@code position}.
			 * 
			 * @param position 
			 * @return the InstancePlaceholder that should be processed
			 *         before {@code position} or null if there is no
			 *         InstancePlaceholder to explore with higher priority.
			 */
			protected InstancePlaceholder testExplorePosition(InstancePlaceholder position) {
				if (!(position instanceof InstancePlaceholder)) {
					return null;
				}
				if (!position.neighborsAreKnown() || !position.foiIsCalculated()) {
					return position;
				}
				for (InstancePlaceholder neighbor : position) {
					if (!neighbor.neighborsAreKnown()) {
						return neighbor;
					}
					if (!neighbor.foiIsCalculated()) {
						return neighbor;
					}
				}
				return null; //No explore position found.
			}
			
			
			/**
			 * Tells if this ant is linked to a {@linkplain InstancePlaceholder}, if it
			 * is remembering one (as if carrying it).
			 * 
			 * @return true, if the ant remembers an InstancePlaceholder, false
			 *         otherwise.
			 */
			public final boolean holdsInstancePlaceholder() {
				return (this.remember instanceof InstancePlaceholder);
			}
			
			
			/**
			 * Tells the {@linkplain InstancePlaceholder} this ant remembers at the
			 * moment.
			 * 
			 * @return the InstancePlaceholder this ant remembers or null if the
			 *         ant does not remember a InstancePlaceholder at the
			 *         moment.
			 */
			public final InstancePlaceholder getRememberedInstancePlaceholder() {
				return this.remember;
			}
			
			
			/**
			 * Remembers another {@linkplain InstancePlaceholder} as if the ant would
			 * pick it up. It releases any previously hold InstancePlaceholder.
			 * 
			 * @param ip the InstancePlaceholder the ant should remember now
			 * @return true on success, false otherwise.
			 */
			protected final boolean take(InstancePlaceholder ip) {
				this.release();
				this.remember = ip;
				return true;
			}
			
			
			/**
			 * Direct the ant to forget an {@linkplain InstancePlaceholder} it possibly
			 * remembers.
			 * 
			 * @return true, if the ant does not remember an InstancePlaceholder
			 *         right now, no matter if it successfully forgot an
			 *         InstancePlaceholder it remembered before or did not
			 *         remember a InstancePlaceholder before. False on failure.
			 */
			protected final boolean release() {
				if (this.remember instanceof InstancePlaceholder) {
					this.remember = null;
				}
				return true;
			}
			
			
			/**
			 * Does calculate all neighbors for the current position of this
			 * ant.
			 */
			private void calculateNeighborsAtPosition() { //See also Handl/Meyer 2002, p. 916.
				Instance instance = this.position.getInstance();
				ArrayList<InstancePlaceholder> candidates = this.antHill.getInstancePlaceholders().getUnexploredNeighborhoodList();
				ArrayList<InstancePlaceholderNeighborTag> neighbors = new ArrayList<InstancePlaceholderNeighborTag>();
				ArrayList<InstancePlaceholderNeighborTag> preToldNeighbors = this.position.getNeighbors();
				double dJV = 0;
				if (optn_distanceFunctionAlign) {
					for (InstancePlaceholder candidate : candidates) {
						double distance = optn_distanceFunction.distance(instance, candidate.getInstance());
						dJV = distance;
					}
				}
				for (InstancePlaceholder candidate : candidates) {
					double distance = optn_distanceFunction.distance(instance, candidate.getInstance());
					if (distance <= this.viewRange && !this.position.equals(candidate)) {
						if (optn_distanceFunctionAlign) { distance = distance / dJV; }
						neighbors.add(new InstancePlaceholderNeighborTag(candidate, distance));
						candidate.preTellNeighbor(new InstancePlaceholderNeighborTag(this.position, distance));
					}
				}
				for (InstancePlaceholderNeighborTag neighbor : preToldNeighbors) { //Now add the pre told neighbors, too.
					if (!neighbors.contains(neighbor)) {
						neighbors.add(neighbor);
					}
				}
				neighbors.sort(new InstancePlaceholderNeighborTagDistanceComparator());
				this.position.setNeighbors(neighbors);
				this.antHill.getInstancePlaceholders().notifyNeighborKnown(this.position);
			}
			
			
			/**
			 * Does calculate the local similarity value for the current
			 * position of this ant.
			 */
			private void calculateFoiAtPosition() {
				double alpha = this.antHill.getAlpha();
				ArrayList<InstancePlaceholderNeighborTag> neighbors = this.position.getNeighbors();
				double sum = 0.0;
				double foi = 0.0;
				int size = neighbors.size();
				if (size == 0) { //When there are no neighbors there is no further need to calculate foi.
					this.position.setFoi(0.0);
					return;
				}
				for (InstancePlaceholderNeighborTag neighborTag : neighbors) {
					sum = sum + (1.0 - (neighborTag.distance / alpha));
				}
				foi = (sum / size) * 10; //foi = (sum / size) * 1; 10 is just scaling factor. Can also be 100, ...
				if (foi < 0) {
					foi = 0;
				}
				this.position.setFoi(foi);
			}
			
			
			/**
			 * Shuts the ant down, so it is in a state where it does not
			 * contribute to the clustering process anymore and can be securely
			 * deleted for example.
			 */
			public void shutdown() {
				if (!this.isActive) {
					return;
				}
				this.release();
				this.isActive = false;
			}
			
			
			/**
			 * Returns a string representation of this object.
			 * 
			 * @return a string representing this object.
			 * @see java.lang.Object#toString()
			 */
			@Override
			public String toString() {
				return "[ant:" + "," + this.isActive + this.position + "," + this.viewRange + "," + this.idleCalls + "]";
			}
			
		}
		
		/** Alpha value for the calculation of the local similarity. */
		protected double alpha;
		
		/** The {@linkplain InstancePlaceholders} object used by this AntHill. */
		protected InstancePlaceholders instancePlaceholders;
		
		/** The {@linkplain Clusters} object used by this AntHill. */
		protected Clusters clusters;
		
		/** All {@linkplain Ant}s in this AntHill. */
		protected ArrayList<Ant> ants;
		
		/** How many ant cycles were already executed. */
		protected int antCycles = 0;
		
		/** Indicates whether this ant colony is shut down or running. */
		private boolean isActive = true;
		
		/** How many {@linkplain Ant}s should be called per antCycle. */
		private int antsCallPerCycle = 0;
		
		/** How many {@linkplain InstancePlaceholder}s remained unclustered after the clustering process is done and the AntHill was shut down. */
//		private int numUnclusteredInLastAssignment = -1;
		
		
		/** The default constructor of this class. */
		public AntHill() {
			this.clusters = new Clusters();
			this.ants = new ArrayList<Ant>();
			this.isActive = true;
		}
		
		
		/**
		 * Initializes the AntHill with start parameters for a clustering task.
		 * 
		 * @param alpha the alpha value to use during clustering
		 * @param data the {@linkplain Instances} to be clustered by the {@linkplain Ant}s
		 *        in this AntHill.
		 * @param numAnts how many ants should work on the clustering task
		 * @param antsCallPerCycle how many ants to call in one ant cycle
		 * @param antViewRange the view range of the ants in this AntHill
		 * @param antsAssumeGlobalAfterNumCalls after how many ant calls in
		 *        series an observed behavior is assumed to be global.
		 * @param foiRaiseTolerance tolerated raise of the local similarity
		 *        during the clustering process
		 * @param foiNoiseThreshold threshold for regarding InstancePlaceholder
		 *        objects with a low local similarity value as noise.
		 * @see #alpha
		 * @see #instancePlaceholders
		 * @see #ants
		 * @see #antsCallPerCycle
		 * @see Ant#viewRange
		 * @see Ant#autoShutdownAfterNumIdleCalls
		 * @see Ant#foiRaiseTolerance
		 * @see Ant#foiNoiseThreshold
		 */
		public void initialize(double alpha, Instances data, int numAnts, int antsCallPerCycle, double antViewRange, int antsAssumeGlobalAfterNumCalls, double foiRaiseTolerance, double foiNoiseThreshold) {
			this.alpha = alpha;
			this.instancePlaceholders = new InstancePlaceholders(data);
			this.ants = new ArrayList<Ant>(numAnts);
			for (int i = 0; i < numAnts; i++) {
				this.ants.add(new Ant(this, antViewRange, foiRaiseTolerance, foiNoiseThreshold, antsAssumeGlobalAfterNumCalls > 0 ? antsAssumeGlobalAfterNumCalls : -1));
			}
			this.antCycles = 0;
			this.isActive = true;
			this.antsCallPerCycle = antsCallPerCycle;
		}
		
		
		/**
		 * Sets the alpha value to be used for clustering.
		 * 
		 * @param alpha alpha value to be used for clustering.
		 * @see #initialize(double, Instances, int, int, double, int, double, double)
		 */
		public void setAlpha(double alpha) {
			this.alpha = alpha;
		}
		
		
		/**
		 * Gets the current alpha value for clustering.
		 * 
		 * @return alpha value for clustering.
		 */
		public double getAlpha() {
			return this.alpha;
		}
		
		
		/**
		 * Tells if this AntHill is active.
		 * 
		 * @return true, if the AntHill is active, false if not, e.g. if it was
		 *         shut down.
		 */
		public boolean isActive() {
			return this.isActive;
		}
		
		
		/**
		 * Tells how many ant cycles were already executed.
		 * 
		 * @return number of ant cycles already executed.
		 */
		public int getAntCycles() {
			return this.antCycles;
		}
		
		
		/**
		 * Returns all {@linkplain InstancePlaceholder}s managed by this
		 * AntHill in one single {@linkplain InstancePlaceholders} object.
		 * 
		 * @return InstancePlaceholders containing all managed
		 * InstancePlaceholder objects.
		 */
		protected InstancePlaceholders getInstancePlaceholders() {
			return this.instancePlaceholders;
		}
		
		
		/**
		 * Returns all {@linkplain Cluster}s managed by this
		 * AntHill in one single {@linkplain Clusters} object.
		 * 
		 * @return Clusters object containing all managed Cluster objects that
		 *         are managed by this AntHill.
		 */
		protected Clusters getClusters() {
			return this.clusters;
		}
		
		
		/**
		 * Runs one ant cycle.
		 */
		public void runAntCycle() {
			if (!this.isActive()) {
				return;
			}
			ArrayList<Ant> activeAnts = new ArrayList<Ant>();
			for (Ant ant : this.ants) {
				if (ant.isActive()) {
					activeAnts.add(ant);
				}
			}
			int size = activeAnts.size();
			if (size == 0) {
				this.shutdown();
				return;
			} //Else:
			for (int i = 0; i < this.antsCallPerCycle; i++) {
				activeAnts.get(rand.nextInt(size)).call();
			}
			this.antCycles++;
		}
		
		
		/**
		 * Shuts this AntHill down after a clustering task.
		 */
		public void shutdown() {
			if (!this.isActive) {
				return;
			}
			for (Ant ant : this.ants) {
				ant.shutdown();
			}
			this.isActive = false;
		}
		
		
		/**
		 * Tells the result of an executed clustering task.
		 * 
		 * @param maxClusterNum maximum number of clusters in the returned
		 *        result. A limitation is set, if this parameter is not -1.
		 * @return an array with InstancePlaceholder index to cluster index
		 *         association.
		 * @throws RuntimeException if the AntHill is still active, and
		 *         therefore the clustering task may not be finished yet.
		 */
		public int[] getClusterAssignments(int maxClusterNum) throws RuntimeException {
			
			/**
			 * Class that reduces the cluster number in the result.
			 */
			class ClusterNumReducer {
				
				/** Indicates if this class is enabled or not. */
				private boolean enabled = true;
				
				/** Leading clusters that can be found in the reduced result. */
				private ArrayList<Cluster> leadingClusters;
				
				/** IDs of the leading clusters. */
				private int[] leadingClusterIDs;
				
				
				/**
				 * The default constructor of this class.
				 * 
				 * @param clusters the starting set of {@linkplain Clusters}
				 * @param maxClusterNum upper limit for clusters in the result.
				 */
				public ClusterNumReducer(Clusters clusters, int maxClusterNum) {
					this.leadingClusters = new ArrayList<Cluster>();
					if (clusters.size() <= maxClusterNum || maxClusterNum < 1) {
						this.enabled = false;
						return;
					}
					else {
						this.enabled = true;
					}
					while (leadingClusters.size() < maxClusterNum) { //Find leading clusters.
						Cluster bestCluster = null;
						int bestSize = 0;
						for (Cluster cluster : clusters) {
							if ((!(bestCluster instanceof Cluster) || cluster.size() > bestSize) && !leadingClusters.contains(cluster)) {
								bestCluster = cluster;
								bestSize = cluster.size();
							}
						}
						leadingClusters.add(bestCluster); //No null check required, if there are more clusters in clusters than allowed, one must be found.
					}
					this.leadingClusters.trimToSize();
					this.leadingClusterIDs = new int[this.leadingClusters.size()];
					int LI = 0;
					for (Cluster cluster : leadingClusters) {
						this.leadingClusterIDs[LI] = cluster.getID();
						LI++;
					}
				}
				
				
				/**
				 * Relabel the ID of the cluster to which {@code ip} belongs to another
				 * cluster ID, if necessary.
				 * 
				 * @param ip the {@linkplain InstancePlaceholder} for which the cluster
				 *        association should be changed in the result, if
				 *        necessary.
				 * @return the new cluster ID for this {@code ip} or the original
				 *         one, if no change of the ID was necessary.
				 */
				public int getClusterRelabeledID(InstancePlaceholder ip) {
					if (!ip.hasCluster() || ip.hasNoiseCluster()) {
						return unassignedClusterID;
					}
					int cID = ip.getCluster().getID();
					if (!this.enabled) {
						return cID;
					}
					int size = this.leadingClusterIDs.length;
					for (int i = 0; i < size; i++) {
						if (this.leadingClusterIDs[i] == cID) {
							return cID;
						}
					} //The ip is not part of a leading cluster. But as the cluster number should be reduced it must be matched to one.
					for (InstancePlaceholder neighbor : ip) {
						if (!neighbor.hasCluster() || neighbor.hasNoiseCluster()) {
							continue;
						}
						int ncID = neighbor.getCluster().getID();
						for (int i = 0; i < size; i++) {
							if (this.leadingClusterIDs[i] == ncID) { //A neighbor of ip is part of a leading cluster. Neighbors must be sorted.
								return ncID;
							}
						}
					} //Still did not find a matching leading cluster.
					Cluster bestCluster = null;
					double bestDistance = 0.0;
					for (Cluster cluster : this.leadingClusters) { //Try to find the closest leading cluster by start instance (ca. centroid) now.
						double distance = optn_distanceFunction.distance(ip.getInstance(), ((InstancePlaceholder) cluster.getStart()).getInstance());
						if (!(bestCluster instanceof Cluster) || bestDistance > distance) {
							bestCluster = cluster;
							bestDistance = distance;
						}
					}
					return bestCluster.getID();
				}
				
			}
			
			
			/**
			 * Class to trim the cluster indexes in the result.
			 * <p>
			 * When merging clusters gaps arise in the cluster index numbers,
			 * because InstancePlaceholder objects are moved from clusters to
			 * other clusters. These clusters, from which the
			 * InstancePlaceholder objects are removed, remain empty and when
			 * retrieving the cluster indexes there can be gaps in the order of
			 * this cluster indexes. This class trims the cluster index order,
			 * so that an ascending order of cluster indexes without gaps
			 * result.
			 */
			class ClusterIDCompactifier {
				
				/**
				 * Association class for linking a starting cluster index to
				 * a cluster index from the compact order, that produces no
				 * gaps.
				 */
				class Association {
					
					/** Previous cluster index. */
					public int clusterID;
					
					/** New cluster index in the order without gaps. */
					public int compactifiedClusterID;
					
					/**
					 * The default constructor for this class.
					 * 
					 * @param clusterID given cluster index
					 * @param compactifiedClusterID new cluster index.
					 */
					public Association(int clusterID, int compactifiedClusterID) {
						this.clusterID = clusterID;
						this.compactifiedClusterID = compactifiedClusterID;
					}
					
				}
				
				/** Next index from the compact order to be assigned to a cluster. */
				private int nextCompactifiedClusterID = -1;
				
				/** Set of all known {@linkplain Association}s. */
				private ArrayList<Association> knownAssociations = null;
				
				/**
				 * The default constructor for this class.
				 */
				@SuppressWarnings("unused")
				public ClusterIDCompactifier() {
					this.nextCompactifiedClusterID = unassignedClusterID == 0 ? 1 : 0;
					this.knownAssociations = new ArrayList<Association>();
					this.knownAssociations.add(new Association(unassignedClusterID, unassignedClusterID));
				}
				
				
				/**
				 * Returns a cluster index from the order of compact cluster
				 * indexes for a given starting cluster index.
				 * 
				 * @param cID starting cluster index to find a cluster index
				 *        from the compact order for
				 * @return a cluster index from the compact order.
				 */
				public int getCompactifiedClusterID(int cID) {
					if (cID == unassignedClusterID) {
						return unassignedClusterID;
					}
					Association association = this.getAssociationFor(cID);
					if (association == null) {
						association = new Association(cID, nextCompactifiedClusterID);
						this.knownAssociations.add(association);
						do {
							nextCompactifiedClusterID++;
						}
						while (nextCompactifiedClusterID == unassignedClusterID);
					}
					return association.compactifiedClusterID;
				}
				
				
				/**
				 * Get a cluster index association for a given previous cluster
				 * index.
				 * 
				 * @param cID the previous cluster index to retrieve the
				 *        association for
				 * @return the association where {@code cID} is the previous cluster
				 *         index, or null if no matching association was found.
				 */
				protected Association getAssociationFor(int cID) {
					for (Association association : this.knownAssociations) {
						if (association.clusterID == cID) {
							return association;
						}
					}
					return null;
				}
				
			}
			
			if (this.isActive()) {
				throw new RuntimeException("To get the cluster assignments from the AntHill it must be shut down first.");
			}
			ClusterNumReducer clusterNumReducer = new ClusterNumReducer(this.clusters, maxClusterNum);
			ClusterIDCompactifier clusterIDCompactifier = new ClusterIDCompactifier();
			int size = this.instancePlaceholders.size();
			int[] assignments = new int[size];
//			this.numUnclusteredInLastAssignment = 0;
			for (int i = 0; i < size; i++) {
				assignments[i] = clusterIDCompactifier.getCompactifiedClusterID(clusterNumReducer.getClusterRelabeledID(this.instancePlaceholders.get(i)));
//				if (assignments[i] == unassignedClusterID) {
//					this.numUnclusteredInLastAssignment++;
//				}
			}
			return assignments;
		}
		
		
		/**
		 * Tell how many {@linkplain InstancePlaceholder}s remained unclustered during the
		 * recent clustering task.
		 * 
		 * @return number of unclustered InstancePlaceholder objects.
		 */
//		public int getCountUnclusteredInLastAssignment() {
//			return this.numUnclusteredInLastAssignment;
//		}
		
		
		/**
		 * Get the size of the largest neighborhood, namely how many
		 * {@linkplain InstancePlaceholder} objects it contains.
		 * 
		 * @return size of the largest neighborhood.
		 */
//		public int getMaxNeighborhoodSize() {
//			if (this.isActive) {
//				throw new RuntimeException("To get the maximum neighborhood size from AntHill, it must be shutdown first.");
//			}
//			int maxNeighborhoodSize = 0;
//			for (InstancePlaceholder ip : this.instancePlaceholders) {
//				if (ip.getNeighborhoodSize() > maxNeighborhoodSize) {
//					maxNeighborhoodSize = ip.getNeighborhoodSize();
//				}
//			}
//			return maxNeighborhoodSize;
//		}
		
		
		/**
		 * Tells how many {@linkplain InstancePlaceholder} objects were found with a local
		 * similarity value equal or below the noise threshold and therefore
		 * are recognized as noise.
		 * 
		 * @return number of InstancePlaceholder objects recognized as noise.
		 */
//		public int getFoiCountBelowOrEqualNoiseThreshold() {
//			if (this.isActive) {
//				throw new RuntimeException("To get the minimum local similarity value below or equal its noise threshold from AntHill, it must be shutdown first.");
//			}
//			if (optn_foiNoiseThreshold < 0) {
//				return -1; //When there is no threshold, it does not make sense to count.
//			}
//			int count = 0;
//			for (InstancePlaceholder ip : this.instancePlaceholders) {
//				if (ip.getFoi() <= optn_foiNoiseThreshold) {
//					count++;
//				}
//			}
//			return count;
//		}
		
		
		/**
		 * Retrieves the minimum local similarity value, that was above the
		 * noise threshold local similarity value in the recent clustering task.
		 * 
		 * @return the minimum local similarity value above the noise threshold.
		 */
//		public double getMinimumFoiValueAboveNoiseThreshold() {
//			if (this.isActive) {
//				throw new RuntimeException("To get the minimum local similarity value above its noise threshold from AntHill, it must be shutdown first.");
//			}
//			double min = -1;
//			for (InstancePlaceholder ip : this.instancePlaceholders) {
//				if ((ip.getFoi() < min || min < 0) && ip.getFoi() > optn_foiNoiseThreshold) {
//					min = ip.getFoi();
//				}
//			}
//			return min;
//		}
		
		
		/**
		 * Tells the average local similarity value of the recent clustering
		 * task.
		 * 
		 * @return the average local similarity value of the recent clustering
		 * task.
		 */
//		public double getAverageFoiValue() {
//			if (this.isActive) {
//				throw new RuntimeException("To get the average local similarity value from AntHill, it must be shutdown first.");
//			}
//			int count = 0;
//			double sum = 0;
//			for (InstancePlaceholder ip : this.instancePlaceholders) {
//				sum += ip.getFoi();
//				count++;
//			}
//			return count > 0 ? sum / count : -1;
//		}
		
		
		/**
		 * Tells the maximum local similarity value of the recent clustering
		 * task.
		 * 
		 * @return the maximum local similarity value of the recent clustering
		 * task.
		 */
//		public double getMaximumFoiValue() {
//			if (this.isActive) {
//				throw new RuntimeException("To get the maximum local similarity value from AntHill, it must be shutdown first.");
//			}
//			double max = -1;
//			for (InstancePlaceholder ip : this.instancePlaceholders) {
//				if (ip.getFoi() > max) {
//					max = ip.getFoi();
//				}
//			}
//			return max;
//		}
		
	}
	
	
	/**
	 * Builds the clusterer with the given {@link weka.core.Instances}.
	 * 
	 * @param data Instances to be clustered
	 * @throws Exception for various reasons
	 */
	@Override
	public void buildClusterer(Instances data) throws Exception {
		
		if (m_Debug) { System.out.println("# Starting DBACluster."); }
		
		if (m_Debug) { System.out.println("# " + data.size() + " instances given."); }
		
		if (m_Debug) { System.out.println("# Test capabilities."); }
		getCapabilities().testWithFail(data);
		
		if (m_Debug) { System.out.println("# Initialization of variables."); }
		this.data = new Instances(data);
		optn_distanceFunction.setInstances(this.data);
		this.antHill = new AntHill();
		this.replaceMissingValuesFilter = new ReplaceMissingValues();
		
		if (optn_replaceMissing) {
			if (m_Debug) { System.out.println("#   Running replacement for missing values."); }
			this.replaceMissingValuesFilter.setInputFormat(this.data);
			this.data = Filter.useFilter(this.data, this.replaceMissingValuesFilter);
		}
		
		if (m_Debug) { System.out.println("#   Preparing the ants."); }
		this.antHill.initialize(optn_alpha, this.data, optn_antsNum, optn_antsCallPerAntCycle, optn_s, optn_antsAssumeGlobalAfterNumCalls, optn_foiRaiseTolerance, optn_foiNoiseThreshold);
		
		if (m_Debug) { System.out.println("# Start ant clustering."); }
		if (m_Debug) { System.out.println("#   Foi raise tolerance is " + optn_foiRaiseTolerance + "."); }
		while (this.antHill.getAntCycles() < optn_antsMaxAntCycles && this.antHill.isActive()) {
			this.antHill.runAntCycle();
		}
		if (this.antHill.getAntCycles() >= optn_antsMaxAntCycles) { //While loop ended for this reason.
			this.out_atAntCycleExecutionLimit = true;
		}
		
		if (m_Debug) { System.out.println("# Ant clustering done. Shutdown ant population now and cleanup."); }
		this.antHill.shutdown();
		optn_distanceFunction.clean();
		
		if (m_Debug) { System.out.println("# Get clustering results" + (optn_maxClusterNum > 0 ? " and limit clusters to " + optn_maxClusterNum + " clusters maximum": "") + "."); }
		this.out_antCycles = this.antHill.getAntCycles();
//		if (m_Debug) {
//			this.out_maxNeighborhoodSize = this.antHill.getMaxNeighborhoodSize();
//			this.out_foiCountBelowOrEqualNoiseThreshold = this.antHill.getFoiCountBelowOrEqualNoiseThreshold();
//			this.out_minimumFoiAboveNoiseThreshold = this.antHill.getMinimumFoiValueAboveNoiseThreshold();
//			this.out_averageFoi = this.antHill.getAverageFoiValue();
//			this.out_maximumFoi = this.antHill.getMaximumFoiValue();
//		}
		this.out_clusterAssignments = this.antHill.getClusterAssignments(optn_maxClusterNum);
//		this.out_numUnclustered = this.antHill.getCountUnclusteredInLastAssignment();
		
		if (m_Debug) { System.out.println("# Perform final cleanup."); }
		this.antHill = null; //Important for Weka.
		
		if (m_Debug) { System.out.println("# DBACluster finished.\n"); }
		
		return;
		
	}
	
	
	/**
	 * Tells cluster count in the current result.
	 * 
	 * @return number of clusters in the current result.
	 */
	@Override
	public int numberOfClusters() {
		int maxClusterNum = 0;
		int size = this.out_clusterAssignments.length;
		for (int i = 0; i < size; i++) {
			if (this.out_clusterAssignments[i] > maxClusterNum) {
				maxClusterNum = this.out_clusterAssignments[i];
			}
		}
		return maxClusterNum + 1; //+ 1 as cluster counting starts with 0.
	}
	
	
	/**
	 * Cluster the given {@code instance}.
	 * 
	 * @param instance the {@linkplain Instance} to be clustered.
	 * @return cluster number of the {@code instance}.
	 */
	@Override
	public int clusterInstance(Instance instance) throws Exception {
		Instance instance2 = null;
		if (optn_replaceMissing) {
			replaceMissingValuesFilter.input(instance);
			replaceMissingValuesFilter.batchFinished();
			instance2 = replaceMissingValuesFilter.output();
		}
		else {
			instance2 = instance;
		}
		return this.clusterProcessedInstance(instance2);
	}
	
	
	/**
	 * Cluster an {@code instance} that was also used for building the clusterer.
	 * 
	 * @param instance the {@linkplain Instance} to be clustered.
	 * @return cluster number of the {@code instance}.
	 * @see #buildClusterer(Instances)
	 */
	private int clusterProcessedInstance(Instance instance) {
		int index = DBACluster.indexOfInstanceInInstances(this.data, instance);
		if (index >= 0) { //Cluster known instance.
			return out_clusterAssignments[index];
		}
		else { //Cluster new instance.
			if (m_Debug) {
				System.out.println("# ! Trying to cluster an instance that was not clustered before. It automatically gets the cluster number -1 assigned.");
			}
			return -1;
		}
	}
	
	
	/**
	 * Tells the index of an {@link weka.core.Instance} object in a
	 * {@link weka.core.Instances} object.
	 * <p>
	 * This is an alternative method for the {@code instances.indexOf(instance)} task, but
	 * does not use the {@code instance.equals(Object obj)} invoked by
	 * {@code instances.indexOf(instance)}. For the context this method is used
	 * in, the {@code instances.indexOf(instance)} did not work as desired, so
	 * this method was written. If there is a way to execute also {@code indexOf}, this
	 * method becomes obsolete.
	 * 
	 * @param instances Instances to search in
	 * @param instance the Instance to search for in {@code instances}
	 * @return index of {@code instance} in {@code instances} or -1 if {@code instance} was not found.
	 */
	private static int indexOfInstanceInInstances(Instances instances, Instance instance) {
		if (instance == null) {
			return -1;
		}
		double[] instanceAsArray = instance.toDoubleArray();
		Iterator<Instance> iterator = instances.iterator();
		Instance currentInstance = null;
		int loopCount = -1;
		while (iterator.hasNext()) {
			currentInstance = iterator.next();
			loopCount++;
			if (Arrays.equals(instanceAsArray, currentInstance.toDoubleArray())) {
				return loopCount;
			}
		}
		return -1; //E.g. currentInstance is still null.
	}
	
	
	/**
	 * Returns a string representation of this object.<br/>
	 * This is also the output for the WEKA Explorer GUI.
	 * 
	 * @return a string representing this object.
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		StringBuffer temp = new StringBuffer();
		temp.append(out_antCycles + " ant cycles were executed" + (out_atAntCycleExecutionLimit ? " (reached antsMaxAntCycles)" : "") + ".\n");
		if (out_maxNeighborhoodSize >= 0) {
			temp.append(out_maxNeighborhoodSize + " was the size of the largest neighborhood found.\n");
		}
		if (out_numUnclustered > 0) {
			temp.append("\nUnclustered instances: " + out_numUnclustered + ".\n");
		}
//		if (out_minimumFoiAboveNoiseThreshold >= 0 || out_averageFoi >= 0 || out_maximumFoi >= 0) {
//			temp.append("Local similarity statistics:\n");
//			if (out_minimumFoiAboveNoiseThreshold >= 0) {
//				temp.append("  Minimum" + (optn_foiNoiseThreshold > -1 ? " (above noise threshold): " : ": ") + out_minimumFoiAboveNoiseThreshold + "\n");
//			}
//			if (out_averageFoi >= 0) {
//				temp.append("  Average: " + out_averageFoi + "\n");
//			}
//			if (out_maximumFoi >= 0) {
//				temp.append("  Maximum: " + out_maximumFoi + "\n");
//			}
//			if (out_maximumFoi >= 0) {
//				temp.append("  Number of instances regarded as noise: " + (out_foiCountBelowOrEqualNoiseThreshold < 0 ? 0 : out_foiCountBelowOrEqualNoiseThreshold) + "\n");
//			}
//			temp.append("\n");
//		}
		return temp.toString();
	}
	
	
	/**
	 * Main method for executing this class.
	 * 
	 * @param args use -h to list all parameters
	 */
	public static void main(String[] args) {
		runClusterer(new DBACluster(), args);
	}
	
}


/*
 * Bibliography (most frequently cited here, to see all literature used for this file, please refer to the thesis):
 * 
 * Deneubourg et al. 1991:
 * Deneubourg/Goss/Franks/Sendova-Franks/Detrain/Chretien 1991:
 *  Deneubourg, Jean Louis; Goss, Simon; Franks, Nigel R.; Sendova-Franks, Ana B.; Detrain, Claire; Chretien, Ludovic:
 *  The Dynamics of Collective Sorting - Robot-Like Ants and Ant-Like Robots.
 *  In: Meyer, Jean-Arcady; Wilson, Stewart W. (Eds.):
 *  From Animals to Animats - Proceedings of the First International Conference on Simulation of Adaptive Behavior.
 *  Pages 356-365.
 *  MIT Press, Cambridge (Massachusetts), 1991.
 * 
 * Lumer/Faieta 1994:
 * 	Lumer, Erik D.; Faieta, Baldo:
 * 	Diversity and Adaptation in Populations of Clustering Ants.
 * 	In: Cliff, David; Husbands, Phil; Meyer, Jean-Arcady; Wilson, Stewart W. (Eds.):
 * 	From Animals to Animats 3 - Proceedings of the Third International Conference on Simulation of Adaptive Behavior.
 *  Pages 501-508.
 * 	Complex adaptive systems.
 * 	MIT Press, Cambridge (Massachusetts), 1994.
 * 
 * WEKA:
 * Hall et al. 2009:
 * Hall/Eibe/Holmes/Pfahringer/Reutemann/Witten 2009:
 *  Hall, Mark; Frank, Eibe; Holmes, Geoffrey; Pfahringer, Bernhard; Reutemann, Peter; Witten, Ian H.:
 *  The WEKA Data Mining Software - An Update
 *  In: ACM SIGKDD Explorations Newsletter, vol. 11, no. 1, 2009.
 *  Pages 10-18.
 *  DOI: 10.1145/1656274.1656278.
 * 
 * Zhe et al. 2011:
 * 	Zhe, Gong; Dan, Li; Baoyu, An; Yangxi, Ou; Wei, Cui; Xinxin, Niu; Yang, Xin:
 * 	An Analysis of Ant Colony Clustering Methods - Models, Algorithms and Applications.
 * 	In: International Journal of Advancements in Computing Technology (IJACT) 11 (2011) 3, 
 * 	p. 112 - 121.
 * 
 */
