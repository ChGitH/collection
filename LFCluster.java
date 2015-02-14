/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program. If not, see <http://www.gnu.org/licenses/>.
 */

package weka.clusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.Random;
import java.util.Vector;

import weka.clusterers.AbstractClusterer;
import weka.clusterers.ClusterEvaluation;
import weka.clusterers.Clusterer;
import weka.clusterers.EM;
import weka.clusterers.RandomizableClusterer;
import weka.clusterers.ant.pile.grid.AntGridClusterer;
import weka.clusterers.ant.pile.grid.InstancesOnAntGrid;
import weka.core.Attribute;
import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.OptionHandler;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.TechnicalInformation;
import weka.core.Utils;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;
import weka.filters.Filter;
import weka.filters.unsupervised.attribute.ReplaceMissingValues;

/**
 * <!-- globalinfo-start -->
 * Clusterer using virtual ants to cluster Instances data on a two dimensional grid. For more information see:
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
 * <pre> -kp &lt;num&gt;
 *  Pick up threshold constant. The pick up threshold constant is used for
 *  adjusting the pick up probability.</pre>
 * 
 * <pre> -kd &lt;num&gt;
 *  Drop down threshold constant. The drop down threshold constant is used for
 *  adjusting the drop down probability.</pre>
 * 
 * <pre> -kdf &lt;type&gt;
 *  Drop down function that is used. Can use the original drop down function by
 *  Lumer&#47;Faieta 1994 or the symmetric function by Deneubourg et al. 1991.
 *  (default = original)</pre>
 * 
 * <pre> -dist &lt;classname and options&gt;
 *  Distance function that is used for instance comparison according to the
 *  attributes of the instances in feature space.
 *  (default = weka.core.EuclideanDistance (without normalization))</pre>
 * 
 * <pre> -gx &lt;num&gt;
 *  Size of the 2-dimensional grid on the x axis.</pre>
 * 
 * <pre> -gy &lt;num&gt;
 *  Size of the 2-dimensional grid on the y axis.</pre>
 * 
 * <pre> -i &lt;num&gt;
 *  Number of ant cycles to be executed. When all ant cycles passed, the
 *  algorithm terminates.</pre>
 * 
 * <pre> -ic &lt;num&gt;
 *  Number of ant calls per ant cycle. Each ant that is called can work and
 *  walk.
 *  (default = 10000)</pre>
 * 
 * <pre> -an &lt;num&gt;
 *  Number of ants on the grid.</pre>
 * 
 * <pre> -avr &lt;num&gt;
 *  Size of the neighborhood or view range of each ant. The radius defines which
 *  area around an ant is regarded as the ant's neighborhood.</pre>
 * 
 * <pre> -as &lt;num&gt;
 *  Speed limit for all ants. The algorithm tries to find for every speed from
 *  1 to the defined maximum ant groups of nearly the same size and assigns them
 *  their individual speeds.</pre>
 * 
 * <pre> -adr &lt;num&gt;
 *  Area, in which ants can drop an instance. It is measured in grid cells
 *  around the ant.</pre>
 * 
 * <pre> -adm &lt;num&gt;
 *  Number of recent drop locations an ant can remember. When the ant remembers
 *  drop locations it goes to remembered locations that are most similar to the
 *  instance the ant is carrying.</pre>
 * 
 * <pre> -abdc &lt;num&gt;
 *  After how many ant cycles carrying nothing an ant switches to destructive
 *  behavior. When an ant did not pick up an instance for the given amount of
 *  ant cycles it becomes destructive and picks up the next instance it can find
 *  regardless of the neighborhood of the instance.</pre>
 * 
 * <pre> -abdn &lt;num&gt;
 *  When an ant picked up enough instances in destructive behavior it turns back
 *  to normal behavior again. This value defines how many pick ups are made in
 *  destructive behavior.</pre>
 * 
 * <pre> -m
 *  Replace missing values.</pre>
 * 
 * <pre> -w
 *  This clusterer is used to make the clusters formed by the ants clear. It is
 *  applied in the end, when no more ant cycles must be executed.</pre>
 * 
 * <!-- options-end -->
 * 
 * @version 0.9
 * @author Christoph
 * @see RandomizableClusterer
 */
public class LFCluster extends RandomizableClusterer implements TechnicalInformationHandler {
	
	/** For serialization */
	private static final long serialVersionUID = -3955155847797759076L;
	
	/** Alpha interval upper bound. */
	static final double alphaMaxValue = 99999.0;
	
	/** Alpha interval lower bound. */
	static final double alphaMinValue = 0.0;
	
	/** How much free space in percent points must be left on grid minimum, when all instances are on the grid. */
	static final double gridMinFreeSpace = 20;
	
	/** How large a grid must be at least in each dimension. */
	static final int gridMinSize = 3;
	
	/** How many attempts an ant is allowed to make to drop its GridInstance regularly until it tries to get rid of it at shutdown. */
	static final int antShutdownRegularAttempts = 10;
	
	/** Tag list. */
	static final int tag_LumerFaieta = 0;
	static final String tag_LumerFaietaLabel = "original";
	static final int tag_DeneubourgEtAl = 1;
	static final String tag_DeneubourgEtAlLabel = "symmetric";
	static final Tag[] tags = { new Tag(tag_LumerFaieta, tag_LumerFaietaLabel), new Tag(tag_DeneubourgEtAl, tag_DeneubourgEtAlLabel) };
	
	/** Notify the user when a certain number of ant cycles passed. If the algorithm takes long time to run it notifies the user that the program is still active. */
	static final int debug_verboseEveryAntCyclesPassed = 100;
	
	/** Colony similarity coefficient alpha. The larger, the more similar the colonies must be. */
	protected double optn_alpha = 5.0; //-a //0.7 //5.0
	
	/** Pick up threshold constant. Used for adjusting the pick up probability. Must be greater than 0. */
	protected double optn_kp = 0.02; //-kp //0.12 //0.02
	
	/** Drop down threshold constant. Used for adjusting the drop down probability. */
	protected double optn_kd = 0.5; //-kd //0.1 //0.5
	
	/** Which kd function should be used. */
	protected int optn_kdFunction = tag_LumerFaieta; //-kdf
	
	/** The distance function used for determining the distance between instances. */
	protected DistanceFunction optn_distanceFunction = new EuclideanDistance(); //-dist
	
	/** Width of the grid. */
	protected int optn_gridSizeX = 52; //-gx
	
	/** Height of the grid. */
	protected int optn_gridSizeY = 52; //-gy
	
	/** Maximum number of ant cycles to be executed. */
	protected int optn_antCycles = 50; //-i
	
	/**
	 * How many ants should be called in one ant cycle.
	 * <p>
	 * For each call an ant is selected randomly. An ant cycle is done when
	 * there are as many ants called as specified here. As the selection of an
	 * ant is random, an ant can also be called several times in the same ant
	 * cycle, whereas other ants may not be called during one ant cycle.
	 */
	protected int optn_antsCallPerAntCycle = 10000; //-ic
	
	/** Number of ants on the grid. */
	protected int optn_antsNum = 40; //-an
	
	/**
	 * Size of the view range of ant ant or neighborhood radius s on the grid.
	 * <p>
	 * All instances within the radius are regarded as neighbors. The ant view
	 * range is measured in grid cells surrounding one ant. It has the shape of
	 * a square. When the ant view range is 0, the ant view range is limited to
	 * the grid cell the ant sits on. As this does not lead to clustering the
	 * ant view range must be at least 1.
	 * */
	protected int optn_antsViewRange = 1; //-avr
	
	/** 
	 * Maximum ant speed, how many steps an ant can make on the grid in
	 * one turn.
	 * <p>
	 * A step means an ant is going from one grid cell to a neighbor grid cell,
	 * diagonal cells not included. When this value is greater than 1, ants can
	 * have different speeds. For example, if the speed distribution limit is
	 * 10, there are 10 groups of ants with the speeds 1,2,..,10. The clusterer
	 * tries to keep the groups of equal size, although this is not possible
	 * when the division of ant count / speed distribution limit has a rest. In
	 * this case the ant speed groups do not all have the same size.
	 */
	protected int optn_antsSpeedDistributionLimit = 1; //-as
	
	/**
	 * The ant drop range is the area in which ants can drop instances they
	 * carry.
	 * <p>
	 * It is measured in grid cells surrounding an ant, so that the area has the
	 * shape of a square. When the ant drop range is 0, the ant can drop the
	 * instance it carries only at its position and if there is no instance yet.
	 * In other cases the ant drops the instance to a random and free grid cell
	 * in this as soon as the ant decided to drop its carried instance.
	 */
	protected int optn_antsDropRange = 1; //-adr

	/** Number of drop locations an ant can remember.
	 * <p>
	 * The ant will remember its recent drop locations and when it picks up a
	 * new instance it checks its memory and retrieves the instance that is most
	 * similar to the instance picked up just now. Then the ant goes in the
	 * direction to where it dropped the memorized instance. As soon the ant
	 * reaches the memorized location it starts walking randomly again, no
	 * matter if it dropped the instance there or not. When the ant picks up
	 * another instance the process starts again. Ants do not have a drop
	 * location memory when this option is set to -1.
	 */
	protected int optn_antsDropMemorySize = -1; //-adm
	
	/**
	 * After how many ant cycles carrying nothing the ant switches to
	 * destructive behavior.
	 * <p>
	 * When an ant did not pick up an instance for the given amount of ant
	 * cycles it becomes destructive and picks up the next instance regardless
	 * of the neighborhood of the instance. Set to -1 to deactivate this
	 * behavior.
	 */
	protected int optn_antsBehaviorDestructiveAfterNumFreeCycles = -1; //-abdc, Lumer/Faieta 1994, p. 505-507.
	
	/**
	 * How many times an ant will pick up an instance in destructive behavior
	 * before the ant switches back to normal pick up behavior.
	 * <p>
	 * Destructive pick up behavior means the ant picks up instances regardless
	 * of the instance environments, therefore the ant does not care if it
	 * destroys clusters of already grouped instances. After an ant picked up
	 * enough instances in that behavior it turns back to normal pick up
	 * behavior again. This value determines how many instances the ant wants to
	 * pick up without caring for the instance environment, before it switches
	 * back to normal behavior. When this value is -1, the ant remains
	 * destructive.
	 */
	protected int optn_antsBehaviorDestructiveForNextPickUps = 3; //-abdn
	
	/** Replace missing values globally? */
	protected boolean optn_replaceMissing = false; //-m
	
	/** Clusterer that is used to explain the clusters (instance groups) on the grid. */
	protected Clusterer optn_gridClusterer = new AntGridClusterer(); //-w
	
	/** The current alpha value. */
	protected double alpha = 0.0;
	
	/**
	 * Instances to be clustered.
	 * <p>
	 * This variable holds the given Instances object, which contains all
	 * Instance objects to be clustered. This variable is best understood as a
	 * global variable and once the Instances are assigned, it must not be
	 * altered anymore. Other classes of this clusterer rely on the contents of
	 * this variable and expect that this variable is available.
	 * 
	 * @see weka.core.Instances
	 */
	protected Instances data = null;
	
	/** Grid on which the instances are distributed, clustered and where the ants run. */
	protected Grid grid; //Construct with target size.
	
	/** Ants to be used. */
	protected Ant[] ants;
	
	/** Number of ant cycles that were already executed. */
	protected int antCycles = 0;
	
	/** Replace missing values filter. */
	protected ReplaceMissingValues replaceMissingValuesFilter;
	
	/** Random number generator. */
	protected Random rand = null; //It is required globally, and can not be instantiated every time it is used, because the generator starts with the same seed then again -> always same numbers are generated.
	
	/** Assignments of the most recently clustered gridInstances to clusters. */
	protected double[] out_clusterAssignments = null;
	
	/** String output of the used gridClusterer. */
	protected String out_gridClustererResults = null;
	
	/** The default constructor. */
	public LFCluster() {
		super();
	}
	
	
	/**
	 * Returns a description of this clusterer.
	 * 
	 * @return a brief text describing this clusterer.
	 */
	public String globalInfo() {
		return "Clusterer using artificial ants (agents) to cluster instances on a two dimensional grid. "
				+ "The groupings on the grid are interpreted as clusters. Another cluster algorithm is needed "
				+ "to explain the groups on the grid. When using a special AntGridClusterer for that task the cluster algorithm is called with "
				+ "a specialized object of the Instances object. For more information about the ant algorithm see:\n\n" + getTechnicalInformation().toString();
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
		ti.setValue(Field.PAGES, "501-508");
		ti.setValue(Field.ISBN13, "9780262531221");
		return ti;
	}
	
	
	/**
	 * Tip text provider for the alpha value setting.
	 * 
	 * @return Text that briefly describes the functionality of the alpha value
	 *         setting. 
	 */
	public String alphaTipText() {
		return "Colony similarity coefficient Alpha for similarity of clusters. The greater the value, the higher the similarity must be.";
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
	 * Tip text provider for the pick up threshold setting.
	 * 
	 * @return Text that briefly describes the functionality of the
	 *         pick up threshold.
	 */
	public String pickUpThresholdConstantTipText() {
		return "The pick up threshold constant adjusts the pick up probability for instances.";
	}
	
	
	/**
	 * Sets the pick up threshold constant.
	 * 
	 * @param value the pick up threshold to be used.
	 * @throws IllegalArgumentException if {@code value} is equal or
	 *         less than 0.
	 */
	public void setPickUpThresholdConstant(double value) throws IllegalArgumentException {
		if (value <= 0) { //The ant uses optn_kp later to calculate if it wants to pick up a GridInstance. If optn_kp is 0, division with 0 can occur in case foi is also 0.
			if (m_Debug) {
				throw new IllegalArgumentException("The pick up threshold constant must be greater than 0.");
			}
			else {
				value = 0.01;
			}
		}
		this.optn_kp = value;
	}
	
	
	/**
	 * Tells the currently set pick up threshold.
	 * 
	 * @return the current pick up threshold.
	 */
	public double getPickUpThresholdConstant() {
		return this.optn_kp;
	}
	
	
	/**
	 * Tip text provider for the drop down threshold setting.
	 * 
	 * @return Text that briefly describes the functionality of the
	 *         drop down threshold.
	 */
	public String dropDownThresholdConstantTipText() {
		return "The drop down threshold constant adjusts the drop down probability for instances.";
	}
	
	
	/**
	 * Sets the drop down threshold constant.
	 * 
	 * @param value the drop down threshold to be used.
	 * @throws IllegalArgumentException if {@code value} is equal or
	 *         less than 0.
	 */
	public void setDropDownThresholdConstant(double value) throws IllegalArgumentException {
		if (value <= 0) {
			if (m_Debug) {
				throw new IllegalArgumentException("The drop down threshold constant must be greater than 0.");
			}
			else {
				value = 0.01;
			}
		}
		this.optn_kd = value;
	}
	
	
	/**
	 * Tells the currently set drop down threshold.
	 * 
	 * @return the current drop down threshold.
	 */
	public double getDropDownThresholdConstant() {
		return this.optn_kd;
	}
	
	
	/**
	 * Tip text provider for the drop down function setting.
	 * 
	 * @return Text that briefly describes the functionality of the
	 *         drop down threshold.
	 */
	public String dropDownFunctionTipText() {
		return "Use the original drop down function proposed by Lumer/Faieta (1994) or the symmetric one by Deneubourg/Goss/Franks/Sendova-Franks/Detrain/Chretien (1991).";
	}
	
	
	/**
	 * Sets the drop down function. Two functions are accepted: The original
	 * function is the one used by Lumer/Faieta 1994 and described in
	 * {@link #getTechnicalInformation()}. The other one is the symmetric
	 * counterpart to the pick up function and suggested by Deneubourg/Goss/
	 * Franks/Sendova-Franks/Detrain/Chretien 1991.
	 * 
	 * @param value a {@link SelectedTag} describing the drop down function to
	 *        use.
	 * @throws IllegalArgumentException if in debug mode and {@code value} is not a
	 *         known {@link SelectedTag}.
	 */
	public void setDropDownFunction(SelectedTag value) throws IllegalArgumentException {
		if (value.getTags() == tags) {
			this.optn_kdFunction = value.getSelectedTag().getID();
		}
		else {
			if (m_Debug) {
				throw new IllegalArgumentException("The drop function can only be set to known tags.");
			}
			else {
				this.optn_kdFunction = tag_LumerFaieta;
			}
		}
	}
	
	
	/**
	 * Tells the currently set drop down function.
	 * 
	 * @return the {@link SelectedTag} naming the current drop down function.
	 */
	public SelectedTag getDropDownFunction() {
		return new SelectedTag(this.optn_kdFunction, tags);
	}
	
	
	/**
	 * Tip text provider for the distance function setting.
	 * 
	 * @return Text that briefly describes the functionality of the
	 *         distance function.
	 */
	public String distanceFunctionTipText() {
		return "Distance function to use for instance comparison, it is recommended to turn normalization off if possible.";
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
	 * Tip text provider for the grid size on x axis setting.
	 * 
	 * @return Text that briefly describes the grid size on x axis setting.
	 */
	public String gridSizeXTipText() {
		return "Size of the 2-dimensional grid on x axis measured in grid cells.";
	}
	
	
	/**
	 * Sets the size of the grid on the x axis.
	 * 
	 * @param value size of the grid on the x axis measured in grid cells.
	 * @throws IllegalArgumentException if {@code value} is smaller than {@value #gridMinSize},
	 *         only in debug mode.
	 */
	public void setGridSizeX(int value) throws IllegalArgumentException {
		if (value < gridMinSize) {
			if (m_Debug) {
				throw new IllegalArgumentException("Only positive integer numbers equal than " + gridMinSize + " or greater are allowed for grid size.");
			}
			else {
				value = gridMinSize;
			}
		}
		this.optn_gridSizeX = value;
	}
	
	
	/**
	 * Tells the currently set size of the grid on the x axis.
	 * 
	 * @return size of the grid on the x axis measured in grid cells.
	 */
	public int getGridSizeX() {
		return this.optn_gridSizeX;
	}
	
	
	/**
	 * Tip text provider for the grid size on y axis setting.
	 * 
	 * @return Text that briefly describes the grid size on y axis setting.
	 */
	public String gridSizeYTipText() {
		return "Size of the 2-dimensional grid on y axis measured in grid cells.";
	}
	
	
	/**
	 * Sets the size of the grid on the y axis.
	 * 
	 * @param value size of the grid on the y axis measured in grid cells.
	 * @throws IllegalArgumentException if {@code value} is smaller than {@value #gridMinSize},
	 *         only in debug mode.
	 */
	public void setGridSizeY(int value) throws IllegalArgumentException {
		if (value < gridMinSize) {
			if (m_Debug) {
				throw new IllegalArgumentException("Only positive integer numbers equal than " + gridMinSize + " or greater are allowed for grid size.");
			}
			else {
				value = gridMinSize;
			}
		}
		this.optn_gridSizeY = value;
	}
	
	
	/**
	 * Tells the currently set size of the grid on the y axis.
	 * 
	 * @return size of the grid on the y axis measured in grid cells.
	 */
	public int getGridSizeY() {
		return this.optn_gridSizeY;
	}
	
	
	/**
	 * Tip text provider for the number of ant cycles to execute setting.
	 * 
	 * @return Text that briefly describes the ant cycles setting.
	 */
	public String antCyclesTipText() {
		return "Maximum number ant cycles (iterations).";
	}
	
	
	/**
	 * Sets the maximum number of ant cycles to be executed.
	 * 
	 * @param value the maximum number of iterations
	 * @throws IllegalArgumentException if {@code value} is smaller than 1, 
	 *         only in debug mode.
	 */
	public void setAntCycles(int value) throws IllegalArgumentException {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("Maximum number of ant cycles must be positive integer > 0");
			}
			else {
				value = 1;
			}
		}
		this.optn_antCycles = value;
	}
	
	
	/**
	 * Tells the currently set number of ant cycles to execute.
	 * 
	 * @return number of ant cycles to execute.
	 */
	public int getAntCycles() {
		return this.optn_antCycles;
	}
	
	
	/**
	 * Tip text provider for the ant calls per ant cycle setting.
	 * 
	 * @return Text that briefly describes the ant calls per ant cycle setting.
	 */
	public String antsPerAntCycleTipText() {
		return "How many ants are called per ant cycle.";
	}
	
	
	/**
	 * Sets the amount of ants to be called per ant cycle.
	 * 
	 * @param value Ant calls per ant cycle.
	 * @throws IllegalArgumentException if {@code value} is smaller than 1 and
	 *         debug mode is activated.
	 */
	public void setAntsPerAntCycle(int value) throws IllegalArgumentException {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The number of ants to call per ant cycle must be a positive integer value.");
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
	public int getAntsPerAntCycle() {
		return this.optn_antsCallPerAntCycle;
	}
	
	
	/**
	 * Tip text provider for the number of ants setting.
	 * 
	 * @return Text that briefly describes the number of ants setting.
	 */
	public String antsNumTipText() {
		return "Number of ants that run on the grid.";
	}
	
	
	/**
	 * Sets the number of available ants on the grid.
	 * 
	 * @param value how many ants are on the grid.
	 * @throws IllegalArgumentException if {@code value} is smaller than 1 and in
	 *         debug mode.
	 */
	public void setAntsNum(int value) throws IllegalArgumentException {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The number of ants must be a positive integer value. Also at least one ant must be on the grid, otherwise nothing can be clustered.");
			}
			else {
				value = 1;
			}
		}
		this.optn_antsNum = value;
	}
	
	
	/**
	 * Tells how many ants must be on the grid.
	 * 
	 * @return ant count on the grid.
	 */
	public int getAntsNum() {
		return this.optn_antsNum;
	}
	
	
	/**
	 * Tip text provider for the ants view range setting.
	 * 
	 * @return Text that briefly describes the ants view range setting.
	 */
	public String antsViewRangeTipText() {
		return "Size of the view range of each ant measured in number of grid cells around an ant."; 
	}
	
	
	/**
	 * Sets the ants view range.
	 * 
	 * @param value grid cells surrounding an ant
	 * @throws IllegalArgumentException if {@code value} is smaller than 1 and debug
	 *         mode is activated.
	 */
	public void setAntsViewRange(int value) throws IllegalArgumentException {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("Clustering can not be performed when the ant view range is below 1.");
			}
			else {
				value = 1;
			}
		}
		this.optn_antsViewRange = value;
	}
	
	
	/**
	 * Tells the size of the view range of the ants.
	 * @return size of the ants view range measured in grid cells.
	 */
	public int getAntsViewRange() {
		return this.optn_antsViewRange;
	}
	
	
	/**
	 * Tip text provider for the ants speed distribution limit setting.
	 * 
	 * @return Text that briefly describes the ants speed distribution limit
	 *         setting.
	 */
	public String antsSpeedDistributionLimitTipText() {
		return "The maximum speed an ant can have.";
	}
	
	
	/**
	 * Sets the maximum speed of the ants.
	 * 
	 * @param value the maximum speed an ant can have.
	 * @throws IllegalArgumentException if {@code value} is smaller than one and debug
	 *         mode is activated.
	 * @throws IllegalArgumentException if there are not enough ants available
	 *         to increment the speed from 1 to {@code value} with step size 1
	 *         and assign this speed to at least one ant. This is the case when
	 *         {@code value} &gt; {@link #getAntsNum()}.
	 */
	public void setAntsSpeedDistributionLimit(int value) throws IllegalArgumentException {
		if (value < 1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The minimum speed for an ant is 1.");
			}
			else {
				value = 1;
			}
		}
		if (this.getAntsNum() < this.getAntsSpeedDistributionLimit()) {
			throw new IllegalArgumentException("The ant population is smaller than the ant speed distribution limit. Can not assign the speeds to the ants in the population, so that for every speed value from 1 up to antsSpeedDistributionLimit there exists an ant with this speed.");
		}
		this.optn_antsSpeedDistributionLimit = value;
	}
	
	
	/**
	 * Tells the maximum speed an ant can have.
	 * 
	 * @return maximum speed of an ant.
	 */
	public int getAntsSpeedDistributionLimit() {
		return this.optn_antsSpeedDistributionLimit;
	}
	
	
	/**
	 * Tip text provider for the ants drop range setting.
	 * 
	 * @return Text that briefly describes the ants speed distribution limit
	 *         setting.
	 */
	public String antsDropRangeTipText() {
		return "Size of the area in which ants are allowed to drop instances measured in number of grid cells around an ant.";
	}
	
	
	/**
	 * Sets the drop range size of the ants.
	 * 
	 * @param value size of the ants drop range.
	 * @throws IllegalArgumentException if {@code value} is smaller than 0 and
	 * debug mode is activated.
	 */
	public void setAntsDropRange(int value) throws IllegalArgumentException {
		if (value < -1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The drop range of an ant must be a positive integer value or 0 or -1.");
			}
			else {
				value = -1;
			}
		}
		this.optn_antsDropRange = value;
	}
	
	
	/**
	 * Tells the currently set size of the ants drop range.
	 * 
	 * @return size of the ants drop range.
	 */
	public int getAntsDropRange() {
		return this.optn_antsDropRange;
	}
	
	
	/**
	 * Tip text provider for the ants drop memory size setting.
	 * 
	 * @return Text that briefly describes the ants drop memory size setting.
	 */
	public String antsDropMemorySizeTipText() {
		return "How many recent drop locations an ant can remember.";
	}
	
	
	/**
	 * Sets the size of the drop location memory (how many recent drop
	 * locations ants can remember). Set to -1 to deactivate the drop location
	 * memory for the ants.
	 * 
	 * @param value size of the drop location memory
	 * @throws IllegalArgumentException if {@code value} is another negative
	 *         value than -1 and debug mode is activated.
	 */
	public void setAntsDropMemorySize(int value) throws IllegalArgumentException {
		if (value < -1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The memory size of an ant must be a positive integer value or 0 or -1.");
			}
			else {
				value = -1;
			}
		}
		if (value == 0) {
			value = -1;
		}
		this.optn_antsDropMemorySize = value;
	}
	
	
	/**
	 * Tells the size of the ant drop memory.
	 * 
	 * @return size of the ant drop memory.
	 */
	public int getAntsDropMemorySize() {
		return this.optn_antsDropMemorySize;
	}
	
	
	/**
	 * Tip text provider for the ants turn into destructive behavior setting.
	 * 
	 * @return Text that briefly describes the ants turn into destructive
	 *         behavior setting.
	 */
	public String antsBehaviorDestructiveAfterNumFreeCyclesTipText() {
		return "When an ant does not carry an instance for the given number of ant cycles it turns to destructive behavior, set to -1 to make ants never turn to destructive behavior.";
	}
	
	
	/**
	 * Set after how many ant cycles carrying nothing ants turn to destructive
	 * behavior.
	 * 
	 * @param value after how many ant cycles carrying nothing the ant turns to
	 *              destructive behavior.
	 * @throws IllegalArgumentException if {@code value} is smaller than 1, not
	 *         -1 and debug mode is activated.
	 */
	public void setAntsBehaviorDestructiveAfterNumFreeCycles(int value) throws IllegalArgumentException {
		if (value <= 0 && value != -1) {
			if (m_Debug) {
				throw new IllegalArgumentException("The ant can become destructive only after a positive number of free ant cycles or set this value to -1.");
			}
			else {
				value = -1;
			}
		}
		this.optn_antsBehaviorDestructiveAfterNumFreeCycles = value;
	}
	
	
	/**
	 * Tells after how many ant cycles carrying nothing ants behave destructive.
	 * @return after how many ant cycles carrying nothing ants behave
	 *         destructive.
	 */
	public int getAntsBehaviorDestructiveAfterNumFreeCycles() {
		return this.optn_antsBehaviorDestructiveAfterNumFreeCycles;
	}
	
	
	/**
	 * Tip text provider for the ants remain in destructive behavior setting.
	 * 
	 * @return Text that briefly describes the ants remain in destructive
	 *         behavior setting.
	 */
	public String antsBehaviorDestructiveForNextPickUpsTipText() {
		return "The ant does not take care about the environment of the instance when picking it up for this number of pick ups before it switches back to normal behavior."; 
	}
	
	
	/**
	 * Sets how often ants pick up instances without caring about the instance
	 * environments before they behave normal again.
	 * 
	 * @param value pick up count for not caring about the instance environments
	 *              before turning back to normal behavior again.
	 * @throws IllegalArgumentException if {@code value} is smaller than 1, not
	 *         -1 and debug mode is activated.
	 */
	public void setAntsBehaviorDestructiveForNextPickUps(int value) throws IllegalArgumentException {
		if (value < -1 || value == 0) {
			if (m_Debug) {
				throw new IllegalArgumentException("The value for ants remain destructive for the next pick ups must be a positive integer value or -1.");
			}
			else {
				value = -1;
			}
		}
		this.optn_antsBehaviorDestructiveForNextPickUps = value;
	}
	
	
	/**
	 * Tells how often ants pick up instances without caring for the instance
	 * environments.
	 * 
	 * @return how often ants pick up instances before they become normal.
	 */
	public int getAntsBehaviorDestructiveForNextPickUps() {
		return this.optn_antsBehaviorDestructiveForNextPickUps;
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
	 * Tip text provider for the grid clusterer setting.
	 * 
	 * @return Text that briefly describes the grid clusterer setting.
	 */
	public String gridClustererTipText() {
		return "Clusterer to explain the clusters on the grid.";
	}
	
	
	/**
	 * Sets the grid clusterer.
	 * 
	 * @param value an instance of {@link Clusterer} to cluster instances on the grid.
	 */
	public void setGridClusterer(Clusterer value) {
		this.optn_gridClusterer = value;
	}
	
	
	/**
	 * Returns the grid clusterer instance that is used for explaining the
	 * instance groups on the grid.
	 * 
	 * @return the grid clusterer as an instance of {@link Clusterer}.
	 */
	public Clusterer getGridClusterer() {
		return this.optn_gridClusterer;
	}
	
	
	/**
	 * Provides information about the available options for this clusterer.
	 * 
	 * @return an {@link Enumeration} holding the descriptions of available {@link Option}s.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();
		result.addElement(new Option("\tAlpha, colony similarity coefficient.\n\tParameter for cluster similarity. The more differences between instances should be emphasized the smaller is the value alpha. Small alpha values emphasize differences between instances a lot, a high alpha value clusters almost everything.", "a", 1, "-a <num>"));
		result.addElement(new Option("\tPick up threshold constant kp.\n\tThe pick up threshold constant is used for adjusting the pick up probability.", "kp", 1, "-kp <num>"));
		result.addElement(new Option("\tDrop down threshold constant kd.\n\tThe drop down threshold constant is used for adjusting the drop down probability.", "kd", 1, "-kd <num>"));
		result.addElement(new Option("\tDrop down function for kd.\n\tThis sets the drop down function that is used. Set to original to use the original drop down function used by Lumer/Faieta 1994 (default). Set to symmetric to use a function that is symmetric to the pick up function and was uses by Deneubourg/Goss/Franks/Sendova-Franks/Detrain/Chretien 1991.", "kdf", 1, "-kdf <type>"));
		result.addElement(new Option("\tDistance function to use for instance comparison.\n\tThis distance function is used to determine the distance between two instances according to their attributes. It is the distance function for the feature space.\n\t(default = weka.core.EuclideanDistance (without normalization))", "dist", 1, "-dist <classname and options>"));
		result.addElement(new Option("\tSize of the 2-dimensional grid on the x axis.\n\tSpecify here using a positive integer > 0 the size of the grid on the x axis. Make sure, that all instances fit on the grid.", "gx", 1, "-gx <num>"));
		result.addElement(new Option("\tSize of the 2-dimensional grid on the y axis.\n\tSpecify here using a positive integer > 0 the size of the grid on the y axis. Make sure, that all instances fit on the grid.", "gy", 1, "-gy <num>"));
		result.addElement(new Option("\tNumber of ant cycles to be executed.\n\tEach ant cycle is composed of a fixed number of ant calls. When all ant cycles passed, the algorithm terminates.", "i", 1, "-i <num>"));
		result.addElement(new Option("\tNumber of ant calls per ant cycle.\n\tEach ant that is called can work and walk.", "ic", 1, "-ic <num>"));		
		result.addElement(new Option("\tNumber of ants on the grid.\n\tHow many ants are walking on the grid.", "an", 1, "-an <num>"));
		result.addElement(new Option("\tSize of the neighborhood or view range of each ant.\n\tThe radius defines which area around an ant is regarded as the ant's neighborhood. The greater the value the bigger is the neighborhood. Its unit is number of grid cells.", "avr", 1, "-avr <num>"));
		result.addElement(new Option("\tSpeed limit for all ants.\n\tThis is the maximum speed an ant can have. The speed is measured in grid steps and must be a positive integer >= 1. Ants perform as many steps during one ant cycle as their individual speed value defines. The algorithm tries to find for every speed from 1 to the defined maximum ant groups of nearly the same size and assigns them their individual speeds.\n\t(default = 1)", "as", 1, "-as <num>"));
		result.addElement(new Option("\tArea, in which ants can drop an instance.\n\tThe area around an ant where it can drop the carried instance. It is measured in grid cells around the ant. If it is 0, the ant can only drop the carried instance at its current position.", "adr", 1, "-adr <num>"));
		result.addElement(new Option("\tNumber of recent drop locations an ant can remember.\n\tThe ant will remember its recent drop locations and go in direction to the remembered location that is most similar to the instance the ant is carrying.\n\t(default = true)", "adm", 1, "-adm <num>"));
		result.addElement(new Option("\tAfter how many ant cycles carrying nothing an ant switches to destructive behavior.\n\tWhen an ant did not pick up an instance for the given amount of ant cycles it becomes destructive and picks up the next instance it can find regardless of the neighborhood of the instance. Set to -1 to let ants never behave destructive.", "abdc", 1, "-abdc <num>"));
		result.addElement(new Option("\tHow many times an ant will pick up an instance regardless of its environment before it switches back to normal behavior.\n\tThe ant will pick up as many as specified instances immediately and regardless of the instance environment once the ant turned to destructive behavoir. When an ant picked up enough instances in destructive behavior it turns back to normal behavior again. Set to -1 to let ants remain destructive once they changed their behavior.", "abdn", 1, "-abdn <num>"));
		result.addElement(new Option("\tReplace missing values.\n\tReplace missing values globally.\n\t(default = true)", "m", 0, "-m"));
		result.addElement(new Option("\tCluster algorithm for finding clusters on the grid.\n\tThis clusterer is used to make the clusters formed by the ants clear. It is applied in the end, when no more ant cycles must be executed.", "w", 1, "-w"));
		result.addAll(Collections.list(super.listOptions()));
		if (this.optn_gridClusterer instanceof OptionHandler) {
			result.addElement(new Option("", "", 0, "\nOptions specific to clusterer " + optn_gridClusterer.getClass().getName() + ":"));
			result.addAll(Collections.list(((OptionHandler) optn_gridClusterer).listOptions()));
		}
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
		
		temp = Utils.getOption("kp", options);
		if (temp.length() > 0) {
			this.setPickUpThresholdConstant(Double.parseDouble(temp));
		}
		
		temp = Utils.getOption("kd", options);
		if (temp.length() > 0) {
			this.setDropDownThresholdConstant(Double.parseDouble(temp));
		}
		
		temp = Utils.getOption("kdf", options);
		if (temp.compareTo(tag_LumerFaietaLabel) == 0) {
			this.setDropDownFunction(new SelectedTag(tag_LumerFaieta, tags));
		}
		if (temp.compareTo(tag_DeneubourgEtAlLabel) == 0) {
			this.setDropDownFunction(new SelectedTag(tag_DeneubourgEtAl, tags));
		}
		
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
		
		temp = Utils.getOption("gx", options);
		if (temp.length() > 0) {
			this.setGridSizeX(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("gy", options);
		if (temp.length() > 0) {
			this.setGridSizeY(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("i", options);
		if (temp.length() > 0) {
			this.setAntCycles(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("ic", options);
		if (temp.length() > 0) {
			this.setAntsPerAntCycle(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("an", options);
		if (temp.length() > 0) {
			this.setAntsNum(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("avr", options);
		if (temp.length() > 0) {
			this.setAntsViewRange(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("as", options);
		if (temp.length() > 0) {
			this.setAntsSpeedDistributionLimit(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("adr", options);
		if (temp.length() > 0) {
			this.setAntsDropRange(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("adm", options);
		if (temp.length() > 0) {
			this.setAntsDropMemorySize(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("abdc", options);
		if (temp.length() > 0) {
			this.setAntsBehaviorDestructiveAfterNumFreeCycles(Integer.parseInt(temp));
		}

		temp = Utils.getOption("abdn", options);
		if (temp.length() > 0) {
			this.setAntsBehaviorDestructiveForNextPickUps(Integer.parseInt(temp));
		}
		
		this.setReplaceMissing(Utils.getFlag("m", options));
		
		temp = Utils.getOption("w", options);
		if (temp.length() > 0) {
			this.setGridClusterer(AbstractClusterer.forName(temp, null));
			this.setGridClusterer(AbstractClusterer.forName(temp, Utils.partitionOptions(options)));
		}
		else {
			this.setGridClusterer(AbstractClusterer.forName(EM.class.getName(), null));
			this.setGridClusterer(AbstractClusterer.forName(EM.class.getName(), Utils.partitionOptions(options)));
		}
		
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
		
		result.add("-kp");
		result.add("" + this.getPickUpThresholdConstant());
		
		result.add("-kd");
		result.add("" + this.getDropDownThresholdConstant());
		
		result.add("-kdf");
		switch (this.optn_kdFunction) {
			case tag_LumerFaieta: result.add(tag_LumerFaietaLabel); break;
			case tag_DeneubourgEtAl: result.add(tag_DeneubourgEtAlLabel); break;
		}
		
		result.add("-dist");
		result.add((this.optn_distanceFunction.getClass().getName()
			 + " " + Utils.joinOptions(this.optn_distanceFunction.getOptions())).trim());
		
		result.add("-gx");
		result.add("" + this.getGridSizeX());
		
		result.add("-gy");
		result.add("" + this.getGridSizeY());
		
		result.add("-i");
		result.add("" + this.getAntCycles());
		
		result.add("-ic");
		result.add("" + this.getAntsPerAntCycle());
		
		result.add("-an");
		result.add("" + this.getAntsNum());
		
		result.add("-avr");
		result.add("" + this.getAntsViewRange());
		
		result.add("-as");
		result.add("" + this.getAntsSpeedDistributionLimit());
		
		result.add("-adr");
		result.add("" + this.getAntsDropRange());
		
		result.add("-adm");
		result.add("" + this.getAntsDropMemorySize());
		
		result.add("-abdc");
		result.add("" + this.getAntsBehaviorDestructiveAfterNumFreeCycles());
		
		result.add("-abdn");
		result.add("" + this.getAntsBehaviorDestructiveForNextPickUps());
		
		if (this.optn_replaceMissing) {
			result.add("-m");
		}
		
		Collections.addAll(result, super.getOptions());
		
		result.add("-w");
		result.add(this.getGridClusterer().getClass().getName());
		if (this.getGridClusterer() instanceof OptionHandler) {
			String[] options = ((OptionHandler) this.getGridClusterer()).getOptions();
			if (options.length > 0) {
				result.add("--");
			}
			Collections.addAll(result, options);
		}
		
		return result.toArray(new String[result.size()]);
		
	}
	
	
	/**
	 * Coordinate in an imaginary two-dimensional coordinate system.
	 * <p>
	 * Instances of this class describe a point in an imaginary two-dimensional
	 * coordinate system. Only integer values are allowed as coordinate values.
	 * The Coordinate can also contain negative values. This object is literally
	 * used as an object, that means other objects manipulate and use this
	 * object. Therefore the Coordinate members are also public. The Coordinate
	 * does not alter other objects, except itself, and is passive.
	 */
	protected class Coordinate {
		
		/** Position of this Coordinate on the x axis. */
		public int x;
		
		/** Position of this Coordinate on the y axis. */
		public int y;
		
		
		/**
		 * Construct a Coordinate instance without providing coordinate values.
		 */
		public Coordinate () {
			this.x = 0;
			this.y = 0;
		}
		
		
		/**
		 * Construct a Coordinate instance with initial coordinate values.
		 * 
		 * @param xValue position of the new coordinate on the imaginary x axis.
		 * @param yValue position of the new coordinate on the imaginary y axis.
		 */
		public Coordinate (int xValue, int yValue) {
			this.x = xValue;
			this.y = yValue;
		}
		
		
		/**
		 * Construct a Coordinate instance at the same position where the
		 * template is.
		 * 
		 * @param template template for the new Coordinate instance to
		 *        construct.
		 */
		public Coordinate (Coordinate template) {
			this.x = template.x;
			this.y = template.y;
		}
		
		
		/**
		 * Sets the position for this Coordinate on the x axis.
		 * <p>
		 * As the {@linkplain #x} axis value is public, there is no need for this method. It
		 * is here for continuity reasons only.
		 * 
		 * @param value the position of this Coordinate on the x axis.
		 */
		public void setX(int value) {
			this.x = value;
		}
		
		
		/**
		 * Tells the position of this Coordinate on the x axis.
		 * <p>
		 * As the {@linkplain #x} axis value is public, there is no need for this method. It
		 * is here for continuity reasons only.
		 * 
		 * @return the position of this Coordinate on the x axis.
		 */
		public int getX() {
			return this.x;
		}
		
		
		/**
		 * Sets the position for this Coordinate on the y axis.
		 * <p>
		 * As the {@linkplain #y} axis value is public, there is no need for this method. It
		 * is here for continuity reasons only.
		 * 
		 * @param value the position of this Coordinate on the y axis.
		 */
		public void setY(int value) {
			this.y = value;
		}
		
		
		/**
		 * Tells the position of this Coordinate on the y axis.
		 * <p>
		 * As the {@linkplain #y} axis value is public, there is no need for this method. It
		 * is here for continuity reasons only.
		 * 
		 * @return the position of this Coordinate on the y axis.
		 */
		public int getY() {
			return this.y;
		}
		
		
		/**
		 * Moves this Coordinate up for a certain amount of steps on the
		 * coordinate system. When {@code steps} is negative, the coordinate is
		 * actually moved down.
		 * 
		 * @param steps steps to move this coordinate up.
		 */
		public void moveUp(int steps) { //Variable steps can also be negative.
			this.y = this.y + steps;
		}
		
		
		/**
		 * Moves this Coordinate down for a certain amount of steps on the
		 * coordinate system. When {@code steps} is negative, the coordinate is
		 * actually moved up.
		 * 
		 * @param steps steps to move this coordinate down.
		 */
		public void moveDown(int steps) { //Variable steps can also be negative.
			this.y = this.y - steps;
		}
		
		
		/**
		 * Moves this Coordinate left for a certain amount of steps on the
		 * coordinate system. When {@code steps} is negative, the coordinate is
		 * actually moved right.
		 * 
		 * @param steps steps to move this coordinate left.
		 */
		public void moveLeft(int steps) { //Variable steps can also be negative.
			this.x = this.x - steps;
		}
		
		
		/**
		 * Moves this Coordinate right for a certain amount of steps on the
		 * coordinate system. When {@code steps} is negative, the coordinate is
		 * actually moved left.
		 * 
		 * @param steps steps to move this coordinate right.
		 */
		public void moveRight(int steps) { //Variable steps can also be negative.
			this.x = this.x + steps;
		}
		
		
		/**
		 * Adds a given coordinate to this Coordinate.
		 * <p>
		 * Example: This Coordinate is (5,20), the coordinate to add is (5,-8),
		 * so the new position of this Coordinate is (10,12).
		 * 
		 * @param diff the coordinate to add to this Coordinate.
		 */
		public void add(Coordinate diff) {
			this.x = this.x + diff.x;
			this.y = this.y + diff.y;
		}
		
		
		/**
		 * Tells if this Coordinate is above of another coordinate on the
		 * imaginary coordinate system.
		 * 
		 * @param coordinate the coordinate to compare with.
		 * @return true, if this Coordinate is above the given {@code coordinate},
		 *         false, if the given {@code coordinate} is above this Coordinate or
		 *         both coordinates are at the same height on the y axis.
		 */
		public boolean isAboveOf(Coordinate coordinate) {
			return this.y > coordinate.y;
		}
		
		
		/**
		 * Tells if this Coordinate is below another coordinate on the imaginary
		 * coordinate system.
		 * 
		 * @param coordinate the coordinate to compare with.
		 * @return true, if this Coordinate is below the given {@code coordinate},
		 *         false, if the given {@code coordinate} is below this Coordinate or
		 *         both coordinates are at the same height on the y axis.
		 */
		public boolean isBelowOf(Coordinate coordinate) {
			return this.y < coordinate.y;
		}
		
		
		/**
		 * Tells if this Coordinate is on the left of another coordinate on the
		 * imaginary coordinate system.
		 * 
		 * @param coordinate the coordinate to compare with.
		 * @return true, if this Coordinate is on the left of the given
		 *         {@code coordinate}, false, if the given {@code coordinate} is
		 *         on the left of this Coordinate or both coordinates are at the
		 *         same position on the x axis.
		 */
		public boolean isLeftOf(Coordinate coordinate) {
			return this.x < coordinate.x;
		}
		
		
		/**
		 * Tells if this Coordinate is on the right of another coordinate on the
		 * imaginary coordinate system.
		 * 
		 * @param coordinate the coordinate to compare with.
		 * @return true, if this Coordinate is on the right of the given
		 *         {@code coordinate}, false, if the given {@code coordinate} is
		 *         on the right of this Coordinate or both coordinates are at
		 *         the same position on the x axis.
		 */
		public boolean isRightOf(Coordinate coordinate) {
			return this.x > coordinate.x;
		}
		
		
		/**
		 * Detects if a coordinate is a direct neighbor of this Coordinate. A
		 * coordinate is a direct neighbor of this Coordinate object, when it is
		 * directly above, below, left or right of this Coordinate.
		 * 
		 * @param coordinate the coordinate to compare with.
		 * @return true, if {@code coordinate} is a direct neighbor, false otherwise.
		 */
		public boolean isDirectNeighborOf(Coordinate coordinate) {
			if (coordinate == null) {
				return false;
			}
			if (coordinate.x + 1 == this.x || coordinate.x - 1 == this.x) {
				return true;
			}
			if (coordinate.y + 1 == this.y || coordinate.y - 1 == this.y) {
				return true;
			}
			return false;
		}
		
		
		/**
		 * Tells if this Coordinate has the same x value like the given
		 * coordinate.
		 * 
		 * @param coordinate the coordinate to compare with.
		 * @return true, if {@code coordinate} has the same x value like this one.
		 */
		public boolean hasSameX(Coordinate coordinate) {
			if (!(coordinate instanceof Coordinate)) { //E.g. null.
				return false;
			}
			return this.x == coordinate.x;
		}
		
		
		/**
		 * Tells if this Coordinate has the same y value like the given
		 * coordinate.
		 * 
		 * @param coordinate the coordinate to compare with.
		 * @return true, if {@code coordinate} has the same y value like this one.
		 */
		public boolean hasSameY(Coordinate coordinate) {
			if (!(coordinate instanceof Coordinate)) { //E.g. null.
				return false;
			}
			return this.y == coordinate.y;
		}
		
		
		
		/**
		 * Tells if this object equals the given object, by customized criteria.
		 * 
		 * @param comp the object to compare with.
		 * @return true, if this object equals the compared {@code comp} object.
		 * @see java.lang.Object#equals
		 */
		@Override
		public boolean equals(Object comp) {
			if ((!(comp instanceof Coordinate)) || comp == null) {
				return false;
			}
			return (this.x == ((Coordinate) comp).x && this.y == ((Coordinate) comp).y);
		}
		
		
		/**
		 * Returns a string representation of this object.
		 * 
		 * @return a string representing this object.
		 * @see java.lang.Object#toString()
		 */
		@Override
		public String toString() {
			return "(" + this.x + "," + this.y + ")";
		}
	}
	
	
	/**
	 * Placeholder for a {@link weka.core.Instance} object that can be managed on the
	 * grid.
	 * <p>
	 * GridInstances are agents for the weka.core.Instance objects, as the usual
	 * Instance objects are hard to manage on a grid. Therefore each
	 * GridInstance represents exactly one Instance in Instances. GridInstances
	 * can be placed on the grid, picked up and dropped by ants and tell their
	 * represented Instance on demand. So by moving GridInstances on the grid,
	 * Instance objects are moved on the grid indirectly.
	 * <p>
	 * Internally only the referenced index of an Instance is stored. Each
	 * GridInstance object relies on the global {@link LFCluster#data} variable,
	 * holding the Instances. A GridInstance can only be interpreted with
	 * respect of the data variable. So when the data variable changes, the
	 * indexes of the GridInstances are not valid anymore.
	 * 
	 * @see weka.core.Instance
	 * @see weka.core.Instances
	 */
	protected class GridInstance extends DenseInstance {
		
		/** For serialization. */
		private static final long serialVersionUID = 5059389908009115582L;
		
		/** Index of the represented Instance in Instances. */
		protected int instanceIndex = -1;
		
		/**
		 * Construct a new GridInstance without specifying a position on grid.
		 * 
		 * @param index index of the Instance in {@link LFCluster#data}.
		 */
		public GridInstance(int index) {
			super(1.0, new double[]{0,0});
			if (index >= data.size()) {
				throw new IndexOutOfBoundsException("The index of the Instance this GridInstance should represent is greater than the size of the Instances to be clustered.");
			}
			this.instanceIndex = index;
		}
		
		
//		private GridInstance(Instance instance) {
//			super(1.0, new double[]{0,0});
//		}
//		
//		
//		private GridInstance(double weight, double[] attValues) {
//			super(1.0, new double[]{0,0});
//		}
		
		
		/**
		 * Construct a new GridInstance and specifying its position on a grid.
		 * 
		 * @param index index of the Instance in {@link LFCluster#data}.
		 * @param position the position of this GridInstance on the grid.
		 */
		public GridInstance(int index, Coordinate position) {
			super(1.0, new double[]{position.x, position.y});
			if (index >= data.size()) {
				throw new IndexOutOfBoundsException("The index of the Instance this GridInstance should represent is greater than the size of the Instances to be clustered.");
			}
			this.instanceIndex = index;
		}
		
		
		/**
		 * Tells which Instance index is stored.
		 * 
		 * @return the index of the Instance.
		 */
		public int getIndexOfInstance() {
			return this.instanceIndex;
		}
		
		
		/**
		 * Returns the represented Instance object.
		 * 
		 * @return Instance for which this GridInstance is the placeholder.
		 * @see weka.core.Instance
		 */
		public Instance getInstance() {
			if (this.instanceIndex == -1) {
				return null;
			}
			return data.get(this.instanceIndex);
		}
		
		
		/**
		 * Sets the position of this GridInstance.
		 * 
		 * @param coordinate the new position of this GridInstance.
		 */
		public void setPosition(Coordinate coordinate) {
			m_AttValues[0] = (double) coordinate.x;
			m_AttValues[1] = (double) coordinate.y;
		}
		
		
		/**
		 * Tells the position of this GridInstance, where it is currently
		 * located.
		 * 
		 * @return the position of this object.
		 */
		public Coordinate getPosition() {
			return new Coordinate((int) m_AttValues[0], (int) m_AttValues[1]);
		}
		
		
		/**
		 * Tells if this object equals the given object, by customized criteria.
		 * 
		 * @param comp the object to compare with.
		 * @return true, if this object equals the compared {@code comp} object.
		 * @see java.lang.Object#equals
		 */
		@Override
		public boolean equals(Object comp) {
			if (!(comp instanceof GridInstance)) {
				return false;
			}
			return this.instanceIndex == ((GridInstance) comp).instanceIndex;
		}
		
		
		/**
		 * Returns a string representation of this object.
		 * 
		 * @return a string representing this object.
		 * @see java.lang.Object#toString()
		 */
		@Override
		public String toString() {
			return "" + this.instanceIndex + " (" + m_AttValues[0] + "," + m_AttValues[1] + ") GridInstance";
		}
		
	}
	
	
	/**
	 * Class of the ants running on the grid.
	 * <p>
	 * Instances of this class are the ants running on the grid, picking up and
	 * dropping GridInstance objects and perform the clustering task this way.
	 * 
	 * @see Grid
	 */
	protected class Ant {
		
		
		/**
		 * This is a memory for ants where they can remember positions. An ant
		 * can remember a position, e.g. where it did something like dropping or
		 * picking up a {@linkplain GridInstance}.
		 */
		protected class PositionMemory {
			
			/* Internal class. */
			protected class PositionMemoryEntry {
				
				public GridInstance gridInstance;
				public Coordinate position;
				
				public PositionMemoryEntry(GridInstance gridInstance, Coordinate pos) {
					this.gridInstance = gridInstance;
					this.position = new Coordinate(pos);
				}
				
			}
			
			/** The actual memory as array. */
			protected PositionMemoryEntry[] memorized = null;
			
			/** Position where to insert the next entry in {@link #memorized}. */
			private int nextInsertPosition = 0;
			
			/**
			 * Constructs a new PositionMemory.
			 * 
			 * @param size size of the memory.
			 */
			public PositionMemory(int size) {
				this.memorized = new PositionMemoryEntry[size];
				this.nextInsertPosition = 0;
			}
			
			
			/**
			 * Remember a new position together with a {@linkplain GridInstance}.
			 * 
			 * @param gridInstance GridInstance to be remembered together with
			 *        {@code pos}.
			 * @param pos position to be remembered together with {@code gridInstance}.
			 */
			public void memorize(GridInstance gridInstance, Coordinate pos) {
				if (!(gridInstance instanceof GridInstance) || !(pos instanceof Coordinate)) {
					return;
				}
					/*if (!(this.memorized[0] instanceof PositionMemoryEntry)) { //Activate this block (1/2), to force ants always to the first memory entry.
						this.memorized[0] = new PositionMemoryEntry(gridInstance, pos);
					}
					else {
						return;
					}*/
				this.memorized[this.nextInsertPosition] = new PositionMemoryEntry(gridInstance, pos);
				this.nextInsertPosition++;
				if (nextInsertPosition >= this.memorized.length) {
					nextInsertPosition = 0;
				}
			}
			
			
			/**
			 * Retrieve a specific memory position.
			 * 
			 * @param index index of the memory position entry to recall.
			 * @return position of the memorized entry with the provided {@code index}
			 *         or null when there is no entry at the given {@code index}.
			 */
			public Coordinate getPosition(int index) {
				if (this.memorized[index] instanceof PositionMemoryEntry) {
					return this.memorized[index].position;
				}
				else {
					return null;
				}
			}
			
			
			/**
			 * Gets the position of the {@linkplain GridInstance} that is most similar to the
			 * given {@code gridInstance}. Similarity is measured in distance of two
			 * GridInstance objects to each other, so the distance function of
			 * the options is used here.
			 * 
			 * @param gridInstance gridInstance to seek a similar position for.
			 * @return the position of the GridInstance in memory that is most
			 *         similar to the given {@code gridInstance} or null, if
			 *         nothing was found or the memory is not yet completely
			 *         filled.
			 * @see LFCluster#setDistanceFunction(DistanceFunction)
			 */
			public Coordinate getPositionOfMostSimilarGridInstance(GridInstance gridInstance) {
				PositionMemoryEntry bestMatch = null;
				double bestMatchDistance = 0.0;
					/*if (this.memorized[0] instanceof PositionMemoryEntry) { //Activate this block (2/2), to force ants always to the first memory entry.
						return this.memorized[0].position;
					}*/
				for (PositionMemoryEntry entry : this.memorized) {
					if (entry == null) {
						return null; //Do not use the memory yet, unless it is filled, because in the beginning there can be only one GridInstance and this always and drags the ant to one position.
					}
					double entryDistance = optn_distanceFunction.distance(gridInstance.getInstance(), entry.gridInstance.getInstance());
					if (!(bestMatch instanceof PositionMemoryEntry) || entryDistance < bestMatchDistance) {
						bestMatch = entry;
						bestMatchDistance = entryDistance;
					}
				}
				return (bestMatch instanceof PositionMemoryEntry) ? bestMatch.position : null;
			}
			
			
			/**
			 * Empties this PositionMemory.
			 */
			public void empty() {
				this.memorized = new PositionMemoryEntry[this.memorized.length];
				this.nextInsertPosition = 0;
			}
			
			
			/**
			 * Tells the size of this PositionMemory.
			 * 
			 * @return capacity of this PositionMemory.
			 */
			public int size() {
				return this.memorized.length;
			}
			
		}
		
		/**
		 * View range of this ant.
		 * 
		 * @see LFCluster#optn_antsViewRange
		 */
		protected int viewRange = 1;
		
		/**
		 * Walking speed of this ant on the grid.
		 * 
		 * @see LFCluster#optn_antsSpeedDistributionLimit
		 */
		protected int speed = 1;
		
		/**
		 * Drop range of this ant.
		 * 
		 * @see LFCluster#optn_antsDropRange
		 */
		protected int dropRange = 0;
		
		/**
		 * Position, where the ant currently is.
		 * 
		 * @see Coordinate
		 */
		protected Coordinate position = null;
		
		/**
		 * Load of the ant. When the ant carries a {@linkplain GridInstance} object it is
		 * stored here, if the ant is not carrying anything the variable is
		 * null.
		 * 
		 * @see GridInstance
		 */
		protected GridInstance carry = null;
		
		/**
		 * The memory of the ant for memorizing the recent drop locations. If
		 * the ant does not have a drop memory this variable is null.
		 */
		protected PositionMemory dropMemory = null;
		
		/**
		 * The destination to which the ant wants to go when it carries a
		 * {@linkplain GridInstance} and therefore wants to drop it somewhere. When the ant
		 * has a drop destination this variable holds a {@link Coordinate} object
		 * (otherwise null) and the ant must only perform moves on the grid that
		 * bring the ant closer to the destination.
		 * 
		 * @see Coordinate
		 */
		protected Coordinate dropDestination = null;
		
		/**
		 * After how many ant cycles carrying nothing the ant behaves
		 * destructive.
		 * 
		 * @see LFCluster#optn_antsBehaviorDestructiveAfterNumFreeCycles
		 */
		protected int behaviorDestructiveAfterNumFreeCycles = -1;
		
		/**
		 * How many pick ups the ant remains destructive.
		 * 
		 * @see LFCluster#optn_antsBehaviorDestructiveForNextPickUps
		 */
		protected int behaviorDestructiveForNextPickUps = 1;
		
		/**
		 * How many pick ups this ant made without regarding the environment of
		 * the picked up {@linkplain GridInstance} objects.
		 * 
		 * @see #behaviorDestructiveForNextPickUps
		 */
		protected int destructivePickUpsCount = 0;
		
		/**
		 * For statistical purpose. How often this ant was called.
		 */
		protected int callCounter = 0;
		
		/**
		 * For statistical purpose. How often this ant dropped a {@linkplain GridInstance}
		 * on the grid.
		 */
		protected int dropCounter = 0;
		
		/**
		 * For statistical purpose. How often this ant picked up a {@linkplain GridInstance}
		 * from the grid.
		 */
		protected int pickUpCounter = 0;
		
		
		/**
		 * Current ant cycle in which the ant is called.
		 */
		protected int currentAntCycle = -1;
		
		
		/**
		 * Ant cycle in which the ant did something recently.
		 */
		protected int lastActedAntCycle = -1;
		
		
		/**
		 * Constructs a new ant.
		 * 
		 * @param viewRange view range of this ant
		 * @param speed walking speed of this ant
		 * @param dropRange range in which the ant can drop carried {@linkplain GridInstance}
		 *        objects
		 * @param startPosition initial position of this ant on a grid
		 * @param dropMemorySize size of the memory to remember recent drop
		 *        locations
		 * @param behaviorDestructiveAfterNumFreeCycles after how many ant
		 *        cycles carrying nothing this ant becomes destructive
		 * @param behaviorDestructiveForNextPickUps how many pick ups the ant
		 *        behaves destructive.
		 */
		public Ant(int viewRange, int speed, int dropRange, Coordinate startPosition, int dropMemorySize, int behaviorDestructiveAfterNumFreeCycles, int behaviorDestructiveForNextPickUps) {
			this.viewRange = viewRange < 0 ? 0 : viewRange;
			this.speed = speed < 0 ? 0 : speed;
			this.dropRange = dropRange < 0 ? 0 : dropRange;
			this.setPosition(startPosition);
			this.dropMemory = dropMemorySize > 0 ? new PositionMemory(dropMemorySize) : null;
			this.behaviorDestructiveAfterNumFreeCycles = behaviorDestructiveAfterNumFreeCycles > 0 ? behaviorDestructiveAfterNumFreeCycles : -1;
			this.behaviorDestructiveForNextPickUps = behaviorDestructiveForNextPickUps >= 1 || behaviorDestructiveForNextPickUps == -1 ? behaviorDestructiveForNextPickUps : 1;
		}
		
		
		/**
		 * Tells the walk speed of this ant.
		 * 
		 * @return speed of this ant.
		 */
		public int getSpeed() {
			return this.speed;
		}
		
		
		/**
		 * Place this ant on the given position on the grid.
		 * 
		 * @param position the new position of the ant.
		 */
		public void setPosition(Coordinate position) {
			if (!grid.positionIsValid(position)) {
				throw new IllegalArgumentException("The new position for the ant is not a valid position on the grid! Can not place the ant here.");
			}
			this.position = position;
		}
		
		
		/**
		 * Tells the position where the ant currently is.
		 * 
		 * @return the current position of this ant.
		 */
		public Coordinate getPosition() {
			return this.position;
		}
		
		
		/**
		 * Tells if this ant is currently carrying a {@linkplain GridInstance}.
		 * 
		 * @return true, if the ant carries a GridInstance.
		 */
		public boolean carriesGridInstance() {
			return this.carry instanceof GridInstance;
		}
		
		
		/**
		 * Returns the drop memory of this ant.
		 * 
		 * @return the drop memory of this ant or null, if the ant has no drop
		 *         memory.
		 */
		protected PositionMemory getDropMemory() {
			return this.dropMemory;
		}
		
		
		/**
		 * Tells if this ant has a drop memory or not.
		 * 
		 * @return true, if the ant has a drop memory.
		 */
		protected boolean hasDropMemory() {
			return this.dropMemory instanceof PositionMemory;
		}
		
		
		/**
		 * Sets a drop destination for this ant. The ant walks in the direction
		 * to the destination only when it walks.
		 * 
		 * @param destination the drop destination of this ant.
		 */
		protected void setDropDestination(Coordinate destination) {
			if (destination != null && !grid.positionIsValid(destination)) {
				throw new IllegalArgumentException("The drop destination is not a valid Coordinate that can be found on the grid.");
			}
			this.dropDestination = destination;
		}
		
		
		/**
		 * Tells the drop destination of this ant.
		 * 
		 * @return the drop destination of this ant or null when the ant has no
		 *         drop destination.
		 */
		protected Coordinate getDropDestination() {
			return this.dropDestination;
		}
		
		
		/**
		 * Makes the ant forget its drop destination.
		 */
		protected void deleteDropDestination() {
			this.dropDestination = null;
		}
		
		
		/**
		 * Selects a favorite drop destination for a given {@linkplain GridInstance}. If the
		 * ant is not allowed to use its drop memory it will not tell favorite
		 * drop destinations.  
		 * 
		 * @param instance the GridInstance to search a favorite drop
		 *        destination for.
		 * @return The favorite drop destination or null, if there is no
		 *         favorite drop destination.
		 */
		protected Coordinate chooseDropDestinationFor(GridInstance instance) {
			Coordinate found = null;
			if (this.hasDropMemory()) { //If the ant is not allowed to use its memory no destination will be set below.
				found = this.getDropMemory().getPositionOfMostSimilarGridInstance(instance);
			}
			if (found instanceof Coordinate) { //Can also be null, when nothing similar was found in the memory.
				return found; 
			}
			else {
				return null;
			}
		}
		
		
		/**
		 * Tells if the ant has a drop destination.
		 * 
		 * @return true, if the ant has a drop destination.
		 */
		protected boolean hasDropDestination() {
			return this.dropDestination instanceof Coordinate;
		}
		
		
		/**
		 * Tell the ant to add a new drop location to the drop memory.
		 * 
		 * @param last the recently dropped {@linkplain GridInstance}
		 * @param where where the GridInstance {@code last} was dropped.
		 */
		protected void updateDropMemory(GridInstance last, Coordinate where) {
			if (this.hasDropMemory()) { //Is the ant allowed to use its drop memory?
				this.dropMemory.memorize(last, where);
			}
		}
		
		
		/**
		 * Resets the drop memory of this ant and start with a new, empty drop
		 * memory.
		 */
		protected void resetDropMemory() {
			if (this.hasDropMemory()) { //Is the ant allowed to use its drop memory?
				this.dropMemory = new PositionMemory(this.dropMemory.size());
			}
		}
		
		
		/**
		 * Calls an ant to action.
		 * 
		 * @param antCycle the current ant cycle in which the ant is called.
		 */
		public void call(int antCycle) {
			this.currentAntCycle = antCycle;
			this.work();
			this.walk();
		}
		
		
		/**
		 * Start the working process of this ant.
		 * <p>
		 * When the ant is called to work it can pick up or drop {@linkplain GridInstance}
		 * objects from or to the grid. Which action the ant can perform depends
		 * on the load status of the ant. When the ant is called to work, it
		 * reads its load status (if it is carrying a GridInstance) and
		 * automatically determines the appropriate work action. If the ant
		 * executes the determined work action or not depends on the
		 * circumstances (e.g. local environment). Therefore it is possible that
		 * the ant does nothing during a work call.
		 */
		protected void work() {
			if (this.behaviorDestructiveAfterNumFreeCycles > 0 && (this.currentAntCycle - this.lastActedAntCycle) > this.behaviorDestructiveAfterNumFreeCycles) { //Become destructive? Lumer/Faieta 1994, p. 507: Ants become destructive when they did not manipulate an instance for a preset number of ant cycles (not ant calls).
				this.destructivePickUpsCount = this.behaviorDestructiveForNextPickUps;
			}
			if ((this.destructivePickUpsCount > 0 || this.destructivePickUpsCount == -1) && grid.positionHasGridInstance(this.position) && !this.carriesGridInstance()) { //Behave destructive?
				this.pickGridInstance(this.position);
				return;
			}
			if (!this.carriesGridInstance() && grid.positionHasGridInstance(this.position)) { //Ant can pick up a gridInstance (normal).
				GridInstance gridInstance = grid.previewGridInstance(this.position);
				if (gridInstance instanceof GridInstance) {
					if (this.doesWantToPickUp(gridInstance)) {
						this.pickGridInstance(this.position);
					}
				}
				return;
			}
			if (this.carriesGridInstance() && grid.positionHasFreeStorage(this.position)) { //Ant can drop the gridInstance (normal).
				if (this.doesWantToDrop()) {
					this.dropGridInstance();
				}
				return;
			}
			return; // In all other cases the ant has nothing to do.
		}
		
		
		/**
		 * Moves an ant on the grid.
		 * <p>
		 * How far ants can go during one walk call depends on the individual
		 * speed of this ant. During one walk call the ant tries to execute all
		 * walk steps (the speed of the ant is measured in steps during one walk
		 * call), although it can be forced to stop walking earlier, e.g. by a
		 * grid border.
		 * <p>
		 * Usually an ant goes from one grid cell to a direct neighbor of this
		 * grid cell (up, down, left, right) during one step. On which grid
		 * cell the ant wants to go next is random. If the ant has a destination
		 * it is limited to directions that can bring the ant closer to the
		 * destination cell. Once the ant reaches the destination it forgets its
		 * destination.
		 */
		protected void walk() {
			if (!(this.position instanceof Coordinate)) {
				throw new NullPointerException("The ant position is not a valid Coordinate.");
			}
			Coordinate destination = null; //The destination, if any, the ant faces.
			if (this.carriesGridInstance() && this.hasDropDestination()) {
				destination = this.getDropDestination();
				if (m_Debug && !grid.positionIsValid(destination)) {
					throw new NullPointerException("The pick up destination of the ant is not a valid position on the grid.");
				}
			}
			int walkDirection = 0;
			if (this.dropDestination instanceof Coordinate && this.position.equals(destination)) {
				this.deleteDropDestination(); //To be safe, otherwise an undeleted but reached drop destination may cause an infinite loop later in choosing the walkDirection.
				return; //The ant reached its destination.
			}
			do { //Determine walk direction.
				walkDirection = rand.nextInt(4);
			}
			while (
					(walkDirection == 0 && grid.getDistanceToTopBorder(this.position) <= 0) || 
					(walkDirection == 1 && grid.getDistanceToBottomBorder(this.position) <= 0) || 
					(walkDirection == 2 && grid.getDistanceToLeftBorder(this.position) <= 0) || 
					(walkDirection == 3 && grid.getDistanceToRightBorder(this.position) <= 0) || 
					(this.getDropDestination() instanceof Coordinate && walkDirection == 0 && (this.position.isAboveOf(destination) || this.position.hasSameY(destination))) || 
					(this.getDropDestination() instanceof Coordinate && walkDirection == 1 && (this.position.isBelowOf(destination) || this.position.hasSameY(destination))) || 
					(this.getDropDestination() instanceof Coordinate && walkDirection == 2 && (this.position.isLeftOf(destination) || this.position.hasSameX(destination))) || 
					(this.getDropDestination() instanceof Coordinate && walkDirection == 3 && (this.position.isRightOf(destination) || this.position.hasSameX(destination)))
					);
			for (int i = 1; i <= this.speed; i++) {
				Coordinate intendedPosition = new Coordinate(this.position); //The next position where the ant wants to go to. But if the ant attempts to move out of the grid this is not a valid movement attempt. So test movement first with intendedPosition.
				switch (walkDirection) { //Determine the position where the ant should go next.
					case 0: intendedPosition.moveUp(1); break;
					case 1: intendedPosition.moveDown(1); break;
					case 2: intendedPosition.moveLeft(1); break;
					case 3: intendedPosition.moveRight(1); break;
				}
				if (!grid.positionIsValid(intendedPosition) || 
						destination instanceof Coordinate && (
								(walkDirection == 0 && intendedPosition.isAboveOf(destination)) || 
								(walkDirection == 1 && intendedPosition.isBelowOf(destination)) || 
								(walkDirection == 2 && intendedPosition.isLeftOf(destination)) || 
								(walkDirection == 3 && intendedPosition.isRightOf(destination))
								)
						) {
					break; //The ant can not go further in this direction.
				}
				this.setPosition(intendedPosition); //Go there.
				if (this.carriesGridInstance() && this.position.equals(this.getDropDestination())) {
					this.deleteDropDestination(); //The destination must be deleted immediately after the ant reached it and not when it picks up or drops a gridInstance there. If the destination is not deleted right now and the environment is not suitable for the carried gridInstance, the ant is pulled back to this location the next time it walks, although the ant already decided not to drop the carried GridInstance here.
					break; //Always stop walking when the ant reaches a destination, regardless of how many steps remain to give the ant a chance to work here the next time it is called.
				}
			}
			//grid.debug_notifyAntPresence(this.position);
		}
		
		
		/**
		 * Lists all {@linkplain GridInstance} objects in the current view range.
		 * <p>
		 * The GridInstance objects found depend on the size of the ants view
		 * range and the current position of this ant.
		 * 
		 * @return an {@link ArrayList} of all found GridInstance objects in the ants
		 *         view range.
		 * @see #viewRange
		 * @see #position
		 */
		protected ArrayList<GridInstance> getGridInstancesInViewRange() {
			ArrayList<GridInstance> list = new ArrayList<GridInstance>(5);
			int xStart = this.position.x - this.viewRange;
			int xStop = this.position.x + this.viewRange;
			int yStart = this.position.y - this.viewRange;
			int yStop = this.position.y + this.viewRange;
			for (int i = xStart; i <= xStop; i++) {
				for (int j = yStart; j <= yStop; j++) {
					GridInstance preview = grid.previewGridInstance(new Coordinate(i, j));
					if (preview instanceof GridInstance) {
						list.add(preview);
					}
				}
			}
			list.trimToSize();
			return list;
		}
		
		
		/**
		 * Lists all free positions on the grid in the current drop range.
		 * <p>
		 * The positions that are found depend on the size of the ants drop
		 * range and the current position of this ant.
		 * 
		 * @return an {@link ArrayList} of all found free positions in the ants
		 *         drop range.
		 * @see #dropRange
		 * @see #position
		 */
		protected ArrayList<Coordinate> getGridFreePositionsInDropRange() {
			ArrayList<Coordinate> list = new ArrayList<Coordinate>(5);
			int xStart = this.position.x - this.dropRange;
			int xStop = this.position.x + this.dropRange;
			int yStart = this.position.y - this.dropRange;
			int yStop = this.position.y + this.dropRange;
			for (int i = xStart; i <= xStop; i++) {
				for (int j = yStart; j <= yStop; j++) {
					Coordinate preview = new Coordinate(i, j);
					if (grid.positionHasFreeStorage(preview)) {
						list.add(preview);
					}
				}
			}
			list.trimToSize();
			return list;
		}
		
		
		/**
		 * Calculate the local similarity for the given {@code instance} at the
		 * current position of the ant.
		 * <p>
		 * If the ant is calculating the local similarity in order to determine
		 * the pick up probability and therefore does not carry a {@linkplain GridInstance},
		 * the GridInstance {@code instance} can also be from the grid and this method
		 * does not include the provided {@code instance} in the calculation of the
		 * local similarity.
		 * 
		 * @param instance the GridInstance object to calculate the local
		 *        similarity for
		 * @return the local similarity of {@code instance} as a double value
		 * @see #position
		 */
		protected double calculateFoi(GridInstance instance) { //Lumer/Faieta 1994, p. 503. See also Zhe et al. 2011, p. 117 and others listed in the thesis.
			double sum = 0.0;
			double foi = 0.0;
			ArrayList<GridInstance> neighbors = this.getGridInstancesInViewRange();
			if (neighbors.size() == 0) { //When there are no neighbors there is no further need to calculate foi. Return 0.
				return 0;
			}
			Instance regardedInstance = instance.getInstance();
			for (GridInstance neighbor : neighbors) {
				if (!this.carriesGridInstance() && neighbor.equals(instance)) { //For this GridInstance on the grid the foi should be calculated, but itself must not be regarded in the calculation.
					continue;
				}
				double distance = optn_distanceFunction.distance(regardedInstance, neighbor.getInstance());
				double div = 0.0;
				div = alpha + ((alpha * (this.speed - 1)) / optn_antsSpeedDistributionLimit); //E.g. Lumer/Faieta 1994, p. 504, Zhe et al. 2011, p. 118.
				sum = sum + (1.0 - (distance / div));
			}
			int viewRangeEdgeLength = (2 * this.viewRange) + 1;
			foi = ((1.0 / (viewRangeEdgeLength * viewRangeEdgeLength)) * sum); //1.0 instead of 1, because otherwise division with two integers and result is integer 0! Lumer/Faieta 1994, p. 503: "d^2 equals the total number of sites [grid cells] in the local area of interest".
			return foi > 0 ? foi : 0;
		}
		
		
		/**
		 * Calculates if the ant does want to pick up the given {@code instance}.
		 * <p>
		 * As this method depends on the calculation result of
		 * {@link #calculateFoi(GridInstance)}, the {@linkplain #position} of the ant is relevant.
		 * 
		 * @param instance the {@linkplain GridInstance} for which the pick up decision
		 *        should be made
		 * @return true, if the ant wants to pick up {@code instance} now, false if not.
		 */
		protected boolean doesWantToPickUp(GridInstance instance) {
			if (!grid.positionHasGridInstance(this.position) || !grid.previewGridInstance(this.position).equals(instance)) {
				if (m_Debug) {
					throw new RuntimeException("The position " + this.position + " of the ant does not contain the GridInstance " + instance + " the ant wants to pick up.");
				}
				return false;
			}
			double foi = this.calculateFoi(instance);
			double pre = optn_kp / (optn_kp + foi);
			double pickUpProbability = pre * pre;
			double randThreshold = rand.nextDouble();
			if (randThreshold <= pickUpProbability) { //Ant decided to pick up the gridInstance.
				return true;
			}
			else {
				return false;
			}
		}
		
		
		/**
		 * Calculates if the ant does want to drop the currently carried
		 * {@linkplain GridInstance}.
		 * <p>
		 * As this method depends on the calculation result of
		 * {@link #calculateFoi(GridInstance)}, the {@linkplain #position} of the ant is relevant.
		 * 
		 * @return true, if the ant wants to drop the carried GridInstance here,
		 *         false if the ant wants to keep on carrying the GridInstance
		 *         or if it does not carry an GridInstance object.
		 */
		protected boolean doesWantToDrop() {
			if (!this.carriesGridInstance()) {
				return false;
			}
			double foi = this.calculateFoi(this.carry);
			double dropProbability = 0.0;
			if (optn_kdFunction == tag_DeneubourgEtAl) { //Deneubourg et al. 1991
				double pre = foi / (optn_kd + foi);
				dropProbability = pre * pre;
			}
			else { //Lumer/Faieta 1994
				if (foi < optn_kd) {
					dropProbability = 2 * foi;
				}
				else {
					dropProbability = 1.0;
				}
			}
			double randThreshold = rand.nextDouble();
			if (randThreshold <= dropProbability) { //Ant decided to drop the gridInstance.
				return true;
			}
			else {
				return false;
			}
		}
		
		
		/**
		 * Executes the actual pick up process for an {@linkplain GridInstance} on the grid.
		 * 
		 * @param position the position from where the ant should pick up an
		 *        GridInstance.
		 * @return true, if the ant successfully picked up a GridInstance at the
		 *         {@code position}, false if the ant could not pick up the GridInstance
		 *         for any reason.
		 */
		protected boolean pickGridInstance(Coordinate position) {
			if (this.carriesGridInstance()) { //Check first: Can the ant pick up a gridInstance? Maybe this method is called when the ant carries a gridInstance.
				if (m_Debug) {
					throw new RuntimeException("The ant tries to pick up an GridInstance, but already has one.");
				}
				return false;
			}
			GridInstance instance = grid.pickGridInstance(position);
			if (instance instanceof GridInstance) { //The ant successfully picked up an instance.
				this.carry = instance;
				this.pickUpCounter++;
				this.destructivePickUpsCount = this.destructivePickUpsCount > 0 ? this.destructivePickUpsCount - 1 : this.destructivePickUpsCount;
				this.setDropDestination(this.chooseDropDestinationFor(this.carry)); //Only choose the drop location once (at pick up), otherwise (e.g. when the destination is updated after the ant reaches it) the ant is likely to stick to a position when it uses its memory and must drop its gridInstance there, as the memory always comes up with the same target to go to.
				this.notifyActionInAntCycle(this.currentAntCycle);
				return true;
			}
			else {
				return false;
			}
		}
		
		
		/**
		 * Executes the actual drop down process for an {@linkplain GridInstance} on the
		 * grid.
		 * 
		 * @return true, if the ant successfully dropped the carried
		 *         GridInstance at the {@code position}, false if the ant could not drop
		 *         the GridInstance for any reason, including the case when the
		 *         ant does not carry a GridInstance.
		 */
		protected boolean dropGridInstance() {
			if (!this.carriesGridInstance()) { //Check first: Does the ant have a gridInstance to drop? Maybe this method is called when the ant does not carry a gridInstance.
				if (m_Debug) {
					throw new RuntimeException("The ant tries to drop an GridInstance, but does not carry any.");
				}
				return false;
			}
			Coordinate dropPosition; //Drop somewhere in drop range or not? Determine the dropPosition first.
			if (this.dropRange > 0) {
				ArrayList<Coordinate> list = new ArrayList<Coordinate>();
				list = this.getGridFreePositionsInDropRange();
				int size = list.size();
				if (size == 0) { //No free positions found in the CoordinateArea. Running rand.nextInt(0) would cause an error.
					return false; //Can not drop.
				}
				int pos = rand.nextInt(size);
				dropPosition = new Coordinate(list.get(pos));
			}
			else {
				dropPosition = new Coordinate(this.position);
			} //Now the drop position is known and the GridInstance can be dropped regularly.
			if (grid.dropGridInstance(this.carry, dropPosition)) {
				this.deleteDropDestination();
				this.updateDropMemory(this.carry, dropPosition);
				this.carry = null;
				this.notifyActionInAntCycle(this.currentAntCycle);	
				return true;
			}
			else {
				if (this.dropRange > 0 && this.getGridFreePositionsInDropRange().size() > 0) {
					return this.dropGridInstance(); //Try again.
				}
				else {
					return false;
				}
			}
		}
		
		
		/**
		 * Notes the ant cycle in which the ant did something recently.
		 * 
		 * @param antCycle the ant cycle in which the ant did something
		 *        recently.
		 */
		protected void notifyActionInAntCycle(int antCycle) {
			this.lastActedAntCycle = antCycle;
		}
		
		
		/**
		 * Returns a string representation of this object.
		 * 
		 * @return a string representing this object.
		 * @see java.lang.Object#toString()
		 */
		@Override
		public String toString() {
			return "Ant(position=" + this.position + 
					", carries=" + this.carry + 
					", speed=" + this.speed + 
					", visualRange=" + this.viewRange + 
					")";
		}
		
		
		/**
		 * Shuts this ant down, so it is in a state where it no longer
		 * contributes to the clustering process on the grid and can be safely
		 * deleted for example.
		 * <p>
		 * When deleting an ant that is not shut down, the {@linkplain GridInstance}
		 * it may be carrying is also deleted and can not be clustered anymore.
		 */
		public void shutdown() {
			if (!this.carriesGridInstance()) {
				return; //Shutdown done.
			}
			Coordinate dropDestination = this.getDropDestination();
			this.deleteDropDestination(); //More freedom to walk somewhere.
			if (this.carriesGridInstance() && this.hasDropMemory()) {
				Coordinate positionOfMostSimilar = this.getDropMemory().getPositionOfMostSimilarGridInstance(this.carry);
				if (positionOfMostSimilar instanceof Coordinate) {
					this.setPosition(positionOfMostSimilar);
				}
			}
			int shutdownCounter = 0;
			while (this.carriesGridInstance()) {
				if (shutdownCounter <= antShutdownRegularAttempts) {
					this.work();
					shutdownCounter++;
				}
				else {
					this.dropGridInstance();
				}
				this.walk();
			}
			if (dropDestination instanceof Coordinate) { //Revert previous state, although ant is likely not used anymore. The dropDestination could have been also null before. 
				this.setDropDestination(dropDestination);
			}
		}
		
	}
	
	
	/**
	 * The grid object where the {@linkplain GridInstance} objects are placed and the ants
	 * run to cluster these GridInstance objects.
	 */
	protected class Grid {
		
		/**
		 * Size of the grid in the x dimension measured in number of grid cells.
		 */
		protected int xSize;
		
		/**
		 * Size of the grid in the y dimension measured in number of grid cells.
		 */
		protected int ySize;
		
		/**
		 * The surface of the grid as an array, where the {@linkplain GridInstance} are
		 * placed. The two dimensions of the array represent the two grid
		 * dimensions, where the first one is the x dimension.
		 */
		protected int[][] surface;
		
		/**
		 * An association of {@linkplain GridInstance} object indexes to positions, to
		 * answer the question where a GridInstance is without checking each
		 * position of the surface until the GridInstance is found.
		 */
		protected Coordinate[] gridInstances;
		
		
		/**
		 * For statistical purposes. Memorize how often ants visited each grid
		 * cell.
		 */
//		private int[][] debug_antPresence;
		
		/**
		 * For statistical purposes. Memorize how many {@linkplain GridInstance} objects were
		 * picked up from each grid cell.
		 */
//		private int[][] debug_pickUps;
		
		/**
		 * For statistical purposes. Memorize how many {@linkplain GridInstance} objects were
		 * dropped to each grid cell.
		 */
//		private int[][] debug_drops;
		
		
		/**
		 * Constructs a new grid.
		 * 
		 * @param x size of the grid in the x dimension.
		 * @param y size of the grid in the y dimension.
		 * @param gridInstanceCapacity how many {@linkplain GridInstance} objects are stored
		 *        on this grid.
		 */
		public Grid(int x, int y, int gridInstanceCapacity) {
			this.xSize = x > 0 ? x : 0;
			this.ySize = y > 0 ? y : 0;
			this.surface = new int[this.xSize][this.ySize];
			this.gridInstances = new Coordinate[(gridInstanceCapacity > 0 ? gridInstanceCapacity : 0)];
//			this.debug_antPresence = new int[this.xSize][this.ySize];
//			this.debug_pickUps = new int[this.xSize][this.ySize];
//			this.debug_drops = new int[this.xSize][this.ySize];
			for (int i = 0; i < this.xSize; i++) {
				for (int j = 0; j < this.ySize; j++) {
					this.surface[i][j] = -1; //Because 0 is already a valid GridInstance index.
//					this.debug_antPresence[i][j] = 0;
//					this.debug_pickUps[i][j] = 0;
//					this.debug_drops[i][j] = 0;
				}
			}
		}
		
		
		/**
		 * Tells if a given {@linkplain Coordinate} is a valid position on this
		 * grid.
		 * 
		 * @param position the position to be checked
		 * @return true if {@code position} is an addressable position, false otherwise.
		 */
		public boolean positionIsValid(Coordinate position) {
			if (!(position instanceof Coordinate)) {
				return false;
			}
			if (position.x >= 0 && position.x < this.xSize && position.y >= 0 && position.y < this.ySize) {
				return true;
			}
			else {
				return false;
			}
		}
		
		
		/**
		 * Tells if the given {@linkplain Coordinate} is available for storing a
		 * {@linkplain GridInstance} on the grid.
		 * 
		 * @param position the position to be checked
		 * @return true, if a GridInstance can be stored at {@code position}, false
		 *         if there is already a GridInstance or the {@code position} is not
		 *         on the grid.
		 */
		public boolean positionHasFreeStorage(Coordinate position) {
			if (!(position instanceof Coordinate) || !this.positionIsValid(position)) {
				return false;
			}
			return this.surface[position.x][position.y] < 0;
		}
		
		
		/**
		 * Tells if the given {@linkplain Coordinate} on the grid currently holds a
		 * {@linkplain GridInstance} object.
		 * 
		 * @param position the position to be checked
		 * @return true, if a {@code position} is on the grid and holds a GridInstance
		 *         object, false otherwise.
		 */
		public boolean positionHasGridInstance(Coordinate position) {
			if (!(position instanceof Coordinate) || !this.positionIsValid(position)) {
				return false;
			}
			return this.surface[position.x][position.y] >= 0;
		}
		
		
		/**
		 * Tells how many grid cells are between the given {@code position} and the top
		 * border of the grid.
		 * 
		 * @param position the {@linkplain Coordinate} for the measurement
		 * @return grid cell count between {@code position} and the top grid border.
		 */
		public int getDistanceToTopBorder(Coordinate position) {
			return this.ySize - position.y - 1;
		}
		
		
		/**
		 * Tells how many grid cells are between the given {@code position} and the bottom
		 * border of the grid.
		 * 
		 * @param position the {@linkplain Coordinate} for the measurement
		 * @return grid cell count between {@code position} and the bottom grid border.
		 */
		public int getDistanceToBottomBorder(Coordinate position) {
			return position.y - 0;
		}
		
		
		/**
		 * Tells how many grid cells are between the given {@code position} and the left
		 * border of the grid.
		 * 
		 * @param position the {@linkplain Coordinate} for the measurement
		 * @return grid cell count between {@code position} and the left grid border.
		 */
		public int getDistanceToLeftBorder(Coordinate position) {
			return position.x - 0;
		}
		
		
		/**
		 * Tells how many grid cells are between the given {@code position} and the right
		 * border of the grid.
		 * 
		 * @param position the {@linkplain Coordinate} for the measurement
		 * @return grid cell count between {@code position} and the right grid border.
		 */
		public int getDistanceToRightBorder(Coordinate position) {
			return this.xSize - position.x - 1;
		}
		
		
		/**
		 * Returns the {@linkplain GridInstance} at the given {@code position}, but does not pick up
		 * the GridInstance from the grid.
		 * 
		 * @param position Coordinate from where to preview the GridInstance
		 * @return the GridInstance at this {@code position} or null if no proper access
		 *         to the GridInstance was possible.
		 */
		public GridInstance previewGridInstance(Coordinate position) {
			if (!this.positionIsValid(position)) {
				return null;
			}
			int index = this.surface[position.x][position.y];
			if (index < 0) {
				return null; //No proper access to the gridInstance was possible or no gridInstance here.
			}
			return new GridInstance(index, position);
		}
		
		
		/**
		 * Pick up the {@linkplain GridInstance} from the given {@code position}.
		 * <p>
		 * The parameters are chosen like that to keep the method body small. In
		 * parallel access this method must be likely locked, so a smaller
		 * method body can lead to a faster unlock. Disadvantage is that the
		 * parameters are not consistent among other methods of this class. The
		 * parameters must have been validated before, this method omits the
		 * check. The check is done by {@link #pickGridInstance(Coordinate)}.
		 * 
		 * @param x x coordinate of the position from where to pick up a
		 *            GridInstance.
		 * @param y y coordinate of the position from where to pick up a
		 *            GridInstance.
		 * @return the picked up GridInstance.
		 * @see #pickGridInstance(Coordinate)
		 */
		protected final synchronized GridInstance doGridInstancePick(int x, int y) {
			if (this.surface[x][y] < 0) {
				return null;
			}
			int index = this.surface[x][y];
			this.surface[x][y] = -1;
			this.gridInstances[index] = null;
			return new GridInstance(index, new Coordinate(x, y));
		}
		
		
		/**
		 * Tells the grid to return the {@linkplain GridInstance} of the given {@code position} and to
		 * remove it from there.
		 * <p>
		 * This method mainly checks the given parameters and then calls the
		 * protected method {@link #doGridInstancePick(int, int)}, which executes the
		 * actual pick up process.
		 * 
		 * @param position position on the grid from where to get the GridInstance.
		 * @return the GridInstance from {@code position} or null on failure.
		 * @see #doGridInstancePick(int, int)
		 */
		public GridInstance pickGridInstance(Coordinate position) {
			if (!this.positionIsValid(position)) {
				return null;
			}
			GridInstance gridInstance = this.doGridInstancePick(position.x, position.y);
			//if (gridInstance instanceof GridInstance) {
			//	this.debug_notifyPickUp(position);
			//}
			return gridInstance;
		}
		
		
		/**
		 * Drops the {@linkplain GridInstance} to the given {@code position}.
		 * <p>
		 * The parameters are chosen like that to keep the method body small. In
		 * parallel access this method must be likely locked, so a smaller
		 * method body can lead to a faster unlock. Disadvantage is that the
		 * parameters are not consistent among other methods of this class. The
		 * parameters must have been validated before, this method omits the
		 * check. The check is done by
		 * {@link #dropGridInstance(GridInstance, Coordinate)}.
		 * 
		 * @param index index of the GridInstance to drop
		 * @param position where to drop the GridInstance with index {@code index}.
		 * @return true on success, false otherwise.
		 * @see #dropGridInstance(GridInstance, Coordinate)
		 */
		protected final synchronized boolean doGridInstanceDrop(int index, Coordinate position) { //The parameters must be valid and checked before! Keep the synchronized methods as small as possible to free the access to the grid soon.
			if (!this.positionHasFreeStorage(position)) {
				return false; //Maybe another ant was faster with dropping an gridInstance at this position.  
			}
			this.surface[position.x][position.y] = index;
			this.gridInstances[index] = position;
			return true;
		}
		
		
		/**
		 * Drops the given {@code gridInstance} to the {@code position} on the grid.
		 * <p>
		 * This method mainly checks the given parameters and then calls the
		 * protected method {@link #doGridInstanceDrop(int, Coordinate)}, which executes
		 * the actual drop down process.
		 * 
		 * @param gridInstance the GridInstance to be dropped to the grid.
		 * @param position where to drop the {@code gridInstance} on the grid.
		 * @return true, if the gridInstance was dropped successfully. False, if
		 *         the gridInstance was not dropped for any reason.
		 * @see #doGridInstanceDrop(int, Coordinate)
		 */
		public boolean dropGridInstance(GridInstance gridInstance, Coordinate position) {
			if (!this.positionIsValid(position)) {
				throw new IllegalArgumentException("Attempt to drop a grid instance outside of the grid! The Coordinate " + position.x + ", " + position.y + " is not on the grid.");
			}
			boolean success = this.doGridInstanceDrop(gridInstance.getIndexOfInstance(), position);
			//if (success) {
			//	this.debug_notifyDrop(position);
			//}
			return success;
		}
		
		
		/**
		 * Returns a random position on the grid, that holds a {@linkplain GridInstance}.
		 * 
		 * @return {@link Coordinate} of a position on the grid that holds a GridInstance.
		 */
		public Coordinate getRandomGridInstancePosition() {
			Coordinate position = null;
			while (position == null && this.gridInstances.length > 0) { //Usually the first attempt should lead to a usable result.
				int num = rand.nextInt(this.gridInstances.length);
				if (this.gridInstances[num] instanceof Coordinate) {
					position = new Coordinate(this.gridInstances[num]);
				}
			}
			return (position instanceof Coordinate) ? position : null;
		}
		
		
		/**
		 * Returns a random position on the grid, that can take a {@linkplain GridInstance}.
		 * 
		 * @return {@link Coordinate} of a position on the grid that holds no GridInstance.
		 */
		public Coordinate getRandomFreePosition() {
			Coordinate position = new Coordinate(rand.nextInt(this.xSize), rand.nextInt(this.ySize));
			while (!this.positionHasFreeStorage(position)) { //Must end at some time, as constant gridMinFreeSpace guarantees some free space on the grid.
				position = new Coordinate(rand.nextInt(this.xSize), rand.nextInt(this.ySize));
			}
			return position;
		}
		
		
		/**
		 * Returns a {@link weka.core.Instances} object, containing
		 * {@link weka.core.DenseInstance} objects to make the GridInstances on the
		 * grid accessible for a {@link weka.clusterers.Clusterer}.
		 * <p>
		 * These Instances can be clustered by any {@link weka.clusterers.Clusterer}.
		 * 
		 * @return an Instances object containing the GridInstances casted to
		 *         DenseInstance objects.
		 */
		public Instances getAllGridInstancesAsDenseInstances() {
			int size = this.gridInstances.length;
			ArrayList<Attribute> attributes = new ArrayList<Attribute>(2);
			attributes.add(0, new Attribute("x"));
			attributes.add(1, new Attribute("y"));
			Instances instances = new Instances("gridInstances", attributes, size);
			for (int i = 0; i < size; i++) {
				if (this.gridInstances[i] instanceof Coordinate) {
					double[] attValues = new double[2];
					attValues[0] = (double) this.gridInstances[i].x;
					attValues[1] = (double) this.gridInstances[i].y;
					instances.add(new DenseInstance(1.0, attValues));
				}
				else {
					if (m_Debug) {
						throw new RuntimeException("# ! Error in getAllGridInstancesAsDenseInstances: Position " + i + " of the internal gridInstances array does not contain a Coordinate.");
					}
					double[] attValues = new double[2];
					attValues[0] = 0.0;
					attValues[1] = 0.0;
					instances.add(new DenseInstance(1.0, attValues));
				}
			}
			return instances;
		}
		
		
		/**
		 * For statistical purposes. Tells the grid that an ant visited the
		 * given {@code position}.
		 * 
		 * @param position position where an ant was.
		 */
//		public void debug_notifyAntPresence(Coordinate position) {
//			if (position instanceof Coordinate) {
//				this.debug_antPresence[position.x][position.y]++;
//			}
//		}
		
		
		/**
		 * For statistical purposes. Tells the grid that an ant picked up a  
		 * {@linkplain GridInstance} from the given {@code position}.
		 * 
		 * @param position position where an ant picked up a GridInstance.
		 */
//		protected void debug_notifyPickUp(Coordinate position) {
//			if (position instanceof Coordinate) {
//				this.debug_pickUps[position.x][position.y]++;
//			}
//		}
		
		
		/**
		 * For statistical purposes. Tells the grid that an ant dropped a  
		 * {@linkplain GridInstance} to the given {@code position}.
		 * 
		 * @param position position where an ant dropped a GridInstance.
		 */
//		protected void debug_notifyDrop(Coordinate position) {
//			if (position instanceof Coordinate) {
//				this.debug_drops[position.x][position.y]++;
//			}
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
		
		if (m_Debug) { System.out.println("# Starting LFCluster."); }
		
		int dataSize = data.size();
		if (m_Debug) { System.out.println("# " + dataSize + " instances given."); }
		
		if (m_Debug) { System.out.println("# Test capabilities."); }
		getCapabilities().testWithFail(data);
		
		if (m_Debug) { System.out.println("# Initialization of global variables."); }
		this.alpha = optn_alpha;
		this.data = new Instances(data);
		this.data.compactify();
		this.data.setClassIndex(-1);
		this.grid = new Grid(optn_gridSizeX, optn_gridSizeY, dataSize);
		this.ants = new Ant[optn_antsNum];
		this.antCycles = 0;
		this.replaceMissingValuesFilter = new ReplaceMissingValues();
		this.rand = new Random(getSeed());
		
		optn_distanceFunction.setInstances(this.data);
		
		if (optn_replaceMissing) {
			if (m_Debug) { System.out.println("# Running replacement for missing values."); }
			this.replaceMissingValuesFilter.setInputFormat(this.data);
			this.data = Filter.useFilter(this.data, this.replaceMissingValuesFilter);
		}
		
		if (m_Debug) { System.out.println("# Preparing the grid."); }
		if (optn_gridSizeX < gridMinSize || optn_gridSizeY < gridMinSize) {
			throw new IllegalArgumentException("The grid can not be used, because a grid smaller than the minimum grid size of " + gridMinSize + " is not allowed.");
		}
		long spaceAvailable = optn_gridSizeX * optn_gridSizeY; //Type long is sufficient to hold the multiplication result of the two max. positive integer values: (2^31)^2 = 2^62 < 2^63 (calculation approximately, because for example max. positive integer is 2^31-1).
		if ((int) ((spaceAvailable * (100.0/gridMinFreeSpace)) + 1) < spaceAvailable - dataSize) {
			throw new IllegalArgumentException("There is not enough free space left on the grid. There must be " + gridMinFreeSpace + "% free space at least.");
		}
		for (int i = 0; i < dataSize; i++) {
			Coordinate position = this.grid.getRandomFreePosition();
			this.grid.dropGridInstance(new GridInstance(i, position), position);
		}
		
		if (m_Debug) { System.out.println("# Preparing the ants."); }
		if ((int) ((spaceAvailable * (100.0/gridMinFreeSpace)) + 1) < spaceAvailable - dataSize) {
			throw new IllegalArgumentException("There are too many ants. There must be " + gridMinFreeSpace + "% space at least without ants on the grid.");
		}
		int speedGroupRaise = optn_antsNum / optn_antsSpeedDistributionLimit; //How many ants of each speed group exist, e.g. 5000 ants, max speed 5 -> every 1000 ants have the same speed. There can be optn_antsNum % optn_antsSpeedDistributionLimit speed slots still left to distribute.
		int speed = 1; //The speed to use for the next ant by regular distribution.
		int nextSpeed = 1; //The speed to use for the next ant.
		for (int i = 0; i < this.ants.length; i++) {
			Coordinate randomPosition = new Coordinate(rand.nextInt(optn_gridSizeX), rand.nextInt(optn_gridSizeY)); //Also later it is possible that 2 ants are on the same grid cell.
			this.ants[i] = new Ant(optn_antsViewRange, nextSpeed, optn_antsDropRange, randomPosition, optn_antsDropMemorySize, optn_antsBehaviorDestructiveAfterNumFreeCycles, optn_antsBehaviorDestructiveForNextPickUps);
			//this.grid.debug_notifyAntPresence(randomPosition);
			if ((i + 1) % speedGroupRaise == 0) {
				speed++;
				nextSpeed = speed;
			}
			if (speed > optn_antsSpeedDistributionLimit && optn_antsSpeedDistributionLimit > 1) { //There may be optn_antsNum % optn_antsSpeedDistributionLimit ants left that still need a speed assigned. The rest, that is too small to raise every speed group by 1, gets a random speed.
				do {
					nextSpeed = rand.nextInt(optn_antsSpeedDistributionLimit);
				}
				while (nextSpeed == 0); //Exclude 0 as a speed value.
			}
		}
		if (m_Debug) {
			int[] speeds = new int[optn_antsSpeedDistributionLimit + 1];
			for (int i = 0; i < this.ants.length; i++) {
				int antSpeed = this.ants[i].getSpeed();
				speeds[antSpeed]++;
			}
			System.out.print("#   Ant speeds:");
			for (int i = 1; i < speeds.length; i++) {
				System.out.print(" " + i + "(" + speeds[i] + ')');
			}
			System.out.println(".");
		}
		
		if (m_Debug) { System.out.println("# Start ant clustering."); }
		for (this.antCycles = 1; this.antCycles <= optn_antCycles; this.antCycles++) { //Use natural count of ant cycles (=the first one is 1).
			for (int i = 0; i < optn_antsCallPerAntCycle; i++) {
				Ant ant = this.ants[rand.nextInt(this.ants.length)];
				ant.call(this.antCycles);
			}
			if (m_Debug && this.antCycles % debug_verboseEveryAntCyclesPassed == 0) { // When there is much to do this gives an notification that the algorithm is still running.
				System.out.println("#   Finished ant cycle " + this.antCycles + ".");
			}
		}
		if (m_Debug && (this.antCycles - 1) % debug_verboseEveryAntCyclesPassed != 0) {
			System.out.println("#   Finished ant cycle " + (this.antCycles - 1) + ".");
		}
		
		if (m_Debug) { System.out.println("# Ant clustering done. Shutdown ant population now."); }
		for (int i = 0; i < this.ants.length; i++) { //All ants must drop the gridInstances back to the grid before reading all gridInstance/s.
			this.ants[i].shutdown();
		}
		
		if (m_Debug) { System.out.println("# Cleanup after ant clustering."); }
		optn_distanceFunction.clean();
		Instances gridInstances = this.grid.getAllGridInstancesAsDenseInstances();
		if (gridInstances.size() != this.data.size()) {
			throw new RuntimeException("Unexpected instances size mismatch. The instances to cluster on the grid do not match the size of the given instances.");
		}
		
		if (optn_gridClusterer instanceof AntGridClusterer) { //=>Provide coordinates + instance values.
			InstancesOnAntGrid instancesOnAntGrid = new InstancesOnAntGrid(this.data);
			instancesOnAntGrid.setGridInstances(gridInstances);
			gridInstances = instancesOnAntGrid;
		}
		
		if (m_Debug) { System.out.println("# Start assigning instances on grid to clusters using the grid clusterer."); }
		this.optn_gridClusterer.buildClusterer(gridInstances);
		ClusterEvaluation ce = new ClusterEvaluation();
		ce.setClusterer(optn_gridClusterer);
		ce.evaluateClusterer(gridInstances);
		this.out_clusterAssignments = ce.getClusterAssignments();
		this.out_gridClustererResults = optn_gridClusterer.toString();
		
		this.grid = null; //Important for Weka.
		this.ants = null;
		
		if (m_Debug) { System.out.println("# LFCluster finished.\n"); }
		
		return;
		
	}
	
	
	/**
	 * Tells cluster count in the current result.
	 * 
	 * @return number of clusters in the current result.
	 */
	@Override
	public int numberOfClusters() throws Exception {
		return optn_gridClusterer.numberOfClusters();
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
		int index = LFCluster.indexOfInstanceInInstances(this.data, instance);
		if (index >= 0) { //Cluster known instance.
			return (int) out_clusterAssignments[index];
		}
		else { //Cluster new instance.
			if (m_Debug) {
				System.out.println("# ! Trying to cluster an instance that was not clustered before.");
			}
			return -1;
		}
	}
	
	
	/**
	 * Returns a string representation of this object.
	 * <p>
	 * This is also the output for the WEKA Explorer GUI.
	 * 
	 * @return a string representing this object.
	 * @see java.lang.Object#toString()
	 */
	@Override
	public String toString() {
		StringBuffer temp = new StringBuffer();
		temp.append("Hint: It is recommended to turn normalization off for the distance function used to calculate the local similarity as intended by Lumer/Faieta. Otherwise adjust the value for Alpha.");
		temp.append("\n\n= Output of the grid clusterer: =\n\n");
		temp.append(this.out_gridClustererResults);
		return temp.toString();
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
	protected static int indexOfInstanceInInstances(Instances instances, Instance instance) {
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
