package weka.clusterers;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Enumeration;
import java.util.Iterator;
import java.util.LinkedList;
import java.util.Vector;

import weka.core.Capabilities;
import weka.core.Capabilities.Capability;
import weka.core.DenseInstance;
import weka.core.DistanceFunction;
import weka.core.EuclideanDistance;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.SelectedTag;
import weka.core.Tag;
import weka.core.Utils;

/**
 * <!-- globalinfo-start -->
 * Clusterer to name instance groups, produced by an ant cluster algorithm on a
 * grid. It first tries to find related instances by their neighborhoods on the
 * grid and then, if there is a maximum cluster number given, join similar
 * groups to match that number.
 * <!-- globalinfo-end -->
 * 
 * <!-- options-start -->
 * Valid options are: <p/>
 * 
 * <pre> -di
 *  Search also at grid cells diagonal to the current cell for instances of the
 *  same cluster. If false only instances at the grid cells above, below, left
 *  and right around the current grid cell are added to the cluster. If set to
 *  true, also instances at diagonal positions are added to the cluster.</pre>
 * 
 * <pre> -jw &lt;num&gt;
 *  Size of the surrounding area around a single instance in which clusters are
 *  searched. If there is only one cluster type found in this area, a single
 *  instance is automatically added to this cluster, if the cluster contains
 *  more than one instance. Usually single instances in direct neighborhood to
 *  one cluster belong to this cluster.</pre>
 * 
 * <pre> -maxcluster &lt;num&gt;
 *  If there are more clusters found than allowed by this setting, clusters must
 *  be merged to match this value. Set to -1 to not merge clusters.</pre>
 * 
 * <pre> -comp &lt;type&gt;
 *  This setting specifies how clusters are compared. Clusters that are regarded
 *  as most similar by this comparison method are merged until the requirement
 *  of maximum number of clusters in the result is met.</pre>
 * 
 * <pre> -dist &lt;classname and options&gt;
 *  Distance function that is used for instance comparison according to the
 *  attributes of the instances.
 *  (default = weka.core.EuclideanDistance)</pre>
 *  
 * <!-- options-end -->
 * 
 * @version 0.9
 * @author Christoph
 */
public class AntGridClusterer extends AbstractAntGridClusterer {

	/** For serialization */
	private static final long serialVersionUID = -1770937100790529575L;
	
	/** Tag list. */
	public static final int tag_compareModeCentroid = 0;
	public static final String tag_compareModeCentroidLabel = "centroid";
	static final Tag[] tags = { new Tag(tag_compareModeCentroid, tag_compareModeCentroidLabel) };
	
	/** Search also in cells diagonal to the current one for cluster members. */
	protected boolean optn_alsoDiagonalNeighborsForClusterSearch = false; //-di
	
	/** Border width in grid cells around a single instance in which it can be joined to an existing cluster having more than one instance. */
	protected int optn_singleInstanceJoinEnvironmentWidth = 1; //-jw
	
	/** Maximum number of clusters in the result. */
	protected int optn_maxClusterNum = -1; //-maxcluster
	
	/**
	 * Compare mode to use for cluster comparison.
	 * <p>
	 * In the current implementation there is only one cluster compare mode
	 * available. This setting is here to give the user a choice later, if more
	 * are added.
	 */
	protected int optn_clusterCompareMode = tag_compareModeCentroid; //-comp
	
	/** The distance function used for determining the distance between instances. */
	protected DistanceFunction optn_distanceFunction = new EuclideanDistance(); //-dist
	
	/** Instances data to be clustered. */
	protected InstancesOnAntGrid data = null; //It must not be altered after the instances were read!
	
	/** The normal instances without grid position information. */
	protected Instances data2;
	
	/** Clustering results. */
	protected int[] out_clusterAssignments = null;
	
	/** Cluster count of the result. */
	protected int out_numClusters = -1;
	
	
	/** The default constructor of this class. */
	public AntGridClusterer() {
		super();
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
		result.enable(Capability.NOMINAL_ATTRIBUTES); //Attributes..
		result.enable(Capability.NUMERIC_ATTRIBUTES);
		result.enable(Capability.MISSING_VALUES);
		return result;
	}
	
	
	/**
	 * Tip text provider for the diagonal neighbors search setting.
	 * 
	 * @return Text that briefly describes the functionality of the diagonal
	 *         neighbors search setting. 
	 */
	public String alsoDiagonalNeighborsForClusterSearchTipText() {
		return "Also add instances to a cluster that are on a diagonal direction towards the current instance.";
	}
	
	
	/**
	 * Sets the diagonal neighbors search directive.
	 * 
	 * @param value true, if diagonal neighbors should be added to one cluster,
	 *        too, false if not.
	 */
	public void setAlsoDiagonalNeighborsForClusterSearch(boolean value) {
		this.optn_alsoDiagonalNeighborsForClusterSearch = value;
	}
	
	
	/**
	 * Tells the currently set neighbors search directive.
	 * 
	 * @return true, if diagonal neighbors should be added to one cluster, too,
	 *         false if not.
	 */
	public boolean getAlsoDiagonalNeighborsForClusterSearch() {
		return this.optn_alsoDiagonalNeighborsForClusterSearch;
	}
	
	
	/**
	 * Tip text provider for the single instance join environment width setting.
	 * 
	 * @return Text that briefly describes the functionality of the single
	 *         instance join environment width setting.
	 */
	public String singleInstanceJoinEnvironmentTipText() {
		return "When there is a cluster with a single instance it is added to an existing, larger cluster, when only instances of this larger cluster are found in the given range around the single instance.";
	}
	
	
	/**
	 * Sets the single instance join environment width.
	 * 
	 * @param value width of the single instance join environment measured in
	 *        grid cells.
	 * @throws IllegalArgumentException if in debug mode, {@code value} &lt;= 0 and not -1.
	 */
	public void setSingleInstanceJoinEnvironment(int value) throws IllegalArgumentException {
		if (value < -1 || value == 0) {
			if (m_Debug) {
				throw new IllegalArgumentException("The environment with around an instance must be a positive integer > 0 or -1 to not join single instances.");
			}
			else {
				value = -1;
			}
		}
		this.optn_singleInstanceJoinEnvironmentWidth = value;
	}
	
	
	/**
	 * Tells the currently set single instance join environment width.
	 * 
	 * @return width of the single instance join environment measured in grid
	 *         cells.
	 */
	public int getSingleInstanceJoinEnvironment() {
		return this.optn_singleInstanceJoinEnvironmentWidth;
	}
	
	
	/**
	 * Tip text provider for the maximum cluster number setting.
	 * 
	 * @return Text that briefly describes the functionality of the maximum
	 *         cluster number setting.
	 */
	public String maxClusterNumTipText() {
		return "Maximum number of clusters in the result.";
	}
	
	
	/**
	 * Sets how many clusters are allowed in the clustering result.
	 * 
	 * @param value how many clusters are allowed in the clustering result. Set
	 *        to -1 to set no limit.
	 * @throws IllegalArgumentException if in debug mode, {@code value} &lt;= 0 and not -1.
	 */
	public void setMaxClusterNum(int value) throws IllegalArgumentException {
		if (value < -1 || value == 0) {
			if(m_Debug) {
				throw new IllegalArgumentException("Only a positive integer > 0 is allowed for specifying the maximum number of clusters in the result. Set to -1 for no limit.");
			}
			else {
				value = -1;
			}
		}
		this.optn_maxClusterNum = value;
	}
	
	
	/**
	 * Tells how many clusters are currently allowed maximum.
	 * 
	 * @return how many clusters are currently allowed maximum.
	 */
	public int getMaxClusterNum() {
		return this.optn_maxClusterNum;
	}
	
	
	/**
	 * Tip text provider for the cluster compare mode setting.
	 * 
	 * @return Text that briefly describes the functionality of the cluster
	 *         compare mode setting.
	 */
	public String clusterCompareModeTipText() {
		return "Specify here how similar clusters on the ant grid should be identified.";
	}
	
	
	/**
	 * Sets the cluster compare mode that must be used.
	 * 
	 * @param value a {@link SelectedTag} describing the drop down function to
	 *        use.
	 * @throws IllegalArgumentException if in debug mode and {@code value} is not a
	 *         known {@link SelectedTag}.
	 */
	public void setClusterCompareMode(SelectedTag value) throws IllegalArgumentException {
		if (value.getTags() == tags) {
			this.optn_clusterCompareMode = value.getSelectedTag().getID();
		}
		else {
			if (m_Debug) {
				throw new IllegalArgumentException("The cluster compare mode can only be set to known tags.");
			}
			else {
				this.optn_clusterCompareMode = tag_compareModeCentroid;
			}
		}
	}
	
	
	/**
	 * Tells the currently set cluster compare mode.
	 * 
	 * @return the {@link SelectedTag} naming the current cluster compare mode.
	 */
	public SelectedTag getClusterCompareMode() {
		return new SelectedTag(this.optn_clusterCompareMode, tags);
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
	 * Provides information about the available options for this clusterer.
	 * 
	 * @return an {@link Enumeration} holding the descriptions of available {@link Option}s.
	 */
	@Override
	public Enumeration<Option> listOptions() {
		Vector<Option> result = new Vector<Option>();
		result.addElement(new Option("\tSearch also at grid cells diagonal to the current cell for instances of the same cluster.\n\tIf false only instances at the grid cells above, below, left and right around the current grid cell are added to the cluster. If set to true, also instances at diagonal positions are added to the cluster. Usually adding instances only above, below, left and right is the better choice.\n\t(default = false)", "di", 1, "-di"));
		result.addElement(new Option("\tSize of the surrounding area around a single instance in which clusters are searched.\n\tThe area is defined as a square around a single instance with this number of grid in each direction. If there is only one cluster type found in this area, a single instance is automatically added to this cluster, if the cluster contains more than one instance. Usually single instances in direct neighborhood to one cluster belong to this cluster. This setting helps to speed up the cluster merging process. Set to -1 to not treat single instances seperately.\n\t(default = 1)", "jw", 1, "-jw <num>"));
		result.addElement(new Option("\tMaximum number of clusters in the result.\n\tIf there are more clusters found than allowed by this setting, clusters must be merged to match this value. Usually instances of the same cluster are separated in different clusters on the grid by the ants, so it is recommended to merge clusters. Set to -1 to not merge clusters.", "maxcluster", 1, "-maxcluster <num>"));
		result.addElement(new Option("\tHow to compare clusters for determining which clusters should be merged.\n\tThis setting specifies how clusters are compared. Clusters that are regarded as most similar by this comparison method are merged until the requirement of maximum number of clusters in the result is met.", "comp", 1, "-comp <type>"));
		result.addElement(new Option("\tDistance function to use for instance comparison.\n\tThis distance function is used to determine the distance between two instances according to their attributes.\n\t(default = weka.core.EuclideanDistance)", "dist", 1, "-dist <classname and options>"));
		result.addAll(Collections.list(super.listOptions()));
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
		
		this.setAlsoDiagonalNeighborsForClusterSearch(Utils.getFlag("di", options));
		
		temp = Utils.getOption("jw", options);
		if (temp.length() > 0) {
			this.setSingleInstanceJoinEnvironment(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("maxcluster", options);
		if (temp.length() > 0) {
			this.setMaxClusterNum(Integer.parseInt(temp));
		}
		
		temp = Utils.getOption("comp", options);
		if (temp.compareTo(tag_compareModeCentroidLabel) == 0) {
			this.setClusterCompareMode(new SelectedTag(tag_compareModeCentroid, tags));
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

		if (this.optn_alsoDiagonalNeighborsForClusterSearch) {
			result.add("-di");
		}
		
		result.add("-jw");
		result.add("" + this.getSingleInstanceJoinEnvironment());
		
		result.add("-maxcluster");
		result.add("" + this.getMaxClusterNum());
		
		result.add("-comp"); // DEL kdFunction
		switch (this.optn_clusterCompareMode) {
			case tag_compareModeCentroid: result.add(tag_compareModeCentroidLabel); break;
		}
		
		result.add("-dist");
		result.add((this.optn_distanceFunction.getClass().getName()
			 + " " + Utils.joinOptions(this.optn_distanceFunction.getOptions())).trim());
		
		Collections.addAll(result, super.getOptions());
		
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
		public Coordinate() {
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
		 * Construct a Coordinate instance with initial coordinate values.
		 * 
		 * @param xValue position of the new coordinate on the imaginary x axis.
		 * @param yValue position of the new coordinate on the imaginary y axis.
		 */
		public Coordinate (double xValue, double yValue) {
			this.x = (int) xValue;
			this.y = (int) yValue;
		}
		
		
		/**
		 * Get a position as Coordinate object relative to this Coordinate.
		 * 
		 * @param fromX shift on the x axis starting from the x value of this
		 *        Coordinate.
		 * @param fromY shift on the y axis starting from the y value of this
		 *        Coordinate.
		 * @return the new Coordinate.
		 */
		public Coordinate getRelativeCoordinate(int fromX, int fromY) {
			return new Coordinate(this.x + fromX, this.y + fromY);
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
			if (!(obj instanceof Coordinate)) {
				return false;
			}
			return ((Coordinate) obj).x == this.x && ((Coordinate) obj).y == this.y;
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
	 * A surface to reflect the positions of the instances on the grid. For this
	 * clusterer just the grid surface is required, but not the whole
	 * functionality of a grid, to do the position calculations.
	 */
	protected class GridSurface {
		
		/**
		 * The surface of the grid as an array, where the {@linkplain InstancePlaceholder}
		 * objects are placed. The two dimensions of the array represent the two
		 * grid dimensions, where the first one is the x dimension.
		 */
		protected int[][] surface;
		
		/**
		 * Size of the grid surface in the x dimension measured in number of
		 * grid cells.
		 */
		private int xSize;
		
		/**
		 * Size of the grid surface in the y dimension measured in number of
		 * grid cells.
		 */
		private int ySize;
		
		
		/**
		 * The default constructor of this class.
		 * 
		 * @param xSize size of the grid surface in the x dimension.
		 * @param ySize size of the grid surface in the y dimension.
		 */
		public GridSurface(int xSize, int ySize) {
			this.surface = new int[xSize][ySize];
			this.xSize = xSize;
			this.ySize = ySize;
			for (int i = 0; i < xSize; i++) {
				for (int j = 0; j < ySize; j++) {
					this.surface[i][j] = -1; //Because 0 is already a valid InstancePlaceholder index.
				}
			}
		}
		
		
		/**
		 * Tells if a given {@linkplain Coordinate} is a valid position on this
		 * grid surface.
		 * 
		 * @param position the position to be checked
		 * @return true if {@code position} is an addressable position, false otherwise.
		 */
		public boolean isPositionOnSurface(Coordinate position) {
			if (position.x < 0 || position.y < 0 || position.x >= this.xSize || position.y >= this.ySize) {
				return false;
			}
			else {
				return true;
			}
		}
		
		
		/**
		 * Sets the value of a cell to the given value. Usually this is the
		 * index of an {@linkplain InstancePlaceholder}. The value -1 is also
		 * accepted, but to delete an index from the grid surface, better use
		 * {@link #removeInstancePlaceholderIndex(Coordinate)}.
		 * 
		 * @param position a {@linkplain Coordinate} addressing the grid surface cell to
		 *        be changed
		 * @param i value for the grid surface cell at {@code position}.
		 * @throws IllegalArgumentException if {@code position} is not a position on the
		 *         grid surface.
		 */
		public void setInstancePlaceholderIndex(Coordinate position, int i) throws IllegalArgumentException {
			if (!isPositionOnSurface(position)) {
				throw new IllegalArgumentException("The given coordinate is not a valid position on the grid surface.");
			}
			this.surface[position.x][position.y] = i;
		}
		
		
		/**
		 * Tells the value of a grid surface cell. Usually this is the index of
		 * an {@linkplain InstancePlaceholder}.
		 * 
		 * @param position the position to get the value from
		 * @return value of the cell {@code position} or -1 if not value is there.
		 * @throws IllegalArgumentException if {@code position} is not a position on the
		 *         grid surface.
		 */
		public int getInstancePlaceholderIndex(Coordinate position) throws IllegalArgumentException {
			if (!isPositionOnSurface(position)) {
				throw new IllegalArgumentException("The given coordinate is not a valid position on the grid surface.");
			}
			return this.surface[position.x][position.y];
		}
		
		
		/**
		 * Tells if a value is stored for a specific cell.
		 * 
		 * @param position the requested grid surface cell
		 * @return true if a value greater or equal 0 is stored at {@code position},
		 *         false otherwise.
		 */
		public boolean hasInstancePlaceholderIndex(Coordinate position) {
			if (!isPositionOnSurface(position)) {
				return false;
			}
			return this.surface[position.x][position.y] >= 0;
		}
		
		
		/**
		 * Removes a stored value from a given grid surface cell.
		 * 
		 * @param position the position to delete the value from
		 * @throws IllegalArgumentException if {@code position} is not a position on the
		 *         grid surface.
		 */
		public void removeInstancePlaceholderIndex(Coordinate position) throws IllegalArgumentException {
			if (!isPositionOnSurface(position)) {
				throw new IllegalArgumentException("The given coordinate is not a valid position on the grid surface.");
			}
			this.surface[position.x][position.y] = -1;
		}
		
	}
	
	
	/**
	 * Placeholder for a {@link weka.core.Instance} object that can be managed on the
	 * grid.
	 * <p>
	 * InstancePlaceholder objects are agents for the
	 * weka.core.Instance objects, as the usual Instance objects are hard to
	 * manage on a grid surface. Therefore each InstancePlaceholder represents
	 * exactly one Instance in Instances. InstancePlaceholder objects can be
	 * placed on the grid surface.
	 */
	protected class InstancePlaceholder {
		
		/** Index of the represented {@link weka.core.Instance}. */
		protected int instanceIndex;
		
		/** Link to the represented {@link weka.core.Instance}. */
		protected Instance instance;
		
		/** Position of this InstancePlaceholder, usually on the {@linkplain GridSurface}. */
		protected Coordinate position;
		
		/** {@linkplain Cluster} to which this InstancePlaceholder belongs. */
		protected Cluster cluster;
		
		/**
		 * The default constructor of this class.
		 * 
		 * @param instanceIndex index of the represented {@linkplain Instance}
		 * @param linkedInstance the Instance to be represented
		 * @param position position of this InstancePlaceholder on the {@linkplain GridSurface}.
		 */
		public InstancePlaceholder(int instanceIndex, Instance linkedInstance, Coordinate position) {
			this.instanceIndex = instanceIndex;
			this.instance = linkedInstance;
			this.position = position;
		}
		
		
		/**
		 * Tells the index of the represented {@linkplain Instance}.
		 * 
		 * @return the index of the represented {@linkplain Instance}.
		 */
		public int getInstanceIndex() {
			return this.instanceIndex;
		}
		
		
		/**
		 * Returns the represented {@linkplain Instance}.
		 * 
		 * @return the {@linkplain Instance} represented by this InstancePlaceholder.
		 */
		public Instance getInstance() {
			return this.instance;
		}
		
		
		/**
		 * Tells where this InstancePlaceholder is located.
		 * 
		 * @return the position of this InstancePlaceholder. It is usually on the
		 *         {@linkplain GridSurface}.
		 */
		public Coordinate getPosition() {
			return this.position;
		}
		
		
		/**
		 * Sets a {@linkplain Cluster} for this InstancePlaceholder.
		 * 
		 * @param cluster Cluster to which this InstancePlaceholder belongs.
		 */
		public void setCluster(Cluster cluster) {
			this.cluster = cluster;
		}
		
		
		/**
		 * Get the {@linkplain Cluster} of this InstancePlaceholder.
		 * 
		 * @return Cluster to which this InstancePlaceholder is assigned or
		 *         null, if it is not assigned to any cluster.
		 */
		public Cluster getCluster() {
			return this.cluster;
		}
		
		
		/**
		 * Returns a string representation of this object.
		 * 
		 * @return a string representing this object.
		 * @see java.lang.Object#toString()
		 */
		@Override
		public String toString() {
			return "[ip" + this.instanceIndex + "]";
		}
		
	}
	
	
	/**
	 * This is the cluster that can be assigned to the {@linkplain InstancePlaceholder}
	 * objects.
	 * <p>
	 * It must not perform actions by itself, but is only used by other
	 * classes.
	 */
	protected class Cluster implements Iterable<InstancePlaceholder> {
		
		/** All {@linkplain InstancePlaceholder} members of this cluster. */
		protected ArrayList<InstancePlaceholder> members;
		
		/** A calculated {@linkplain Instance}, that represents the centroid of this cluster and does not necessarily match an existing member. */
		protected Instance centroid;
		
		/** Indicates if the {@linkplain #centroid} is up to date. As calculating the centroid is costly, but it only changes on member changes, it is worth to track its up to date status. */
		protected boolean centroidUpToDate;
		
		
		/**
		 * The default constructor of this class.
		 */
		public Cluster() {
			this.members = new ArrayList<InstancePlaceholder>();
			this.centroidUpToDate = false;
		}
		
		
		/**
		 * Tells the size of this cluster, namely how many members it has.
		 * 
		 * @return the size of this Cluster.
		 */
		public int size() {
			return this.members.size();
		}
		
		
		/**
		 * Tells if this cluster contains a specific {@linkplain InstancePlaceholder} as
		 * member.
		 * 
		 * @param ip the InstancePlaceholder to search for
		 * @return true if this Cluster contains {@code ip}, false if not.
		 */
		public boolean contains(InstancePlaceholder ip) {
			return this.members.contains(ip);
		}
		
		
		/**
		 * Adds an {@linkplain InstancePlaceholder} to this Cluster.
		 * 
		 * @param ip the InstancePlaceholder to add.
		 */
		public void add(InstancePlaceholder ip) {
			this.members.add(ip);
			this.centroidUpToDate = false;
		}
		
		
		/**
		 * Tell this clusters all its members. Those that possibly existed
		 * before are removed.
		 * 
		 * @param members an {@linkplain ArrayList} containing all members of this cluster.
		 */
		public void setMembers(ArrayList<InstancePlaceholder> members) {
			this.members = members;
			this.members.trimToSize();
			this.centroidUpToDate = false;
		}
		
		
		/**
		 * Gets the {@linkplain InstancePlaceholder} from this Cluster, that is the
		 * {@code index}th member of this Cluster (first index is 0).
		 * 
		 * @param index index of the InstancePlaceholder in this Cluster.
		 * @return the InstancePlaceholder found at the given {@code index} or null
		 *         otherwise.
		 */
		public InstancePlaceholder get(int index) {
			return this.members.get(index);
		}
		
		
		/**
		 * Removes the {@linkplain InstancePlaceholder} from this Cluster, that is stored at
		 * the given {@code index} in this Cluster. 
		 * 
		 * @param index index of the cluster member to remove.
		 */
		public void remove(int index) {
			this.members.remove(index);
			this.centroidUpToDate = false;
		}
		
		
		/**
		 * Removes the given {@linkplain InstancePlaceholder} from this Cluster.
		 * 
		 * @param ip InstancePlaceholder to be removed from this Cluster.
		 */
		public void remove(InstancePlaceholder ip) {
			this.members.remove(ip);
			this.centroidUpToDate = false;
		}
		
		
		/**
		 * Tells the centroid {@linkplain Instance} of this Cluster. This is not
		 * necessarily a member of this cluster.
		 * 
		 * @return Instance as centroid of this cluster.
		 * @throws RuntimeException if this Cluster has no members.
		 */
		public Instance getCentroid() throws RuntimeException {
			if (this.centroidUpToDate) {
				return this.centroid;
			}
			if (this.members.size() == 0) {
				throw new RuntimeException("Can not calculate centroid of cluster, because the cluster has no members.");
			}
			int numAttributes = this.members.get(0).getInstance().numAttributes();
			double[] attValues = new double[numAttributes];
			for (InstancePlaceholder ip : this.members) {
				for (int i = 0; i < numAttributes; i++) {
					attValues[i] = attValues[i] + ip.getInstance().value(i);
				}
			}
			for (int i = 0; i < numAttributes; i++) {
				attValues[i] = attValues[i] / this.members.size();
			}
			this.centroid = new DenseInstance(1.0, attValues);
			this.centroidUpToDate = true;
			return this.centroid;
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
					return position < members.size();
				}
				@Override
				public InstancePlaceholder next() {
					InstancePlaceholder member = members.get(position);
					position++;
					return member;
				}
			};
		}
		
	}
	
	
	/**
	 * Compares two clusters and indicate their similarity by a double value.
	 */
	protected class ClusterComparator {
		
		/** Compare mode to be used. */
		protected int mode = 0;
		
		
		/**
		 * The default constructor of this class.
		 * 
		 * @param mode initial mode for cluster comparison.
		 */
		public ClusterComparator(int mode) {
			this.setMode(mode);
		}
		
		/**
		 * Sets the comparison mode that should be used.
		 * 
		 * @param mode comparison mode for cluster comparison.
		 * @throws IllegalArgumentException if {@code mode} is not a known comparison
		 *         mode.
		 */
		public void setMode(int mode) throws IllegalArgumentException {
			if (mode == tag_compareModeCentroid) {
				this.mode = mode;
			}
			else {
				throw new IllegalArgumentException("Currently only the cluster compare mode 0 is supported.");
			}
		}
		
		
		/**
		 * Tells the comparison mode that is currently set.
		 * 
		 * @return integer representing a cluster comparison mode.
		 */
		public int getMode() {
			return this.mode;
		}
		
		
		/**
		 * Compares two {@linkplain Cluster} using the chosen comparison mode.
		 * 
		 * @param cluster the Cluster to be compared
		 * @param compareTo the Cluster {@code cluster} is compared with
		 * @return comparison result as double value. The more similar the
		 *         clusters are, the smaller is the returned value.
		 * @see #setMode(int)
		 */
		public double compare(Cluster cluster, Cluster compareTo) {
			switch (mode) {
				case tag_compareModeCentroid: return compare_centroid(cluster, compareTo);
				default: throw new RuntimeException("Unknown cluster compare mode.");
			}
		}
		
		
		/**
		 * Runs the centroid comparison value.
		 * <p>
		 * The similarity value is the distance between the centroids of both
		 * clusters.
		 * 
		 * @param cluster the Cluster to be compared
		 * @param compareTo the Cluster {@code cluster} is compared with
		 * @return comparison result as double value. The more similar the
		 *         clusters are, the smaller is the returned value.
		 */
		private double compare_centroid(Cluster cluster, Cluster compareTo) {
			return optn_distanceFunction.distance(cluster.getCentroid(), compareTo.getCentroid());
		}
		
	}
	
	
	/**
	 * Builds the clusterer with the given {@link InstancesOnAntGrid}.
	 * 
	 * @param data InstancesOnAntGrid to be clustered
	 * @throws Exception for various reasons
	 */
	@Override
	public void buildClusterer(InstancesOnAntGrid data) throws Exception {
		if (!data.gridInstancesAreAvailable()) {
			throw new IllegalArgumentException("The instances must be an instance of class InstancesOnAntGrid and provide gridInstance information.");
		}
		this.data = new InstancesOnAntGrid(data);
		int size = this.data.size();
		if (this.data.grid().kthSmallestValue(0, 1) < 0 || this.data.grid().kthSmallestValue(1, 1) < 0) {
			throw new IllegalArgumentException("No attribute value below 0 allowed in the first two attributes, which are the coordinate values. No negative coordinate values are allowed on the grid.");
		}
		int xSize = (int) this.data.grid().kthSmallestValue(0, size) + 1; //When there is an index 7, the range is 0-7, so the size is 8.
		int ySize = (int) this.data.grid().kthSmallestValue(1, size) + 1;
		GridSurface surface = new GridSurface(xSize, ySize);
		ArrayList<InstancePlaceholder> instancePlaceholders = new ArrayList<InstancePlaceholder>(size);
		ArrayList<Cluster> clusters = new ArrayList<Cluster>();
		this.out_clusterAssignments = new int[size];
		LinkedList<InstancePlaceholder> unclustered = new LinkedList<InstancePlaceholder>();
		for (int i = 0; i < size; i++) {
			int x = (int) this.data.grid().instance(i).value(0);
			int y = (int) this.data.grid().instance(i).value(1);
			Coordinate position = new Coordinate(x, y);
			InstancePlaceholder ip = new InstancePlaceholder(i, this.data.instance(i), position);
			instancePlaceholders.add(i, ip);
			surface.setInstancePlaceholderIndex(position, i);
			unclustered.add(i, ip);
		}
		instancePlaceholders.trimToSize();
		optn_distanceFunction.setInstances(this.data);
		while (!unclustered.isEmpty()) { //Assign Instance/s to clusters.
			Cluster collectCluster = new Cluster();
			ArrayList<Coordinate> visited = new ArrayList<Coordinate>(); //Keep track of the coordinates already visited. Otherwise this seeking part is trapped in an infinite loop.
			LinkedList<InstancePlaceholder> tasks = new LinkedList<InstancePlaceholder>();
			tasks.offer(unclustered.pollFirst());
			while (!tasks.isEmpty()) {
				InstancePlaceholder current = tasks.pollFirst();
				visited.add(current.getPosition());
				Coordinate neighborCheck;
				neighborCheck = current.getPosition().getRelativeCoordinate(0, 1);
				if (surface.hasInstancePlaceholderIndex(neighborCheck) && !visited.contains(neighborCheck)) {
					tasks.offer(instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)));
				}
				neighborCheck = current.getPosition().getRelativeCoordinate(0, -1);
				if (surface.hasInstancePlaceholderIndex(neighborCheck) && !visited.contains(neighborCheck)) {
					tasks.offer(instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)));
				}
				neighborCheck = current.getPosition().getRelativeCoordinate(-1, 0);
				if (surface.hasInstancePlaceholderIndex(neighborCheck) && !visited.contains(neighborCheck)) {
					tasks.offer(instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)));
				}
				neighborCheck = current.getPosition().getRelativeCoordinate(1, 0);
				if (surface.hasInstancePlaceholderIndex(neighborCheck) && !visited.contains(neighborCheck)) {
					tasks.offer(instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)));
				}
				if (optn_alsoDiagonalNeighborsForClusterSearch) {
					neighborCheck = current.getPosition().getRelativeCoordinate(-1, 1);
					if (surface.hasInstancePlaceholderIndex(neighborCheck) && !visited.contains(neighborCheck)) {
						tasks.offer(instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)));
					}
					neighborCheck = current.getPosition().getRelativeCoordinate(1, 1);
					if (surface.hasInstancePlaceholderIndex(neighborCheck) && !visited.contains(neighborCheck)) {
						tasks.offer(instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)));
					}
					neighborCheck = current.getPosition().getRelativeCoordinate(-1, -1);
					if (surface.hasInstancePlaceholderIndex(neighborCheck) && !visited.contains(neighborCheck)) {
						tasks.offer(instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)));
					}
					neighborCheck = current.getPosition().getRelativeCoordinate(1, -1);
					if (surface.hasInstancePlaceholderIndex(neighborCheck) && !visited.contains(neighborCheck)) {
						tasks.offer(instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)));
					}
				}
				if (!collectCluster.contains(current)) {
					collectCluster.add(current);
					current.setCluster(collectCluster);
				}
				unclustered.remove(current);
			}
			clusters.add(collectCluster);
		}
		if (optn_singleInstanceJoinEnvironmentWidth > 0) { //Join clusters with only one Instance.
			int xMin = -1 * optn_singleInstanceJoinEnvironmentWidth;
			int xMax = optn_singleInstanceJoinEnvironmentWidth;
			int yMin = -1 * optn_singleInstanceJoinEnvironmentWidth;
			int yMax = optn_singleInstanceJoinEnvironmentWidth;
			ArrayList<Cluster> toRemove = new ArrayList<Cluster>(); //List of clusters to delete later. If they are removed immediately, it will cause a co-modification of clusters.
			for (Cluster cluster : clusters) {
				if (cluster.size() == 1) {
					InstancePlaceholder ip = cluster.get(0);
					boolean found = false;
					Cluster uniqueCluster = null;
					for (int i = xMin; i <= xMax; i++) {
						for (int j = yMin; j <= yMax; j++) {
							if (i == 0 && j == 0) {
								continue;
							}
							Coordinate neighborCheck = ip.getPosition().getRelativeCoordinate(i, j);
							if (surface.hasInstancePlaceholderIndex(neighborCheck)) {
								Cluster compareCluster = instancePlaceholders.get(surface.getInstancePlaceholderIndex(neighborCheck)).getCluster();
								if (!found) {
									if (compareCluster.size() > 1) { // Do not join with other single clusters.
										uniqueCluster = compareCluster;
										found = true;
									}
									continue;
								}
								else if (!compareCluster.equals(uniqueCluster)) {
									uniqueCluster = null;
									i = xMax + 1; //Also stop the outer loop.
									break;
								}
								else {
									continue; //Nothing to do.
								}
							}
						}
					}
					if (found && uniqueCluster instanceof Cluster) { //In the environment there is only one cluster.
						ip.setCluster(uniqueCluster);
						uniqueCluster.add(ip);
						cluster.remove(0);
						toRemove.add(cluster);
					}
				}
			}
			for (Cluster cluster : toRemove) {
				clusters.remove(cluster);
			}
		}
		if (optn_maxClusterNum > 0) { //Merge clusters to match optn_maxClustersNum.
			ClusterComparator clusterComparator = new ClusterComparator(optn_clusterCompareMode);
			while (clusters.size() > optn_maxClusterNum) {
				Cluster candidateOne = null;
				Cluster candidateTwo = null;
				double candidateDistance = 0.0;
				int size2 = clusters.size();
				for (int i = 0; i < size2; i++) { //Better implementation is possible, but here it is rebuilding the compared results every time clusters were merged. (Can also store all distances and rebuild only when clusters should be merged that were already merged.)
					Cluster cluster = clusters.get(i);
					for (int j = i + 1; j < size2; j++) {
						Cluster compareTo = clusters.get(j);
						double distance = clusterComparator.compare(cluster, compareTo);
						if (distance < candidateDistance || candidateOne == null) {
							candidateOne = cluster;
							candidateTwo = compareTo;
							candidateDistance = distance;
						}
					}
				}
				for (InstancePlaceholder mem : candidateTwo) {
					candidateOne.add(mem);
					mem.setCluster(candidateOne); // Do not remove mem from cluster, it disturbs this for loop. The cluster is deleted afterwards anyway.
				}
				clusters.remove(candidateTwo);
			}
		}
		int clusterNum = 0;
		for (Cluster cluster : clusters) {
			for (InstancePlaceholder ip : cluster) {
				this.out_clusterAssignments[ip.getInstanceIndex()] = clusterNum;
			}
			clusterNum++;
		}
		this.out_numClusters = clusterNum;
		this.optn_distanceFunction.clean();
		this.data2 = null; //Not needed anymore.
	}
	
	
	/**
	 * Tells cluster count in the current result.
	 * 
	 * @return number of clusters in the current result.
	 */
	@Override
	public int numberOfClusters() {
		return out_numClusters;
	}
	
	
	/**
	 * Cluster the given {@code instance}.
	 * 
	 * @param instance the {@linkplain Instance} to be clustered.
	 * @return cluster number of the {@code instance}.
	 */
	@Override
	public int clusterInstance(Instance instance) {
		return this.clusterProcessedInstance(instance);
	}
	
	
	/**
	 * Cluster an {@code instance} that was also used for building the clusterer.
	 * 
	 * @param instance the {@linkplain Instance} to be clustered.
	 * @return cluster number of the {@code instance}.
	 * @see #buildClusterer(Instances)
	 */
	private int clusterProcessedInstance(Instance instance) {
		int index = AntGridClusterer.indexOfInstanceInInstances(this.data, instance);
		if (index >= 0) { //Cluster known instance.
			return this.out_clusterAssignments[index];
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
		if (m_Debug) {
			ArrayList<ArrayList<Integer>> clusterContents = new ArrayList<ArrayList<Integer>>(out_numClusters);
			for (int i = 0; i < out_numClusters; i++) {
				clusterContents.add(i, new ArrayList<Integer>());
			}
			for (int i = 0; i < out_clusterAssignments.length; i++) {
				clusterContents.get(out_clusterAssignments[i]).add(new Integer(i));
			}
			for (int i = 0; i < clusterContents.size(); i++) {
				temp.append("Cluster " + i + ":");
				for (Integer entry : clusterContents.get(i)) {
					temp.append("," + entry.toString());
				}
				temp.append("\n");
			}
		}
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
