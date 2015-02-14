package weka.clusterers;

import weka.core.Instances;


/**
 * This class is an extension for the {@link weka.core.Instances} class. Clusterers that
 * are designed to cluster instances on an ant grid can use additional
 * information besides the attributes of the instances to cluster. This
 * InstancesOnAntGrid can store the additional information where an instance is
 * located on the grid.
 * <p>
 * The additional information where an instance is located on the grid is hidden
 * behind the normal attributes of the Instances. So this class is compatible to
 * Instances and its inherited methods work the same as for the Instances class.
 * To set or access the grid locations specific methods must be called.
 * 
 * @version 0.9
 * @author Christoph
 */
public class InstancesOnAntGrid extends Instances {
	
	/** For serialization */
	private static final long serialVersionUID = 8648783956471094360L;
	
	/** The positions of m_Instances on the grid. */
	protected Instances m_gridInstances = null;
	
	
	/**
	 * The default constructor.
	 * 
	 * @param dataset this parameter is the same as the parameter for the
	 *        {@linkplain Instances} class.
	 * @see weka.core.Instances
	 */
	public InstancesOnAntGrid(Instances dataset) {
		super(dataset);
		if (dataset instanceof InstancesOnAntGrid) {
			this.m_gridInstances = ((InstancesOnAntGrid) dataset).m_gridInstances;
		}
	}
	
	
	/**
	 * Sets or (when called the first time adds) the grid position information to
	 * this InstancesOnAntGrid. {@link #grid()} can not provide information
	 * before this method is called, because the information is not added when
	 * constructing InstancesOnAntGrid, in order to not change the expected
	 * behavior and known parameters of the constructor.
	 * 
	 * @param gridInstances {@linkplain Instances} that contain the grid position
	 *        information. Each attribute is for one grid dimension.
	 * @throws IllegalArgumentException if {@code gridInstances} is not an Instance
	 *         object or its size does not match the size of this object.
	 * @see #grid()
	 */
	public void setGridInstances(Instances gridInstances) throws IllegalArgumentException {
		if (!(gridInstances instanceof Instances)) {
			throw new IllegalArgumentException("The parameter must be an instance of Instances.");
		}
		int size = this.m_Instances.size();
		if (size != gridInstances.size()) {
			throw new IllegalArgumentException("There must be as much gridInstances as Instance objects stored in this InstancesOnGrid object.");
		}
		this.m_gridInstances = gridInstances;
	}
	
	
	/**
	 * Tells if there is grid information available for the stored
	 * Instance objects.
	 * 
	 * @return true, if grid information is available, false if not.
	 */
	public boolean gridInstancesAreAvailable() {
		return m_gridInstances instanceof Instances;
	}
	
	
	/**
	 * Access to the stored grid information. Can be used like a switch to
	 * access the grid instead of the stored instances. Usually methods that are
	 * known already from the Instance class access the Instance attributes of this
	 * object as expected (because many other classes rely on that behavior). To
	 * explicitly execute this known methods on the grid data just add grid()
	 * before the call. They also work on the grid as they did before on the
	 * instances. So for example
	 * {@code ((InstancesOnAntGrid) something).instance(37).value(0)}
	 * returns the first value of Instance with the index 37.
	 * {@code ((InstancesOnAntGrid) something).grid().instance(37).value(0)}
	 * returns the position of this Instance 37 in the first grid
	 * dimension.
	 * 
	 * @return the Instance object holding the grid positions.
	 */
	public Instances grid() {
		return this.m_gridInstances;
	}
	
	
	/**
	 * Tell how many dimensions the grid has. Equal call to
	 * {@code ((InstancesOnAntGrid) something).grid().numAttributes()}.
	 * 
	 * @return grid dimension count.
	 */
	public int numGridDimensions() {
		return this.m_gridInstances.numAttributes();
	}
	
}
