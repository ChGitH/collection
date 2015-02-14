package weka.clusterers;

import weka.clusterers.AbstractClusterer;
import weka.core.Instances;


public abstract class AbstractAntGridClusterer extends AbstractClusterer {

	/** For serialization */
	private static final long serialVersionUID = 5504272117517842988L;


	/**
	 * The {@code buildClusterer} as defined by the {@linkplain AbstractClusterer}.
	 * Given {@linkplain Instances} are taken as {@linkplain InstancesOnAntGrid}.
	 */
	@Override
	public void buildClusterer(Instances data) throws Exception {
		this.buildClusterer((InstancesOnAntGrid) data);
	}
	
	
	/**
	 * Build the clusterer with the given {@linkplain InstancesOnAntGrid}. It must set up the
	 * clusterer, so that it can start its clustering process.
	 * 
	 * @param data set of instances, that are laid out on a grid
	 * @throws Exception on error during building the clusterer.
	 */
	public abstract void buildClusterer(InstancesOnAntGrid data) throws Exception;
	
}
