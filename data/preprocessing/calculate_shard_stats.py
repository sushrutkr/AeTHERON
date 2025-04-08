import sys 
import os
import numpy as np
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from dataloader.membrane_data_contruct import generateDatasetMembrane

def calc_mean_variance(data, type=None):
  """
  This subroutine assumes second(1) and third(2) dimensions are nNodes and ntsteps.
  """
  if type is None:
      raise ValueError("The 'type' parameter cannot be None")
    
  if type == "vector":
    components = np.stack([data[0].flatten(), data[1].flatten(), data[2].flatten()], axis=0)  # Shape: (3, 351400)
    component_sum = np.sum(components, axis=0)  # Shape: (351400,)
    mean = np.mean(component_sum)
    print("mean:", mean)

    squared_deviations = (components - mean) ** 2  # Shape: (3, 351400)
    variance = (1 / (data.shape[1] * data.shape[2])) * np.sum(squared_deviations)
    print("variance:", variance)
    return mean, variance
  
  elif type == "scalar":
    # For scalar, we only need min and max for min-max scaling, but keeping this for completeness
    return np.mean(data.flatten()), np.var(data.flatten())

def main():
  """
  This function is designed to compute statistics(mean and s.d.) of each shard of data

  shard : numerical simulation of a particular Strouhal number
  
  """

  stats_list = []
  with open("./data/sim_metadata.json", "r") as f:
    metadata = json.load(f)

    num_simulations = len(metadata)

    for sim in range(num_simulations):
      sim_params = metadata[sim]
      num_samples = int((sim_params["end"]-sim_params["start"])/sim_params["gap"] + 1 )
      print("Data ID : ", sim_params["sim_id"] , ", Number of data samples : ", num_samples)

      membrane_data = generateDatasetMembrane(sim_params["start"], 
                                              sim_params["end"], 
                                              sim_params["gap"], 
                                              splitLen=1, 
                                              folder=sim_params["memb_file"]).get_output_full()
      
      nNodes = membrane_data[0].shape[0]
      ntsteps = membrane_data[0].shape[2]

      pointMass_min = np.min(membrane_data[3].flatten)
      pointMass_max = np.max(membrane_data[3].flatten)

      max_coordinate = np.max([np.max(membrane_data[0][:,0,:]),
                               np.max(membrane_data[0][:,1,:]),
                               np.max(membrane_data[0][:,2,:])])

      min_coordinate = np.min([np.min(membrane_data[0][:,0,:]),
                               np.min(membrane_data[0][:,1,:]),
                               np.min(membrane_data[0][:,2,:])])

      velocity = calc_mean_variance(membrane_data[1].transpose(1,0,2),type="vector")
      force = calc_mean_variance(membrane_data[2].transpose(1,0,2),type="vector")

      sim_stats = {
                  "sim_id": sim_params["sim_id"],
                  "velocity_mean": velocity[0],
                  "velocity_variance": velocity[1],
                  "force_mean": force[0],
                  "force_variance": force[1],
                  "pointMass_min": pointMass_min,
                  "pointMass_max": pointMass_max,
                  "max_coordinate": max_coordinate,
                  "min_coordinate": min_coordinate,
                  "nNodes": nNodes,
                  "ntsteps": ntsteps
              }
      del membrane_data

  stats_list.append(sim_stats)
  stats_df = pd.DataFrame(stats_list)
  stats_df.to_csv("./data/sim_stats_membrane.csv", index=False)
                                               
  return


if __name__ == "__main__":
  print("Computing statisitics for normalization")
  main()