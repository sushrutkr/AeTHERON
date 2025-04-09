import sys 
import os
import numpy as np
import pandas as pd
import json

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))
from dataloader.membrane_data_contruct import generateDatasetMembrane
from dataloader.flow_data_contruct import generateDatasetFluid

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

  stats_list_membrane = []
  stats_list_flow = []

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
      
      velocity_stats = calc_mean_variance(membrane_data[1].transpose(1,0,2),type="vector")
      pressure_stats = calc_mean_variance(membrane_data[2].transpose(1,0),type="scalar")
      force_stats = calc_mean_variance(membrane_data[3].transpose(1,0,2),type="vector")

      sim_stats_membrane = {
        "sim_id": sim_params["sim_id"],
        "velocity_mean": velocity_stats[0],
        "velocity_variance": velocity_stats[1],
        "force_mean": force_stats[0],
        "force_variance": force_stats[1],
        "pressure_mean": pressure_stats[0],
        "pressure_variance": pressure_stats[1],
        "pointMass_min": np.min(membrane_data[4].flatten()),
        "pointMass_max": np.max(membrane_data[4].flatten()),
        "max_coordinate": np.max([np.max(membrane_data[0][:,0,:]),
                                  np.max(membrane_data[0][:,1,:]),
                                  np.max(membrane_data[0][:,2,:])]),
        "min_coordinate": np.min([np.min(membrane_data[0][:,0,:]),
                                  np.min(membrane_data[0][:,1,:]),
                                  np.min(membrane_data[0][:,2,:])]),
        "nNodes": membrane_data[0].shape[0],
        "ntsteps": membrane_data[0].shape[2]
        }

      flow_data = generateDatasetFluid(pathName=sim_params["flow_file"],splitLen=1).combined_data()
      ntsteps, nfeatures, nx, ny, nz = flow_data.shape
      flow_data = flow_data.reshape(ntsteps, nfeatures, nx * ny * nz)

      velocity_stats = calc_mean_variance(flow_data.transpose(1,0,2)[0:3],type="vector")
      pressure_stats = calc_mean_variance(flow_data.transpose(1,0,2)[3],type="scalar")
      sim_stats_flow = {
        "sim_id": sim_params["sim_id"],
        "velocity_mean": velocity_stats[0],
        "velocity_variance": velocity_stats[1],
        "pressure_mean": pressure_stats[0],
        "pressure_variance": pressure_stats[1],
        "nNodes": flow_data.shape[2],
        "ntsteps": flow_data.shape[0]
        }
      
      del membrane_data, flow_data

  stats_list_membrane.append(sim_stats_membrane)
  stats_df = pd.DataFrame(stats_list_membrane)
  stats_df.to_csv("./data/sim_stats_membrane.csv", index=False)

  stats_list_flow.append(sim_stats_flow)
  stats_df = pd.DataFrame(stats_list_flow)
  stats_df.to_csv("./data/sim_stats_flow.csv", index=False)

  return


if __name__ == "__main__":
  print("Computing statisitics for normalization")
  main()