import numpy as np
import pandas as pd
import plotly.graph_objects as go
from simulation_database import simulations_database
from simulation_database import xslices_database
from pathlib import Path
import scipy.signal
from short_scripts.autocorrelation import *
import plotly.io as pio
import matplotlib.pyplot as plt
import os
import pyvista as pv
import time
from scipy.interpolate import griddata
import alphashape
from scipy.spatial import Delaunay
from shapely.geometry import Polygon, Point
from scipy.spatial import distance_matrix
from scipy.spatial import cKDTree
import matplotlib.colors as mcolors
from matplotlib import cm
from matplotlib.colors import ListedColormap
from matplotlib.colors import TwoSlopeNorm
#import plotly.express as px

import xml.etree.ElementTree as ET
from matplotlib.colors import LinearSegmentedColormap
import numpy as np


def read_colormap_from_xml(xml_file, colormap_name=None):
    """
    Read colormap from ParaView-style XML file.
    
    Parameters:
    -----------
    xml_file : str
        Path to the XML file containing ColorMap definitions
    colormap_name : str, optional
        Name of the specific colormap to load. If None, loads the first colormap.
    
    Returns:
    --------
    cmap : matplotlib.colors.LinearSegmentedColormap
        The matplotlib colormap object
    clim : tuple
        The (min, max) data range for the colormap
    colormap_info : dict
        Dictionary containing colormap metadata
    """
    
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # Find all ColorMap elements
    colormaps = root.findall('.//ColorMap')
    
    if not colormaps:
        raise ValueError("No ColorMap found in XML file")
    
    # Select colormap
    if colormap_name is not None:
        colormap = None
        for cm in colormaps:
            if cm.get('name') == colormap_name:
                colormap = cm
                break
        if colormap is None:
            available = [cm.get('name') for cm in colormaps]
            raise ValueError(f"ColorMap '{colormap_name}' not found. Available: {available}")
    else:
        colormap = colormaps[0]
    
    # Extract colormap information
    name = colormap.get('name', 'custom')
    space = colormap.get('space', 'RGB')
    
    # Extract all color points
    points = colormap.findall('Point')
    
    x_values = []
    colors = []
    
    for point in points:
        x = float(point.get('x'))
        r = float(point.get('r'))
        g = float(point.get('g'))
        b = float(point.get('b'))
        # o = float(point.get('o', 1.0))  # opacity, default to 1.0
        
        x_values.append(x)
        colors.append((r, g, b))
    
    # Sort by x values (in case they're not in order)
    sorted_data = sorted(zip(x_values, colors), key=lambda pair: pair[0])
    x_values, colors = zip(*sorted_data)
    
    x_values = list(x_values)
    colors = list(colors)
    
    # Determine data range
    x_min, x_max = min(x_values), max(x_values)
    clim = (x_min, x_max)
    
    # Normalize x values to [0, 1] for matplotlib
    x_norm = [(x - x_min) / (x_max - x_min) for x in x_values]
    
    # Create matplotlib colormap
    cmap = LinearSegmentedColormap.from_list(
        name, 
        list(zip(x_norm, colors)), 
        N=256
    )
    
    # Prepare metadata
    colormap_info = {
        'name': name,
        'space': space,
        'n_points': len(x_values),
        'x_range': (x_min, x_max),
        'x_values': x_values,
        'colors': colors
    }
    
    return cmap, clim, colormap_info

def cart2pol(x,y):
    rho = np.sqrt(x**2 + y**2)
    phi = np.arctan2(y,x)
    return (rho,phi)


class NektarCSVSlice:
    def __init__(self,U_ref = 1, L_ref = 0.25, rho_ref = 1.2, area_ref = 0.25*0.7, legend_label = 'Nektar++ Simulation'):
        self.U_ref      = U_ref
        self.L_ref      = L_ref
        self.rho_ref    = rho_ref
        self.area_ref   = area_ref
        self.CTU        = L_ref/U_ref
        self.variables  = []
        #self.timestamps = []
        #self.filename_readme = None
        self.legend_label = legend_label
        self.type = 'Nektar++'
        
    def import_csv_data(self,filename):
        
        self.Data = pd.read_csv(filename)
        
        if 'Cp' in self.Data.keys():
            self.Data.rename(columns={"Cp": "Avg Cp"}, inplace = True)
        if 'Cf_mag' in self.Data.keys():
            self.Data.rename(columns={"Cf_mag": "Avg Cf"}, inplace = True)
        if 'Cfx' in self.Data.keys():
            self.Data.rename(columns={"Cfx": "Avg Cfx"}, inplace = True)
        if '# x' in self.Data.keys():
            self.Data.rename(columns={"# x": "x"}, inplace = True)
        

class CharLESProbes:

    def __init__(self,U_ref = 12.5, L_ref = 0.25, rho_ref = 1.2, area_ref = 0.25*0.7, legend_label = 'CharLES Simulation'):
        self.U_ref      = U_ref
        self.L_ref      = L_ref
        self.rho_ref    = rho_ref
        self.area_ref   = area_ref
        self.CTU        = L_ref/U_ref
        self.variables  = []
        self.timestamps = []
        self.filename_readme = None
        self.legend_label = legend_label
        self.type = 'CharLES'
        
    def import_probes_README(self,filename_readme):

        f = open(filename_readme, "r")
        line = 0
        list_data = []
        for x in f:
            strings = x.split(' ')
            stringss = [txt for txt in strings if txt]
            if stringss[0] == '0': #when probes are appended it only keeps the last set
                list_data = []
            if line > 1:
                data = stringss
                list_data.append(data)
            if line == 0:
                VARS = x.split('VARS')[-1]
                VARS_list = VARS.split(' ')
                while '' in VARS_list:
                    VARS_list.remove('')
                VARS_list[-1] = VARS_list[-1][:-1]
                print("Importable VARS :", VARS_list)
                self.all_importable_variables = VARS_list
            line = line + 1

        Data = np.zeros((len(list_data),len(list_data[0])))
        rows_to_remove = 0
        for i in range(len(list_data)):
            try:
                Data[i,:] = np.array(list_data[i])
            except:
                Data = Data[:-1]
        self.Data = pd.DataFrame(Data, columns = ['Probe','x','y','z'])
        self.nb_probes = len(self.Data['x'])
        f.close() 
        self.filename_readme = filename_readme
        
    def read_variables(self,filename, variable):
        self.variables.append(variable.lower())
        timestamps = []

        f = open(filename, "r")
        for x in f:
            if '#PROBE NAME' in x:
                pass
            elif '# ' in x:
                pass
            else:
                strings   = x.split(' ')
                time_step = int(strings[0])
                time      = float(strings[1])
                nb_probes = strings[2]
                self.nb_probes = nb_probes
                timestamps.append(time)

                values    = np.array(strings[3:], dtype='float')
                self.Data['t='+str(time) + ':' + variable.lower()] = values
        self.timestamps = timestamps        

        f.close()
        
    def read_all_available_variables(self):
        
        if self.filename_readme == None:
            print("ERROR: .README not yet imported")
        else:
            path = os.path.split(self.filename_readme)[0]
            print(path)
            readme_filename = os.path.split(self.filename_readme)[1]
            print("README :", readme_filename)
            filename_prefix = readme_filename[:-7] #remove .README
            
            files = [i for i in os.listdir(path) if os.path.isfile(os.path.join(path,i)) and filename_prefix in i]
            files.remove(readme_filename)
            print("Files to import:", files)
            NbFiles = len(files)
            for i in range(NbFiles):
                file = path + '/' + files[i]
                myfile = open(file, "r")
                myline = myfile.readline()
                myline = myfile.readline()
                VARIABLE = myline.split('VAR')[-1]
                VARIABLE = VARIABLE.replace('*:', '')
                VARIABLE = VARIABLE.replace('\n', '')
                VARIABLE = VARIABLE.replace(' ', '')
                print('VAR = ', VARIABLE, 'from file =',files[i])
                myfile.close()
                
                self.read_variables(file, VARIABLE)
            N = len(self.Data.columns.values)
            print(self.Data.columns.values[0:min(8,N)], '... are the dataframe headers')
            print("Finished importing all variables")
            
            timestamp_idx = -1 
            time = self.timestamps[timestamp_idx]
            variable = 'proj(avg(p))'
            data = self.Data['t='+str(time) + ':' + variable]
            self.Data['t='+str(time) + ':' +'Avg Cp'] = data / (0.5 * self.rho_ref * self.U_ref*self.U_ref)
            self.Data['Avg Cp'] = data / (0.5 * self.rho_ref * self.U_ref*self.U_ref)
            
            variable = 'avg(tau_wall())'
            data = self.Data['t='+str(time) + ':' + variable]
            self.Data['t='+str(time) + ':' +'Avg Cf'] = data / (0.5 * self.rho_ref * self.U_ref*self.U_ref)
            self.Data['Avg Cf'] = data / (0.5 * self.rho_ref * self.U_ref*self.U_ref)
            
            variable = 'avg(tau_wall(0))'
            data = self.Data['t='+str(time) + ':' + variable]
            self.Data['t='+str(time) + ':' +'Avg Cfx'] = data / (0.5 * self.rho_ref * self.U_ref*self.U_ref)
            self.Data['Avg Cfx'] = data / (0.5 * self.rho_ref * self.U_ref*self.U_ref)

class CharLESForces:
    
    def __init__(self,U_ref = 12.5, L_ref = 0.25, rho_ref = 1.2, area_ref = 0.25*0.75):
        self.U_ref    = U_ref
        self.L_ref    = L_ref
        self.rho_ref  = rho_ref
        self.area_ref = area_ref
        self.CTU      = L_ref/U_ref
        self.drag_ref = None
        self.lift_ref = None
        self.sideforce_ref = None

    def import_forces(self,filename, initial_time = 0, simulation_name = 'CharLES Simulation', legend_label = None):
        
        if filename == None:
            self.forces_df = pd.DataFrame(np.zeros((10, 4)),columns=['time', 'CTUs', 'Lift Coeff.', 'Drag Coeff.'])
            self.forces_df['time'] = np.arange(0,0.01,0.01/10)
            self.forces_df['CTUs'] = self.forces_df['time'] / self.CTU 
            self.sampling_timestep = self.forces_df['time'][1] - self.forces_df['time'][0]
            self.sampling_timestep_CTU = self.forces_df['CTUs'][1] - self.forces_df['CTUs'][0]
        else:
            print("CharLESForces:import_forces: running")
            self.simulation_name = simulation_name
            if legend_label == None:
                self.legend_label    = simulation_name
            else:
                self.legend_label    = legend_label
    
            f = open(filename, "r")
            line = 0
            list_data = []
            for x in f:
                strings = x.split(' ')
                stringss = [txt for txt in strings if txt]
                if line == 2:
                    headers = stringss[1:]
                    headers[-1] = headers[-1].strip() #remove line break from string
                if line > 2:
                    data = stringss
                    list_data.append(data)
                line = line + 1
                
            Data = np.zeros((len(list_data),len(headers)))
            for i in range(len(list_data)):
                Data[i,:] = np.array(list_data[i])
            
            self.forces_df = pd.DataFrame(Data, columns = headers)
            self.forces_df['time'] = initial_time + self.forces_df['time'] - self.forces_df['time'][0]
            self.forces_df['CTUs'] = self.forces_df['time']/self.CTU
            self.forces_df['CTUs'] = self.forces_df['CTUs'] - self.forces_df['CTUs'][0]
            self.sampling_timestep = self.forces_df['time'][1] - self.forces_df['time'][0]
            self.sampling_timestep_CTU = self.forces_df['CTUs'][1] - self.forces_df['CTUs'][0]
            
            non_dimensional_factor = 2/(self.rho_ref*self.U_ref*self.U_ref*self.area_ref)
            self.forces_df['Drag Coeff.'] = non_dimensional_factor * self.forces_df['f-total-x']
            self.forces_df['Lift Coeff.'] = non_dimensional_factor * self.forces_df['f-total-z']
            self.forces_df['Pressure Drag Coeff.'] = non_dimensional_factor * self.forces_df['f-p-x']
            self.forces_df['Pressure Lift Coeff.'] = non_dimensional_factor * self.forces_df['f-p-z']
            self.forces_df['Viscous Drag Coeff.']  = non_dimensional_factor * self.forces_df['f-visc-x']
            self.forces_df['Viscous Lift Coeff.']  = non_dimensional_factor * self.forces_df['f-visc-z']
            
            print("CharLESForces:import_forces: finished")
    
    

class NektarForces:
    
    def __init__(self,U_ref = 12.5, L_ref = 0.25, rho_ref = 1.2, area_ref = 0.25*0.75):
        self.U_ref    = U_ref
        self.L_ref    = L_ref
        self.rho_ref  = rho_ref
        self.area_ref = area_ref
        self.CTU      = L_ref/U_ref
        self.drag_ref = None
        self.lift_ref = None
        self.sideforce_ref = None
        
        
    def import_forces(self,filename, initial_time = 0, simulation_name = 'Nektar Simulation', legend_label = None):
        
        if filename == None:
            self.forces_df = pd.DataFrame(np.zeros((2, 4)),columns=['time', 'CTUs', 'Lift Coeff.', 'Drag Coeff.'])
            self.forces_df['time'][1] = 0.001
            self.forces_df['CTUs'][1] = self.forces_df['time'][1] / self.CTU 
            self.sampling_timestep = self.forces_df['time'][1] - self.forces_df['time'][0]
            self.sampling_timestep_CTU = self.forces_df['CTUs'][1] - self.forces_df['CTUs'][0]
        elif(type(filename)==list):
            print("list type")
            self.simulation_name = simulation_name
            if legend_label == None:
                self.legend_label    = simulation_name
            else:
                self.legend_label    = legend_label
                
            for file in filename:
                forces_df = pd.read_csv(file) 
                forces_df.rename(columns={"time_norm": "CTUs", "coef_lift": "Lift Coeff.", "coef_drag": "Drag Coeff."}, inplace = True)
                
                self.sampling_timestep = forces_df['time'][1] - forces_df['time'][0]
                self.sampling_timestep_CTU = forces_df['CTUs'][1] - forces_df['CTUs'][0]
                
                forces_df['time'] = initial_time + forces_df['time'] - forces_df['time'][0]
                forces_df['CTUs'] = forces_df['CTUs'] - forces_df['CTUs'][0]
                
                if file == filename[0]:
                    self.forces_df = forces_df
                else:
                    #self.forces_df = self.forces_df + forces_df
                    self.forces_df.add(forces_df)
                
        else:
            self.simulation_name = simulation_name
            if legend_label == None:
                self.legend_label    = simulation_name
            else:
                self.legend_label    = legend_label
                
                
                
            self.forces_df = pd.read_csv(filename) 
            self.forces_df.rename(columns={"time_norm": "CTUs", "coef_lift": "Lift Coeff.", "coef_drag": "Drag Coeff."}, inplace = True)
            
            self.sampling_timestep = self.forces_df['time'][1] - self.forces_df['time'][0]
            self.sampling_timestep_CTU = self.forces_df['CTUs'][1] - self.forces_df['CTUs'][0]
            
            self.forces_df['time'] = initial_time + self.forces_df['time'] - self.forces_df['time'][0]
            self.forces_df['CTUs'] = self.forces_df['CTUs'] - self.forces_df['CTUs'][0]
            
        try:
            self.forces_df.rename(columns={
                "coef_drag_p": "Pressure Drag Coeff.", 
                "coef_drag_v": "Viscous Drag Coeff.", 
                "coef_lift_p": "Pressure Lift Coeff.", 
                "coef_lift_v": "Viscous Lift Coeff."}, inplace = True)
        except:
            pass
        
    
            

def plot_forces(list_simulation_names, pid_name = 'ifw_total', forces_type = 'Lift Coeff.'):
    fig5, axs5 = plt.subplots(figsize =(4, 4))
    fig6, axs6 = plt.subplots(figsize =(4, 4))
    fig7, axs7 = plt.subplots(figsize =(4, 4))
    fig8, axs8 = plt.subplots(figsize =(4, 4))
    for i in range(len(list_sim_name)):
        sim_name = list_sim_name[i]
        simulation_path = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Path']
        solver          = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Solver']
        area_ref        = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['area_ref']
        if solver == 'CharLES':
            filename = simulation_path + '/forces/'+pid +'.dat'
            if Path(filename).is_file() == False: #if file doesn't exist then
                if pid == 'ifw_total':
                    filename = simulation_path + '/forces/fw.dat'
                else:
                    print("pid not available")
                    filename = None

            forcesdataset = CharLESForces(U_ref = 12.5, L_ref = 0.25, rho_ref = 1.2, area_ref = area_ref)
            forcesdataset.import_forces(filename, initial_time = 0, simulation_name = 'CharLES Simulation', legend_label = None)
        elif solver == 'Nektar++':
            if pid == 'ifw_total':
                filename = simulation_path + '/forces/forces_split/FWING_TOTAL_forces.csv'
            elif pid == 'fw_wheel':
                 filename = simulation_path + '/forces/forces_split/LWF_TOTAL_forces.csv'  
            elif pid == 'ifw_mainplane':
                filename = simulation_path + '/forces/forces_split/LFW_fia_mp_forces.csv' 
            elif pid == 'ifw_flap1':
                filename = simulation_path + '/forces/forces_split/LFW_element_1_forces.csv'
            elif pid == 'ifw_flap2':
                filename = [simulation_path + '/forces/forces_split/LFW_element_2_forces.csv',simulation_path + '/forces/forces_split/LFW_element_4_forces.csv']
                #filename = None
            elif pid == 'ifw_pod':
                filename = simulation_path + '/forces/forces_split/LNB_nosebox_forces.csv'
            elif pid == 'ifw_support':
                filename = simulation_path + '/forces/forces_split/LNB_hanger_forces.csv'
            elif pid == 'ifw_endplate':
                filename = simulation_path + '/forces/forces_split/LFW_endplate_forces.csv'
            else:
                print('other PID data not available for nektar++')
                filename = None
            forcesdataset = NektarForces(U_ref = 1, L_ref = 0.25, rho_ref = 1.2, area_ref = area_ref)    
            forcesdataset.import_forces(filename, initial_time = 0, simulation_name = 'Nektar Simulation', legend_label = None) 
        else: 
            print("not implemented for Fidelity DES")
            
        print("Finished importing forces data for simulation", sim_name)
        data_forces             = forcesdataset.forces_df
        print(forces_type in data_forces.keys())
        if forces_type in data_forces.keys():
            data                    = data_forces[forces_type]
        else:
            data                    = 0*data_forces['CTUs'] 
      
        
        start_avg_index = 0 # TO DO
        mean_ = np.mean(data[start_avg_index:])
        std_  = np.std(data[start_avg_index:])
        max_  = np.max(data[start_avg_index:])
        min_  = np.min(data[start_avg_index:])
        spread = max_ - min_
        
        
        ##line plot
        axs5.plot(data_forces['CTUs'], data , label = sim_name)
        
        
        ## spectrum plot
        #(strouhal, S)   = scipy.signal.periodogram(data, 1/forcesdataset.sampling_timestep, scaling ='density')
        (strouhal, S)   = scipy.signal.welch(data,       1/forcesdataset.sampling_timestep_CTU, nperseg = int(len(data)/3))
        axs6.plot(strouhal[1:], S[1:], label = sim_name)
        
        ## histogram
        #points = np.linspace(min_,max_,100)
        counts, bins = np.histogram(data, bins = 20)
        axs7.hist(bins[:-1], bins, weights=counts, label = sim_name + f' avg : {mean_:.4f} +/- {std_:.4f}')
        
        
        ## autocorrelation
        length = len(data)
        if length > 100000:
            skip = 10
            lags=range(0, int(len(data)/10), skip)
        else:
            skip = 2
            lags=range(0, int(len(data)), skip) #these hyperparameters can be changed to make the computation run faster
            
        acorr = autocorr(data)
        axs8.plot(data_forces['CTUs'], acorr, label = sim_name)
        
        
        
    axs5.grid('on')
    axs5.set_xlabel('CTUs')
    axs5.set_ylabel(forces_type)
    axs5.legend()

    axs6.grid('on')
    axs6.set_xlabel('Strouhal')
    axs6.set_ylabel('PSD ' + forces_type)
    axs6.set_yscale('log')
    axs6.set_xscale('log')
    axs6.legend()

    axs7.grid('on')
    axs7.set_xlabel(forces_type)
    axs7.set_ylabel('Prob. Density')
    axs7.legend()

    axs8.grid('on')
    axs8.set_xlabel('Time Lag [CTUs]')
    axs8.set_ylabel('Autocorrelation')
    axs8.legend()

    return fig5,fig6,fig7,fig8,axs5,axs6,axs7,axs8
    

def plot_surface_slice(list_simulation_names, slice_name = 'Y = -711.8mm', variable = 'Avg Cp'):
    

    fig,axs = plt.subplots(figsize =(4, 4))

    if slice_name == 'Y = -711.8mm' and variable == 'Avg Cp':
        data_exp = pd.read_csv('E:/PhD/Data/Fackrell_Cp_centerline.csv') 
        axs.plot(data_exp['theta']-180, data_exp['Cp'], label = 'Fackrell exp. (isolated wheel)', linestyle = 'dashed')
        data_exp = pd.read_csv('E:/PhD/Data/Mears_Cp_centerline.csv') 
        axs.plot(data_exp['theta']-180, data_exp['Cp'], label = 'Mears exp. (isolated wheel)', linestyle = 'dashed')
        
        
    filexists = True
    for i in range(len(list_sim_name)):
        sim_name = list_sim_name[i]
        simulation_path = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Path']
        solver = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Solver']
        print(simulation_path)
        
        if solver == 'CharLES':
            if slice_name == 'Y = -250mm':
                readme_filename = simulation_path + '/probes/y250/y250_line.README'
            else:
                readme_filename = simulation_path + '/probes/y711/y711_line.README'
            
            ynormal_slice = CharLESProbes(U_ref = 12.5, L_ref = 0.25, rho_ref = 1.2, area_ref = 0.25*0.7)
            
            try:
                ynormal_slice.import_probes_README(readme_filename)            
                ##importing variable of interest
                ynormal_slice.read_all_available_variables()
            except:
                filexists = False
                
        elif solver == 'Nektar++':
            if slice_name == 'Y = -250mm':
                if variable in ['Avg Cp']:
                    filename = simulation_path + '/surfaces/Jun25/csv_files/y250_surf_avg.csv'                    
                elif variable in ['Avg Cf', 'Avg Cfx']:
                    filename = simulation_path + '/surfaces/Jun25/csv_files/y250_wss_avg.csv'
                else:
                    print("variable unknown")
            else:
                if variable in ['Avg Cp']:
                    filename = simulation_path + '/surfaces/Jun25/csv_files/y711_surf_avg.csv'
                    if os.path.exists(filename) == False:
                        filename = simulation_path + '/surfaces/LWF_tyre_main_surf_y711slice.csv'
                elif variable in ['Avg Cf', 'Avg Cfx']:
                    filename = simulation_path + '/surfaces/Jun25/csv_files/y711_wss_avg.csv'
                else:
                    print("variable unknown")
            
            ynormal_slice = NektarCSVSlice(U_ref = 1, L_ref = 0.25, rho_ref = 1.2, area_ref = 0.25*0.7, legend_label = 'Nektar++ Simulation')
            
            try:
                ynormal_slice.import_csv_data(filename)
            except:
                filexists = False
            
        else: #
            print("not implemented for Fidelity DES")
            
            
        print("File exists:", filexists)    
        
        if filexists == True:
            ## add cyclindrical coordinates
            if slice_name == 'Y = -711.8mm':
                rho, phi = cart2pol(ynormal_slice.Data['x'] - 0.0452, ynormal_slice.Data['z']- 0.2853)    
                ynormal_slice.Data['radius_rho'] = rho
                ynormal_slice.Data['angle'] = 180/np.pi * phi
                ynormal_slice.Data.sort_values(by = 'angle', inplace = True)
                ynormal_slice.Data.reset_index(drop = True)

            ##line plot
            if slice_name == 'Y = -250mm':
                x = ynormal_slice.Data['x']
                xc = (x - np.min(x)) / ynormal_slice.L_ref
                axs.plot( xc, ynormal_slice.Data[variable], '.', label = sim_name)
                axs.set_xlabel('x')
                
            else:                
                axs.plot(ynormal_slice.Data['angle'], ynormal_slice.Data[variable], label = sim_name)
                axs.set_xlabel('Angle')
        


    axs.grid('on')
    axs.set_ylabel(variable)
    axs.legend() 
    
    return fig,axs

def get_local_pressure_min(points, pressure_field, search_radius = 0.01):
    N = len(points)
    
    tree = cKDTree(points)
    
    local_min_indices = []
    df = pd.DataFrame(columns=['idx','x','y','z','Avg Cp','Delta Cp','local min'])
    # Query neighbors in radius
    for i in range(0,points.shape[0]):
        idx = tree.query_ball_point(points[i], r=search_radius)

        # Skip if fewer than 2 points found
        if len(idx) <= 1:
            continue

        neighbor_pressures = pressure_field[idx] #array of pressure in the neighborhood
        if pressure_field[i] == np.min(neighbor_pressures) and np.sum(neighbor_pressures == pressure_field[i]) == 1:
            #found local minimum within target radius
            print(i,"/",N, " Cp: ", round(pressure_field[i],3))
            
            
            ## try to check if they are cp min over a larger radius
            idx = tree.query_ball_point(points[i], r=5*search_radius)
            neighbor_pressures = pressure_field[idx]
            Delta_Cp = np.abs(pressure_field[i] - np.max(neighbor_pressures))
            if pressure_field[i] == np.min(neighbor_pressures) and np.sum(neighbor_pressures == pressure_field[i]) == 1:
                df.loc[len(df)] = [i, points[i,0], points[i,1], points[i,2], pressure_field[i], Delta_Cp, 'local min (r = 5cm)']          
            else:
                df.loc[len(df)] = [i, points[i,0], points[i,1], points[i,2], pressure_field[i], Delta_Cp, 'local min (r = 1cm)']
            local_min_indices.append(i)
    
    data_local_cp_min = df
    return data_local_cp_min

def read_pyvista_object(file, selected_variable, solver_type = 'CharLES'):
    print(selected_variable)
    print()
    print("reading pyvista object")
    rho_inf = 1.2
    U_inf = 12.5
    
    
    
    ymin = -1200/1000 
    ymax = 0/1000 
    zmin = -25/1000
    zmax = 1000/1000
    
    df = pd.DataFrame()
    
    start_time = time.time()
    print("Reading :", file)
    dataset = pv.read(file)
    
    if file.split('.')[-1] == 'case':
        #mesh = dataset[0] 
        mesh = dataset.combine()
    else: #if vtp file
        mesh = dataset
    print(file.split('.')[-1])
    list_available_variables = list(mesh.point_data.keys()) #access the first multiblock since there is 1 multiblock per time step (but we only have the last time step)
    print(list_available_variables)
    
    
    
    if selected_variable == 'Avg Cp0':
        print("CP0")
        if solver_type == 'CharLES':
            if 'AVG<P_TOTAL<>>' in list_available_variables:
                mesh.point_data['Avg Cp0'] = mesh.point_data['AVG<P_TOTAL<>>'] / (0.5 * rho_inf * U_inf **2)
            elif 'AVG<U>' in list_available_variables and 'AVG<P>' in list_available_variables:
                u_mag = np.linalg.norm(mesh.point_data['AVG<U>'], axis = 1)
                mesh.point_data['Avg Cp0'] = (mesh.point_data['AVG<P>'] + 0.5 * rho_inf * u_mag**2) / (0.5 * rho_inf * U_inf **2)
            else:
                print("total pressure not exported for this plane")
        elif solver_type == 'Fidelity DES':
            if 'p_timeAverage' in list_available_variables and 'U_timeAverage' in list_available_variables:
                u_mag = np.linalg.norm(mesh.point_data['U_timeAverage'], axis = 1)
                mesh.point_data['Avg U_mag'] = u_mag
                
                mesh.point_data['Avg Cp0'] = (mesh.point_data['p_timeAverage'] + 0.5 * rho_inf * u_mag **2 ) / (0.5 * rho_inf * U_inf **2)
            else:
                print("total pressure cannot be computed for this plane")             
        elif solver_type == 'Nektar++':
            mesh.point_data['Avg Cp0'] = mesh.point_data['Cp0']
        else:
            print("unknown solver type")
            
    
            
    elif selected_variable == 'Avg Cp':
        print("CP")
        if solver_type == 'CharLES':
            if 'AVG<P>' in list_available_variables:
                mesh.point_data['Avg Cp'] = mesh.point_data['AVG<P>'] / (0.5 * rho_inf * U_inf **2)
            elif 'PROJ<AVG<P>>' in list_available_variables:
                mesh.point_data['Avg Cp'] = mesh.point_data['PROJ<AVG<P>>'] / (0.5 * rho_inf * U_inf **2) 
            else:
                print("pressure not exported for this plane")
        elif solver_type == 'Fidelity DES':
            if 'p_timeAverage' in list_available_variables:
                mesh.point_data['Avg Cp'] = mesh.point_data['p_timeAverage'] / (0.5 * rho_inf * U_inf **2)
            else:
                print("pressure not exported for this plane")
        elif solver_type == 'Nektar++':
            if 'Cp' in list_available_variables:
                mesh.point_data['Avg Cp'] = mesh.point_data['Cp']
            elif 'p' in list_available_variables:
                mesh.point_data['Avg Cp'] = mesh.point_data['p'] / (0.5)
            else:
                print("pressure data not available")
        else:
            print("unknown solver type")
    
    
        
    elif selected_variable == 'Avg Cfx':
        print("CFX")
        
        if solver_type == 'CharLES':
            if 'AVG<TAU_WALL<0>>' in list_available_variables:
                print("creating Cfx field")
                mesh.point_data['Avg Cfx'] = mesh.point_data['AVG<TAU_WALL<0>>'] / (0.5 * rho_inf * U_inf **2)
            else:
                print("Cfx not available")
        elif solver_type == 'Nektar++':
            if 'Shear_x' in list_available_variables:
                mesh.point_data['Avg Cfx'] = mesh.point_data['Shear_x'] / (0.5)
            else:
                print("Cfx not available")
        else:
            print("unknown solver type")
            
    else:
        print("undefined variables")
              
    # Remove all point data except the target
    for field in list(mesh.point_data.keys()):
        if field != selected_variable:
            del mesh.point_data[field]
    # Remove all cell data except the target
    for field in list(mesh.cell_data.keys()):
        if field != selected_variable:
            del mesh.cell_data[field]
    
                        
    """  
    if ymin is not None and ymax is not None:
        mesh_clip = mesh.clip(normal='y', origin=(0, ymin, 0), invert=False)
        mesh_clip = mesh_clip.clip(normal='y', origin=(0, ymax, 0), invert=True)
        
    # Apply clipping for z bounds
    if zmin is not None and zmax is not None:
        mesh_clip = mesh.clip(normal='z', origin=(0, 0, zmin), invert=False)
        mesh_clip = mesh_clip.clip(normal='z', origin=(0, 0, zmax), invert=True) 
    
                
        
    mesh = mesh_clip
            
    points = mesh.points
    values = mesh.point_data[selected_variable]
    x = points[:,0]
    y = points[:,1]
    z = points[:,2] 
    
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df[selected_variable] = values
    """

    end_time = time.time()
    print("Reading VTK file (execution time) :", end_time-start_time, 's')        
        
    return mesh





def read_xslices(sim_name,xslice,selected_variable):
    simulation_path = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Path']
    solver_type     = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Solver']
    xslices_path    = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Xslices Path']
    row_idx = xslices_database[xslices_database['Slice Name'] == xslice].index[0]
    
    file = xslices_path + '\\' + xslices_database[solver_type][row_idx]
    mesh = read_pyvista_object(file, selected_variable, solver_type)
    
    
    
    ymin = -1200/1000 
    ymax = 0/1000 
    zmin = -25/1000
    zmax = 1000/1000
    
    df = pd.DataFrame()
    if ymin is not None and ymax is not None:
        mesh_clip = mesh.clip(normal='y', origin=(0, ymin, 0), invert=False)
        mesh_clip = mesh_clip.clip(normal='y', origin=(0, ymax, 0), invert=True)
        
    # Apply clipping for z bounds
    if zmin is not None and zmax is not None:
        mesh_clip = mesh.clip(normal='z', origin=(0, 0, zmin), invert=False)
        mesh_clip = mesh_clip.clip(normal='z', origin=(0, 0, zmax), invert=True) 
        
    mesh = mesh_clip
            
    points = mesh.points
    values = mesh.point_data[selected_variable]
    x = points[:,0]
    y = points[:,1]
    z = points[:,2] 
    
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df[selected_variable] = values
    
    
    
    return mesh,df


def read_surfaceVTK(sim_name, selected_variable):
    simulation_path = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Path']
    solver_type     = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Solver']
    path = simulations_database.loc[simulations_database['SimulationName'] == sim_name].iloc[0]['Path']
    if solver_type == 'CharLES':
        file = path + '\\' + 'vtk_export\surfaceIFW\surface_data.case'
        mesh = read_pyvista_object(file, selected_variable, solver_type)
        
    elif solver_type == 'Nektar++':
        if selected_variable == 'Avg Cp':
            nektarpath = path + '/surfaces/vtu_files/pressure_velocity/'
            blocks = pv.MultiBlock()
            files = [nektarpath + 'LFW_element_1_surf_avg.vtu', nektarpath + 'LFW_element_2_surf_avg.vtu', nektarpath + 'LFW_element_4_surf_avg.vtu',
                     nektarpath + 'LFW_endplate_surf_avg.vtu', nektarpath + 'LFW_fia_mp_surf_avg.vtu', nektarpath + 'LNB_hanger_surf_avg.vtu',
                     nektarpath + 'LNB_nosebox_surf_avg.vtu']
        
        else:
            nektarpath = path + '/surfaces/vtu_files/wall_shear_stress/'
            blocks = pv.MultiBlock()
            files = [nektarpath + 'LFW_element_1-wss_avg.vtu', nektarpath + 'LFW_element_2-wss_avg.vtu', nektarpath + 'LFW_element_4-wss_avg.vtu',
                     nektarpath + 'LFW_endplate-wss_avg.vtu', nektarpath + 'LFW_fia_mp-wss_avg.vtu', nektarpath + 'LNB_hanger-wss_avg.vtu',
                     nektarpath + 'LNB_nosebox-wss_avg.vtu']
            
            for file in files:
                block = read_pyvista_object(file, selected_variable, solver_type)
                blocks.append(block)
            mesh = blocks.combine()     
            
    else:
        print("solver type not defined")

    
    


    ymin = -1200/1000 
    ymax = 0/1000 
    zmin = -25/1000
    zmax = 1000/1000
    
    df = pd.DataFrame()        
    points = mesh.points
    values = mesh.point_data[selected_variable]
    x = points[:,0]
    y = points[:,1]
    z = points[:,2] 
    
    df['x'] = x
    df['y'] = y
    df['z'] = z
    df[selected_variable] = values
    
    return mesh,df



def compare_xliceVTKs(sim_name1, sim_name2, xslice, selected_variable, isocontour_threshold = 0, tyre_outline = True):
    
    mesh1,df1 = read_xslices(sim_name1, xslice, selected_variable)
    mesh2,df2 = read_xslices(sim_name2, xslice, selected_variable)
    
    if tyre_outline == True:
        df_outline_tyre = pd.read_csv(r'E:\PhD\Data\IFW3D_Wheels\CharLES\Mesh1A\charles_mesh1A_3_highertolerance\vtk_export\surfaceWheel\outline_tyre.csv')
        df_outline_tyre['x'] = np.mean(mesh1.points[:,0])
        coords_outline_tyre = df_outline_tyre[['x', 'y', 'z']].values
        line = pv.lines_from_points(coords_outline_tyre)
    
    
    interpolated_values = griddata(points = df1[['y','z']], values = df1[selected_variable], xi = df2[['y','z']], method = 'linear')
    df2['interpolated ' + selected_variable] = interpolated_values
    df2['delta ' + selected_variable] = df2[selected_variable] - interpolated_values
    mesh2.point_data['interpolated ' + selected_variable] = interpolated_values
    mesh2.point_data['delta ' + selected_variable] = df2['delta ' + selected_variable]
    
    
    
    
    if selected_variable == 'Avg Cp':
        clim = [-4,1]
        cmap = 'viridis'
        xml_file = r'E:\PhD\Data\MRL_colorscheme\cp.xml'
        cmap, clim, info = read_colormap_from_xml(xml_file, colormap_name=None)
    elif selected_variable == 'Avg Cp0':
        clim = [-0.5,1]
        cmap = 'viridis'
        xml_file = r'E:\PhD\Data\MRL_colorscheme\cp0.xml'
        cmap, clim, info = read_colormap_from_xml(xml_file, colormap_name=None)
    else:
        clim = [-1,1]
        cmap = 'coolwarm'
    plotter = pv.Plotter(shape=(1, 3), off_screen=False)
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh1, scalars=selected_variable, clim=clim, cmap=cmap, show_scalar_bar=True, n_colors=20,
                     scalar_bar_args={
                        'title': selected_variable,
                        'position_x': 0.25,        # Center horizontally
                        'position_y': 0.05,       # Near the bottom
                        'width': 0.5,             # Width of the scalar bar (30% of window)
                        'height': 0.05,           # Height of the scalar bar (5% of window)
                        'vertical': False,        # Horizontal bar
                        'fmt': "%.2f"            # Format the numbers
                    })
    # Set camera for 2D top-down view (XY plane)
    plotter.add_text(sim_name1, position='upper_right', font_size=14, color='black')
    if tyre_outline == True:
        plotter.add_points(coords_outline_tyre, color='black', point_size=3, render_points_as_spheres=True)

    if selected_variable in ['Avg Cp', 'Avg Cp0']:
        arg_min = np.argmin(np.array(df1[selected_variable]))
        point = df1.iloc[arg_min,:][['x','y','z']].values
        plotter.add_points(point, color='red', point_size=10, render_points_as_spheres=True)
        
    """
    df_local_var_min = get_local_pressure_min(mesh1.points, df1[selected_variable], search_radius = 0.01)    
    df = df_local_var_min[df_local_var_min['local min'] == 'local min (r = 5cm)']    
    plotter.add_points(df_local_var_min[['x','y','z']].values, color='black', point_size=10, render_points_as_spheres=True)
    """
    if isocontour_threshold != None : 
        contours = mesh1.contour(isosurfaces=[isocontour_threshold], scalars=selected_variable)
        isocontour_pts = contours.points
        if len(isocontour_pts) > 0: #only show if the isocontour exists
            arg_top = np.argmax(isocontour_pts[:,2])
            print("Location of tyre Loss (top): ", isocontour_pts[arg_top,:])
            plotter.add_mesh(contours, color='red', line_width=2)
            #plotter.add_points(isocontour_pts[arg_top,:], color='red', point_size=10, render_points_as_spheres=True)

    """
    # Extract all boundaries
    # Get 2D points
    points_2d = mesh1.points[:, 1:]
    
    # Compute alpha shape
    alpha = 0.0  # Start with 0 (convex hull), increase for tighter fit
    alpha_shape = alphashape.alphashape(points_2d, alpha)  
    
    
    # Get boundary coordinates
    boundary_coords = np.array(alpha_shape.exterior.coords)    
    # Add X coordinate back
    x = mesh1.points[0, 0]  # Get the X value of the slice plane
    boundary_3d = np.column_stack([np.full(len(boundary_coords), x), boundary_coords])
    # Create line polydata
    n = len(boundary_3d) - 1
    lines = np.hstack([[2, i, i+1] for i in range(n)])    
    outer_boundary = pv.PolyData(boundary_3d, lines=lines)    
    plotter.add_mesh(outer_boundary, color='red', line_width=5)
    
    
    # Create a uniform 2D grid in Y-Z plane
    y_min, y_max = points_2d[:, 0].min(), points_2d[:, 0].max()
    z_min, z_max = points_2d[:, 1].min(), points_2d[:, 1].max()
    
    # Create grid
    grid_resolution = 500  # Adjust for finer/coarser grid
    y_grid = np.linspace(y_min, y_max, grid_resolution)
    z_grid = np.linspace(z_min, z_max, grid_resolution)
    yy, zz = np.meshgrid(y_grid, z_grid)
    
    # Flatten grid points
    grid_points = np.column_stack([yy.ravel(), zz.ravel()])
    
    # Build KD-tree for fast nearest neighbor search
    tree = cKDTree(points_2d)
    
    # Find nearest neighbor distance for each grid point
    distances, indices = tree.query(grid_points)
    
    # Threshold to identify hole points
    distance_threshold = 0.01  # Adjust based on your mesh spacing
    hole_mask = distances > distance_threshold
    
    hole_points_2d = grid_points[hole_mask]
    
    print(f"Found {len(hole_points_2d)} grid points in hole region")
    print(f"Distance threshold: {distance_threshold}")
    
    # Visualize
    x_value = mesh1.points[0, 0]
    
    # Convert hole points to 3D
    hole_points_3d = np.column_stack([
        np.full(len(hole_points_2d), x_value),
        hole_points_2d
    ])
    try:
        hole_cloud = pv.PolyData(hole_points_3d)
        plotter.add_mesh(hole_cloud, color='red', point_size=8, render_points_as_spheres=True)
    except:
        pass
    """
    
    
    plotter.subplot(0, 1)
    plotter.add_mesh(mesh2, scalars=selected_variable, clim=clim, cmap=cmap, show_scalar_bar=True, n_colors=21,
                     scalar_bar_args={
                        'title': selected_variable + ' ',
                        'position_x': 0.25,        # Center horizontally
                        'position_y': 0.05,       # Near the bottom
                        'width': 0.5,             # Width of the scalar bar (30% of window)
                        'height': 0.05,           # Height of the scalar bar (5% of window)
                        'vertical': False,        # Horizontal bar
                        'fmt': "%.2f"            # Format the numbers
                    })
    # Set camera for 2D top-down view (XY plane)
    plotter.link_views()
    plotter.add_text(sim_name2, position='upper_right', font_size=14, color='black')
    if tyre_outline == True:
        plotter.add_points(coords_outline_tyre, color='black', point_size=3, render_points_as_spheres=True)
    
    if selected_variable in ['Avg Cp', 'Avg Cp0']:
        arg_min = np.argmin(np.array(df2[selected_variable]))
        point = df2.iloc[arg_min,:][['x','y','z']].values
        plotter.add_points(point, color='blue', point_size=10, render_points_as_spheres=True)
    #plotter.show()
    if isocontour_threshold != None : 
        contours = mesh2.contour(isosurfaces=[isocontour_threshold], scalars=selected_variable)
        isocontour_pts = contours.points
        if len(isocontour_pts) > 0: #only show if the isocontour exists
            arg_top = np.argmax(isocontour_pts[:,2])
            print("Location of tyre Loss (top): ", isocontour_pts[arg_top,:])
            plotter.add_mesh(contours, color='blue', line_width=2)
            #plotter.add_points(isocontour_pts[arg_top,:], color='red', point_size=10, render_points_as_spheres=True)
    
    
    ###############
    ## START DELTAS
    ###############
    if selected_variable == 'Avg Cp':
        clim = [-1,1]
    elif selected_variable == 'Avg Cfx':
        clim = [-0.01,0.01]
    else:
        clim = [-1,1]
        
    levs = range(20)
    assert len(levs) % 2 == 0, 'N levels must be even.'
    cmap = mcolors.LinearSegmentedColormap.from_list(name='red_white_blue', 
                                                 colors =[(0, 0, 1), 
                                                          (1, 1., 1), 
                                                          (1, 0, 0)],
                                                 N=len(levs)-1,
                                                 )

    
    plotter.subplot(0, 2)
    plotter.add_mesh(mesh2.copy(), scalars='delta ' + selected_variable, clim=clim, cmap=cmap, show_scalar_bar=True, n_colors=len(levs)-1,
                     scalar_bar_args={
                        'title': 'delta ' + selected_variable,
                        'position_x': 0.25,        # Center horizontally
                        'position_y': 0.05,       # Near the bottom
                        'width': 0.5,             # Width of the scalar bar (30% of window)
                        'height': 0.05,           # Height of the scalar bar (5% of window)
                        'vertical': False,        # Horizontal bar
                        'fmt': "%.2f",            # Format the numbers
                        'n_labels': 5
                    })
    # Set camera for 2D top-down view (XY plane)
    plotter.add_text('Delta', position='upper_right', font_size=14, color='black')
    if tyre_outline == True:
        plotter.add_points(coords_outline_tyre, color='black', point_size=3, render_points_as_spheres=True)
    if selected_variable in ['Avg Cp', 'Avg Cp0']:
        arg_min = np.argmin(np.array(df1[selected_variable]))
        point = df1.iloc[arg_min,:][['x','y','z']].values
        plotter.add_points(point, color='red', point_size=10, render_points_as_spheres=True)
        
        arg_min = np.argmin(np.array(df2[selected_variable]))
        point = df2.iloc[arg_min,:][['x','y','z']].values
        plotter.add_points(point, color='blue', point_size=10, render_points_as_spheres=True)
    if isocontour_threshold != None : 
        contours = mesh1.contour(isosurfaces=[isocontour_threshold], scalars=selected_variable)
        isocontour_pts = contours.points
        if len(isocontour_pts) > 0: #only show if the isocontour exists
            arg_top = np.argmax(isocontour_pts[:,2])
            print("Location of tyre Loss (top): ", isocontour_pts[arg_top,:])
            plotter.add_mesh(contours, color='red', line_width=2)
            #plotter.add_points(isocontour_pts[arg_top,:], color='red', point_size=10, render_points_as_spheres=True)
        contours = mesh2.contour(isosurfaces=[isocontour_threshold], scalars=selected_variable)
        isocontour_pts = contours.points
        if len(isocontour_pts) > 0: #only show if the isocontour exists
            arg_top = np.argmax(isocontour_pts[:,2])
            print("Location of tyre Loss (top): ", isocontour_pts[arg_top,:])
            plotter.add_mesh(contours, color='blue', line_width=2)
            #plotter.add_points(isocontour_pts[arg_top,:], color='red', point_size=10, render_points_as_spheres=True)
    
    plotter.view_yz() 
    plotter.link_views()
    ###############
    ## END DELTAS
    ###############
    #plotter.set_background('black')
    plotter.show()
    
    

def compare_surfaceVTKs(sim_name1, sim_name2, selected_variable, isocontour_threshold = 0):
    mesh1,df1 = read_surfaceVTK(sim_name = sim_name1, selected_variable = selected_variable)    
    mesh2,df2 = read_surfaceVTK(sim_name = sim_name2, selected_variable = selected_variable)
    
    start_time = time.time()
    print(" Start interpolating")
    interpolated_values = griddata(points = df1[['x','y','z']], values = df1[selected_variable], xi = df2[['x','y','z']], method = 'nearest')
    df2['interpolated ' + selected_variable] = interpolated_values
    df2['delta ' + selected_variable] = df2[selected_variable] - interpolated_values
    mesh2.point_data['delta ' + selected_variable] = df2['delta ' + selected_variable]
    mesh2.point_data['interpolated'] = interpolated_values
    end_time = time.time()
    print("End interpolating : ", end_time - start_time)
    
    print("Start rendering")
    start_time = time.time()
    
    ####################
    ## PLOT SIMULATION 1
    ####################
    
    if selected_variable == 'Avg Cp':
        clim = [-5,1]
        cmap = 'viridis'
    elif selected_variable == 'Avg Cfx':
        clim = [-0.04,0.04]
        cmap = 'coolwarm'
    else:
        clim = [-1,1]
        cmap = 'viridis'
    plotter = pv.Plotter(shape=(1, 3), off_screen=False)
    plotter.subplot(0, 0)
    plotter.add_mesh(mesh1, scalars=selected_variable, clim=clim, cmap=cmap, show_scalar_bar=True, n_colors=21,
                     scalar_bar_args={
                        'title': selected_variable,
                        'position_x': 0.25,        # Center horizontally
                        'position_y': 0.05,       # Near the bottom
                        'width': 0.5,             # Width of the scalar bar (30% of window)
                        'height': 0.05,           # Height of the scalar bar (5% of window)
                        'vertical': False,        # Horizontal bar
                        'fmt': "%.2f"            # Format the numbers
                    })
    # Set camera for 2D top-down view (XY plane)
    plotter.add_text(sim_name1, position='upper_right', font_size=14, color='black')
    if isocontour_threshold != None : 
        contours = mesh1.contour(isosurfaces=[isocontour_threshold], scalars=selected_variable)
        isocontour_pts = contours.points
        if len(isocontour_pts) > 0: #only show if the isocontour exists
            arg_top = np.argmax(isocontour_pts[:,2])
            print("Location of tyre Loss (top): ", isocontour_pts[arg_top,:])
            plotter.add_mesh(contours, color='red', line_width=2)
            #plotter.add_points(isocontour_pts[arg_top,:], color='red', point_size=10, render_points_as_spheres=True)
    
    
    ####################
    ## PLOT SIMULATION 2
    ####################
    
    plotter.subplot(0, 1)    
    plotter.add_mesh(mesh2, scalars=selected_variable, clim=clim, cmap=cmap, show_scalar_bar=True, n_colors=21,
                     scalar_bar_args={
                        'title': selected_variable + ' ',
                        'position_x': 0.25,        # Center horizontally
                        'position_y': 0.05,       # Near the bottom
                        'width': 0.5,             # Width of the scalar bar (30% of window)
                        'height': 0.05,           # Height of the scalar bar (5% of window)
                        'vertical': False,        # Horizontal bar
                        'fmt': "%.2f"            # Format the numbers
                    })
    
    # Set camera for 2D top-down view (XY plane)
    plotter.add_text(sim_name2, position='upper_right', font_size=14, color='black')
    plotter.link_views()
    if isocontour_threshold != None : 
        contours = mesh2.contour(isosurfaces=[isocontour_threshold], scalars=selected_variable)
        isocontour_pts = contours.points
        if len(isocontour_pts) > 0: #only show if the isocontour exists
            arg_top = np.argmax(isocontour_pts[:,2])
            print("Location of tyre Loss (top): ", isocontour_pts[arg_top,:])
            plotter.add_mesh(contours, color='blue', line_width=2)
            #plotter.add_points(isocontour_pts[arg_top,:], color='red', point_size=10, render_points_as_spheres=True)
    
    
    ####################
    ## PLOT DELTAS
    ####################
    
    if selected_variable == 'Avg Cp':
        clim = [-1,1]
    elif selected_variable == 'Avg Cfx':
        clim = [-0.01,0.01]
    else:
        clim = [-1,1]
    plotter.subplot(0, 2)
    plotter.add_mesh(mesh2.copy(), scalars='delta ' + selected_variable, clim=clim, cmap='coolwarm', show_scalar_bar=True, n_colors=21,
                     scalar_bar_args={
                        'title': 'delta ' + selected_variable,
                        'position_x': 0.25,        # Center horizontally
                        'position_y': 0.05,       # Near the bottom
                        'width': 0.5,             # Width of the scalar bar (30% of window)
                        'height': 0.05,           # Height of the scalar bar (5% of window)
                        'vertical': False,        # Horizontal bar
                        'fmt': "%.2f"            # Format the numbers
                    })
    # Set camera for 2D top-down view (XY plane)
    plotter.add_text('Delta', position='upper_right', font_size=14, color='black')
    if isocontour_threshold != None : 
        contours = mesh1.contour(isosurfaces=[isocontour_threshold], scalars=selected_variable)
        isocontour_pts = contours.points
        if len(isocontour_pts) > 0: #only show if the isocontour exists
            arg_top = np.argmax(isocontour_pts[:,2])
            print("Location of tyre Loss (top): ", isocontour_pts[arg_top,:])
            plotter.add_mesh(contours, color='red', line_width=2)
            #plotter.add_points(isocontour_pts[arg_top,:], color='red', point_size=10, render_points_as_spheres=True)
        contours = mesh2.contour(isosurfaces=[isocontour_threshold], scalars=selected_variable)
        isocontour_pts = contours.points
        if len(isocontour_pts) > 0: #only show if the isocontour exists
            arg_top = np.argmax(isocontour_pts[:,2])
            print("Location of tyre Loss (top): ", isocontour_pts[arg_top,:])
            plotter.add_mesh(contours, color='blue', line_width=2)
            #plotter.add_points(isocontour_pts[arg_top,:], color='red', point_size=10, render_points_as_spheres=True)
    
    
    plotter.view_yz() 
    plotter.link_views()
    end_time = time.time()
    print("End rendering : ", end_time - start_time)
    plotter.show()
    
    return mesh1, mesh2
    
    






list_sim_name = ['charles_mesh1A_3','charles_mesh1B_2','charles_mesh2A_1','charles_mesh4A_3_hightolerance', 'charles_mesh3A_2_hightolerance']

plt.close('all')
###########################
## START FORCES
###########################
pid         = 'ifw_total'
forces_type = "Drag Coeff."

"""
fig1,fig2,fig3,fig4,_,_,_,_ = plot_forces(list_sim_name, pid, forces_type)


fig1.figure 
fig2.figure 
fig3.figure 
fig4.figure 
"""

###########################
## END FORCES
###########################


###########################
## START SURFACE SLICES
###########################
slice_name  = 'Y = -711.8mm'
variable    = 'Avg Cp'


#fig5,axs5 = plot_surface_slice(list_sim_name, slice_name, variable)
#fig5.figure 


###########################
## END SURFACE SLICES
###########################

###########################
## X NORMAL SLICES
###########################
xslice = 'PIV2'
#xslice = 'X_400mm'
selected_variable = 'Avg Cp'
isocontour_threshold = 0


sim_name = list_sim_name[0]
 

#msh1,msh2 = compare_surfaceVTKs(list_sim_name[0], list_sim_name[1], selected_variable = 'Avg Cp')
#msh1,msh2 = compare_surfaceVTKs('Nektar++', list_sim_name[0], selected_variable = 'Avg Cfx')
msh1,msh2 = compare_surfaceVTKs('Nektar++', 'charles_mesh4A_3_hightolerance', selected_variable = 'Avg Cfx', isocontour_threshold = -0)

#compare_xliceVTKs('Fidelity_DES_Mesh1', 'Fidelity_DES_Mesh4', xslice = xslice , selected_variable = selected_variable, isocontour_threshold = -0)


#compare_xliceVTKs(list_sim_name[0], 'charles_mesh4A_3_hightolerance', xslice = xslice , selected_variable = selected_variable, isocontour_threshold = -0.0)

#compare_xliceVTKs('Nektar++', 'charles_mesh4A_3_hightolerance', xslice = xslice , selected_variable = selected_variable, isocontour_threshold = -1, tyre_outline = True)

#compare_xliceVTKs('charles_mesh2A_1', 'charles_mesh4A_3_hightolerance', xslice = xslice , selected_variable = selected_variable, isocontour_threshold = -2, tyre_outline = True)

#compare_xliceVTKs('charles_mesh2A_1', 'charles_mesh2C', xslice = xslice , selected_variable = selected_variable, isocontour_threshold = -2, tyre_outline = True)


"""
plotter = pv.Plotter(shape=(1, 1), off_screen=False)
plotter.subplot(0, 0)
plotter.add_mesh(msh1, scalars='Avg Cp', clim=[-4,1], cmap='viridis', show_scalar_bar=True, n_colors=21,
                 scalar_bar_args={
                    'title': selected_variable,
                    'position_x': 0.25,        # Center horizontally
                    'position_y': 0.05,       # Near the bottom
                    'width': 0.5,             # Width of the scalar bar (30% of window)
                    'height': 0.05,           # Height of the scalar bar (5% of window)
                    'vertical': False,        # Horizontal bar
                    'fmt': "%.2f"            # Format the numbers
                })
plotter.show()
"""