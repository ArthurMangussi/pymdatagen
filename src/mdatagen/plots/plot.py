# -*- coding: utf-8 -*

#  =============================================================================
# Aeronautics Institute of Technologies (ITA) - Brazil
# University of Coimbra (UC) - Portugal
# Arthur Dantas Mangussi - mangussiarthur@gmail.com
# =============================================================================

import matplotlib.pyplot as plt
import missingno as msno
import pandas as pd
from typing import Tuple,Optional
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import random as rng

class PlotMissingData:
  """
  A class to visualize missing data in a DataFrame using different types of plots.

  Attributes
  ----------
  data_missing : pd.DataFrame
      The DataFrame containing the data with potential missing values.
  original : pd.DataFrame
      The complete DataFrame.
  palette : str
      The string to indicate the color palette of the plot.
  custom_color : tuple of float or None
      The custom color in RGB format, e.g., (1, 1, 1), or None if not specified.

  Methods
  -------
  visualize_miss(visualization_type: str = "normal", save: bool = True, path_save_fig: str = None)
      Visualizes the missing data using the specified type of plot and optionally saves the plot.
  """
  def __init__(self,
               data_missing:pd.DataFrame,
               data_original:pd.DataFrame,
               palette:str="gray",
               custom_color:Optional[Tuple[float, float, float]]=None) -> None:

    if not isinstance(data_missing, pd.DataFrame):
            raise TypeError("data must be a pandas DataFrame")

    self.data_missing = data_missing
    self.original = data_original
    self.figsize = (12,12)
    self.fontsize = 10
    self.plot_miss = None
    self.palette = self._set_palette(palette)
    self.custom_color = custom_color

  # ------------------------------------------------------------------------
  def _set_palette(self,palette):
    """
    Sets the color attribute based on the specified palette.

    This method assigns a color tuple to `self.color` based on the provided 
    `palette` name. If `palette` is set to `"customize"`, the method uses 
    the value of `self.custom_color`.

    Parameters
    ----------
    palette : str
        The name of the color palette to use. Must be one of the following: 
        `"gray"`, `"blue"`, `"orange"`, `"green"`, `"red"`, `"customize"`.

    Raises
    ------
    ValueError
        If `palette` is `"customize"` but `self.custom_color` is `None` or 
        not a valid tuple of three numeric values.
        If `palette` is not one of the predefined values.

    Example
    -------
    If `palette` is `"blue"`, `self.color` will be set to `(0, 0, 0.5)`.

    Note
    ----
    - If `palette` is `"customize"`, `self.custom_color` should be a tuple with 
    three elements, each being a float.
    """
    match palette:
      case "gray":
        self.color = (0.25,0.25,0.25)
      case "blue":
        self.color = (0,0,0.5)
      case "orange":
        self.color = (1,0.55,0)
      case "green":
        self.color = (0,0.8,0)
      case "red":
        self.color = (0.8,0,0)
      case "customize":
        if self.custom_color is None:
          raise ValueError("You should set the RGB color for 'customize'.")
        if len(self.custom_color) != 3 or not all(isinstance(c, float) for c in self.custom_color):
            raise ValueError("custom_color must be a tuple with exactly 3 numeric values.")
        self.color = self.custom_color
      case _:
          raise ValueError(f"Unknown palette option: {self.palette}")

    return palette

  # ------------------------------------------------------------------------
  def _histogram(self, col_missing:str, num_bins:int=10):
    """
    Plots a histogram comparing the distribution of a specified column's 
    original data with data containing missing values.

    Parameters
    ----------
    col_missing : str
        The name of the column to plot.
    num_bins : int, optional
        The number of bins to use in the histogram (default is 10).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the histogram.
    ax : matplotlib.axes.Axes
        The matplotlib axes object containing the histogram.
    
    """
    data_combined = np.concatenate(
        [self.original[col_missing], self.data_missing[col_missing].dropna(axis=0)]
    )

    bins = np.histogram_bin_edges(data_combined, bins=num_bins)

    fig, ax = plt.subplots(figsize=self.figsize)
    sns.histplot(self.original[col_missing],
                 alpha=0.9,
                 bins=bins,
                 label="Original data",
                 color=self.color)
    sns.histplot(self.data_missing.dropna(axis=0)[col_missing],
                 alpha=0.6,
                 bins=bins,
                 label="Dropping missing values",
                 color="red")
    ax.legend()

    return fig,ax


  # ------------------------------------------------------------------------
  def _boxplot(self,col_missing:str, report:Optional[bool]=False):
    """
    Plots a boxplot comparing the distribution of a specified column's 
    original data with data containing missing values.

    Parameters
    ----------
    col_missing : str
        The name of the column to plot.
    report : bool, optional
        If True, prints the statistical summary of the data (default is False).

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object containing the boxplot.
    ax : matplotlib.axes.Axes
        The matplotlib axes object containing the boxplot.

    """

    fig, ax = plt.subplots(figsize=self.figsize)
    df_boxplot = pd.DataFrame(columns=["Original data","Dropping missing values"])
    df_boxplot["Original data"] = self.original[col_missing].copy()
    df_boxplot["Dropping missing values"] = self.data_missing.dropna(axis=0)[col_missing].copy()
    df_boxplot.boxplot(color=self.palette)
    ax.set_xlabel(col_missing)

    if report:
      print(df_boxplot.describe())

    return fig,ax


# ------------------------------------------------------------------------
  def _geom_miss_plot(self, cols:Tuple[str, str], threshold:Tuple[float,float,float]=(0.5, 0.8, 0.9)):
      """
      Creates a scatter plot to visualize the relationship between a column 
      with missing values and another column of interest.

      This method generates a scatter plot where the x-axis represents the column 
      with missing values and the y-axis represents another column of interest,
      and vice-versa.

      Parameters
      ----------
      cols : Tuple[str, str]
        The data columns that will be the x-axis and y-axis, respectively.
      threshold : Tuple[float,float,float], optional
        Tuple to control the limits of the jitter of the points.

      Returns
      -------
      fig : matplotlib.figure.Figure
          The matplotlib figure object containing the scatter plot.
      ax : matplotlib.axes.Axes
          The matplotlib axes object containing the scatter plot.

      Reference
      -----
      - https://tmb.njtierney.com/exploring-causes-of-missingness

      """
      
      col_x, col_y = cols
      df_aux = pd.DataFrame(columns=[col_x,col_y])
      df_aux[[col_x, col_y]] = self.data_missing[[col_x, col_y]]
      df_aux['Indicator'] = df_aux[[col_x, col_y]].isna().any(axis=1).map({True:"Missing", False: "Not Missing"})

      min_values = df_aux[[col_x, col_y]].min()
      for col in [col_x, col_y]:
        if df_aux[col].isna().any():
          min_val = min_values[col]
          offset_range = (threshold[0] * min_val, threshold[1] * min_val)
          df_aux[col] = df_aux[col].apply(lambda value: rng.uniform(offset_range[0], offset_range[1]) if pd.isna(value) else value)

      fig,ax = plt.subplots()
      sns.scatterplot(data=df_aux,
                  x=col_x,
                  y=col_y,
                  hue="Indicator",
                  palette={"Missing": "red",
                          "Not Missing": self.color}
                      )   
      plt.xlabel(col_x)
      plt.ylabel(col_y)
      plt.legend()
      plt.axvline(x=threshold[2] * min_values[col_x], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
      plt.axhline(y=threshold[2] * min_values[col_y], color='gray', linestyle='--', linewidth=1.5, alpha=0.5)

      return fig, ax

  # ------------------------------------------------------------------------
  def visualize_miss(self,
                     visualization_type:str="normal",
                     save:bool=True,
                     path_save_fig:str=None,
                     col_missing:str=None,
                     num_bins:int=10,
                     report:bool=False,
                     threshold=(0.5, 0.8, 0.9),
                     cols:Tuple=None):
    """
    Visualizes the missing data using the specified type of plot and optionally saves the plot.

    Parameters:
    ----------
    visualization_type : str, optional
        The type of visualization to generate. Options are "normal", "bar", "dendrogram",
        "heatmap", "histogram", "boxplot", "scatterplot". Default is "normal".
    save : bool, optional
        Whether to save the plot to a file. Default is True.
    path_save_fig : str, optional
        The path and filename to save the plot. If None, the plot will be saved as 
        <visualization_type>.png. Default is None.
    col_missing : str
        The name of the column to plot.
    num_bins : int, optional
        The number of bins to use in the histogram (default is 10). 
    report : bool, optional
        If True, prints the statistical summary of the data (default is False).
    threshold : Tuple[float,float,float], optional
        Tuple to control the limits of the jitter of the points.
    cols : Tuple[str, str]
      The data columns that will be the x-axis and y-axis, respectively.

    Returns:
    -------
    plot_miss : plt.Axes
        The plot generated by the visualization method.

    References:
    ----------
    missingno documentation: https://github.com/ResidentMario/missingno
    """
    if path_save_fig is None:
      path_save = f"{visualization_type}.png"
    else:
      path_save = path_save_fig

    match visualization_type:
      case "normal":
        self.plot_miss = msno.matrix(self.data_missing,
                    figsize=self.figsize,
                    fontsize=self.fontsize,
                    color = self.color)
        if save:
          plt.savefig(path_save)

      case "bar":
        self.plot_miss = msno.bar(self.data_missing,
                figsize=self.figsize,
                fontsize=self.fontsize,
                color = self.color)
        if save:
          plt.savefig(path_save)

      case "dendrogram":
        self.plot_miss = msno.dendrogram(self.data_missing,
                       figsize=self.figsize,
                       fontsize=self.fontsize)

        if save:
          plt.savefig(path_save)
      case "heatmap":
        self.plot_miss = msno.heatmap(self.data_missing,
                     figsize=self.figsize,
                     fontsize=self.fontsize)
        if save:
          plt.savefig(path_save)

      case "histogram":
        self.plot_miss,ax = self._histogram(col_missing,
                                            num_bins)
        if save:
          self.plot_miss.savefig(path_save)

      case "boxplot":
        self.plot_miss,ax = self._boxplot(col_missing,
                                       report)
        if save:
          self.plot_miss.savefig(path_save)

      case "scatterplot":
        self.plot_miss,ax = self._geom_miss_plot(cols,
                                                 threshold)
        if save:
          self.plot_miss.savefig(path_save)