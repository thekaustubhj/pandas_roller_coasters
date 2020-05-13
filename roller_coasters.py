import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# load rankings data here:

wood = pd.read_csv('Golden_Ticket_Award_Winners_Wood.csv')
steel = pd.read_csv('Golden_Ticket_Award_Winners_Steel.csv')

#Inspecting the shape and columns of the dataframe
print(wood.head())
print(steel.head())
#print(wood.info())
#print(wood.columns)
#print(wood['Park'].value_counts())

# write function to plot rankings over time for 1 roller coaster here:

def rank_extractor(name, df) :
  filtered_df = df[df['Name'] == name].reset_index(drop = True)
  years = filtered_df['Year of Rank'].values.tolist()
  ranks = filtered_df['Rank'].values.tolist()

  time_series = [2013, 2014, 2015, 2016, 2017, 2018]
  processed_ranks = []
  for year in time_series :
    if year in years :
      index = years.index(year)
      temp = ranks[index]
      processed_ranks.append(temp)
    else :
      processed_ranks.append(np.nan)
  return time_series, processed_ranks, name

def single_plotter(years, ranks, name) :
  plt.close('all')
  plt.figure(figsize = (10,7))
  ax = plt.subplot(1,1,1)
  plt.plot(years, ranks, color = 'orange', linestyle = '-', marker = 'o')
  title = 'Ranking over time for ' + name
  plt.title(title)
  plt.xlabel('Years')
  plt.ylabel('Ranking')
  ax.set_yticks(range(11))
  plt.show()
  return ranks

def single_rank_trender(name, df) :
  plot_years, plot_ranks, plot_name = rank_extractor(name, df)
  single_plotter(plot_years, plot_ranks, plot_name)
  return name

single_rank_trender('Thunderhead', wood)


# write function to plot rankings over time for 2 roller coasters here:

def double_plotter(years, ranks_one, name_one, ranks_two, name_two) :
  plt.close('all')
  plt.figure(figsize = (10,7))
  ax = plt.subplot(1,1,1)
  plt.plot(years, ranks_one, color = 'orange', linestyle = '-', marker = 'o')
  plt.plot(years, ranks_two, color = 'green', linestyle = '-', marker = 'o')
  title = 'Ranking over time for ' + name_one + ' & ' + name_two
  plt.title(title)
  plt.xlabel('Years')
  plt.ylabel('Ranking')
  plt.legend([name_one, name_two])
  ax.set_yticks(range(11))
  plt.show()
  return years

def double_rank_trender(name_one, name_two, df) :
  plot_years, plot_ranks_one, plot_name_one = rank_extractor(name_one, df)
  plot_years, plot_ranks_two, plot_name_two = rank_extractor(name_two, df)
  double_plotter(plot_years, plot_ranks_one, plot_name_one, plot_ranks_two, plot_name_two)
  return name_one

double_rank_trender('El Toro', 'Boulder Dash', wood)

# write function to plot top n rankings over time here:


def top_plotter(n, df) :

  #find all roller costers that have had a rank of n or lower in the dataset.
  top_ranks_only = df[df['Rank']<=n]
  #Create a unique list of those names
  unique_names = top_ranks_only['Name'].unique().tolist()
  #Prep an empty figure and axis for plotting
  plt.close('all')
  plt.figure(figsize = (10,7))
  ax = plt.subplot(1,1,1)
  #Loop through rank extractor for each of those names
  for name in unique_names :
     plot_years, plot_ranks, plot_name = rank_extractor(name, df)
     plt.plot(plot_years, plot_ranks, linestyle = '-', marker = 'o')
  plt.xlabel('Years')
  plt.ylabel('Ranking')
  plt.title('Rnking Trend for Top Roller coasters')
  plt.legend(unique_names)
  ax.set_yticks(range(11))
  plt.show()
  return n

top_plotter(5, wood)


# load roller coaster data here:

stats = pd.read_csv('roller_coasters.csv')
print(stats.head())
print(stats.info())
print(stats['seating_type'].value_counts())
print(stats['status'].value_counts())
print(stats['material_type'].value_counts())


# write function to plot histogram of column values here:
def histogram_plotter(column) :
  plot_data = column.values.tolist()
  max_value = max(plot_data)
  min_value = min(plot_data)
  plt.close('all')
  plt.figure(figsize=(10,7))
  ax = plt.subplot(1,1,1)
  plt.hist(column, bins = 10, range = (min_value, max_value), color = 'blue', alpha = 0.5, normed = True)
  title = 'Distribution of data for '
  plt.title(title)
  plt.xlabel('Value buckets')
  plt.ylabel('Occurances')
  plt.show()
  return 'Plotting Complete'

histogram_plotter(stats['speed'])

#Truying out the scatter_matrix function in Pandas
plt.close('all')
from pandas.plotting import scatter_matrix
scatter_matrix(stats, figsize=(10, 10))
plt.show()


# write function to plot inversions by coaster at a park here:

def inversion_finder(df, park) :
  filtered_df = stats[stats['park'] == park]
  names = filtered_df['name'].values.tolist()
  inversions = filtered_df['num_inversions'].values.tolist()
  plt.close('all')
  plt.figure(figsize = (10,7))
  ax = plt.subplot(1,1,1)
  plt.bar(range(len(names)), inversions, alpha = 0.5, color = 'red')
  title = 'Number of Inversions at ' + park
  plt.title(title)
  plt.xlabel('Roller Coasters')
  plt.ylabel('Number of Inversions')
  ax.set_xticks(range(len(names)))
  ax.set_xticklabels(names, rotation = 90)
  plt.show()
  return 'Plotting Complete'

inversion_finder(stats, 'Parc Asterix')

# write function to plot pie chart of operating status here:

def status_finder(df) :
  summarised = df.groupby('status').name.count().reset_index()
  summarised.rename(columns = {'name' : 'count'}, inplace = True)
  values = summarised['count'].values.tolist()
  status = summarised['status'].values.tolist()
  plt.figure(figsize = (10,7))
  ax = plt.subplot(1,1,1)
  plt.pie(values, labels = status, autopct = '%d%%')
  plt.axis('equal')
  plt.title('Status of Roller Coasters in DataSet')
  plt.show()
  return 'Plotting Complete'

status_finder(stats)


# write function to create scatter plot of any two numeric columns here:
#Already created Scatter using the Scatter Matrix above

#Making a plot using Seaborn

import seaborn as sns

plt.close('all')
plt.figure(figsize = (12,10))
sns.set_context('notebook')
sns.set_style('whitegrid')
sns.set_palette('pastel')
ax = sns.barplot(data = stats, x = 'seating_type', y = 'speed', estimator = np.mean)
ax.set_xticklabels(ax.get_xticklabels(), rotation = 90)
plt.show()


#Merge Table with Stats with Ranking Table

steel['type'] = 'Steel'
wood['type'] = 'Wood'
concatenated = pd.concat([steel, wood])
merged = pd.merge(concatenated, stats, how = 'left', left_on = ['Name', 'Park'], right_on = ['name', 'park'])
print(concatenated.info())
print(merged.info())
print(merged.head())
