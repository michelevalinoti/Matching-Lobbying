#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 27 16:28:28 2023

@author: michelev
"""

import networkx as nx
import matplotlib.pyplot as plt

import pandas as pd
import numpy as np 

from jarowinkler import jarowinkler_similarity

import os

#%%

machine = '/Users/michelev/Dropbox/lobbying'
table_subfolder = os.path.join(machine, 'dataframes')

path_table = os.path.join(table_subfolder, 'reports_merged.csv')
df_all = pd.read_csv(path_table, index_col = 0)
df_all = df_all.loc[(pd.isna(df_all.people_id) == False) & (pd.isna(df_all.bill_id) == False),:]
#%%


# Explode the list into multiple rows while keeping 'ColumnB' the same

pairs =  df_all.loc[:,['beneficial_client', 'principal_lobbyist']].drop_duplicates()
#%%
def compute_distance(array1, array2):
    
    return np.vectorize(jarowinkler_similarity)(array1[:, np.newaxis], array2)

pairs['sim'] =pairs.apply(lambda row: jarowinkler_similarity(row['beneficial_client'], row['principal_lobbyist']), axis=1)
pairs['employed'] = pairs.sim < 0.85

df_all = df_all.merge(pairs, on = ['beneficial_client', 'principal_lobbyist'], how = 'left')

#%%


df_all = df_all.loc[(pd.isna(df_all.people_id) == False) & (pd.isna(df_all.bill_id) == False),:]

#%%

count = 200
df_employed = df_all.loc[df_all.employed == True,:]
df_retained = df_all.loc[df_all.employed == False,:]

df_employed = df_employed.groupby(['beneficial_client', 'principal_lobbyist', 'people_id']).filter(lambda group: len(group) >=count)
df_retained = df_retained.groupby(['beneficial_client', 'principal_lobbyist', 'people_id']).filter(lambda group: len(group) >=count*2)

df = pd.concat((df_employed, df_retained), axis=0)
#%%

df = df.loc[(pd.isna(df.people_id) == False) & (pd.isna(df.bill_id) == False),:]
#df = df.loc[df.year == 2021]

G = nx.Graph()

nodes_clients = pd.unique(df.beneficial_client)
nodes_lobbyists = pd.unique(df_employed.principal_lobbyist)
nodes_public_officials = pd.unique(df.people_id)

G.add_nodes_from(nodes_clients, bipartite=0)
G.add_nodes_from(nodes_lobbyists, bipartite=1)
G.add_nodes_from(nodes_public_officials, bipartite=2)

group_clients_lobbyists = df_employed.groupby(['beneficial_client', 'principal_lobbyist'])
edges_clients_lobbyists_bills_counts = group_clients_lobbyists.bill_id.nunique()
edges_clients_lobbyists_reports_counts = group_clients_lobbyists.parties_lobbied_id.count()
edges_clients_lobbyists_avg_reports_bill = edges_clients_lobbyists_reports_counts/edges_clients_lobbyists_bills_counts

edges_clients_lobbyists_bills_counts = [tuple(row) for row in pd.DataFrame(edges_clients_lobbyists_bills_counts).reset_index().itertuples(index=False)]
edges_clients_lobbyists_reports_counts = [tuple(row) for row in pd.DataFrame(edges_clients_lobbyists_reports_counts).reset_index().itertuples(index=False)]
edges_clients_lobbyists_avg_reports_bill = [tuple(row) for row in pd.DataFrame(edges_clients_lobbyists_avg_reports_bill).reset_index().itertuples(index=False)]

group_lobbyists_officials = df_employed.groupby(['principal_lobbyist', 'people_id'])
edges_lobbyists_officials_bills_counts = group_lobbyists_officials.bill_id.nunique()
edges_lobbyists_officials_reports_counts = group_lobbyists_officials.parties_lobbied_id.count()
edges_lobbyists_officials_avg_reports_bill = edges_lobbyists_officials_reports_counts/edges_lobbyists_officials_bills_counts

edges_lobbyists_officials_bills_counts = [tuple(row) for row in pd.DataFrame(edges_lobbyists_officials_bills_counts).reset_index().itertuples(index=False)]
edges_lobbyists_officials_reports_counts = [tuple(row) for row in pd.DataFrame(edges_lobbyists_officials_reports_counts).reset_index().itertuples(index=False)]
edges_lobbyists_officials_avg_reports_bill = [tuple(row) for row in pd.DataFrame(edges_lobbyists_officials_avg_reports_bill).reset_index().itertuples(index=False)]

group_clients_officials = df_retained.groupby(['beneficial_client', 'people_id'])
edges_clients_officials_bills_counts = group_clients_officials.bill_id.nunique()
edges_clients_officials_reports_counts = group_clients_officials.parties_lobbied_id.count()
edges_clients_officials_avg_reports_bill = edges_clients_officials_reports_counts/edges_clients_officials_bills_counts

edges_clients_officials_bills_counts = [tuple(row) for row in pd.DataFrame(edges_clients_officials_bills_counts).reset_index().itertuples(index=False)]
edges_clients_officials_reports_counts = [tuple(row) for row in pd.DataFrame(edges_clients_officials_reports_counts).reset_index().itertuples(index=False)]
edges_clients_officials_avg_reports_bill = [tuple(row) for row in pd.DataFrame(edges_clients_officials_avg_reports_bill).reset_index().itertuples(index=False)]


G.add_weighted_edges_from(edges_clients_lobbyists_bills_counts)
G.add_weighted_edges_from(edges_lobbyists_officials_bills_counts)
G.add_weighted_edges_from(edges_clients_officials_bills_counts)



#%%

# Set the positions for layout
pos = {}
pos.update((node, (1, index * 1)) for index, node in enumerate(nodes_clients))
pos.update((node, (2, index * 10 - 100)) for index, node in enumerate(nodes_lobbyists))
pos.update((node, (3, index * 1)) for index, node in enumerate(nodes_public_officials))

nx.draw_networkx_nodes(G, pos, node_size=1, nodelist=nodes_clients, node_color='r', label="Clients")
nx.draw_networkx_nodes(G, pos, node_size=1, nodelist=nodes_lobbyists, node_color='g', label="Lobbyists")
nx.draw_networkx_nodes(G, pos, node_size=1, nodelist=nodes_public_officials, node_color='b', label="Lawmakers")

nx.draw_networkx_edges(G, pos, edgelist=edges_clients_lobbyists_reports_counts, edge_color='grey')
nx.draw_networkx_edges(G, pos, edgelist=edges_lobbyists_officials_reports_counts, edge_color='grey')
nx.draw_networkx_edges(G, pos, edgelist=edges_clients_officials_reports_counts, edge_color='grey')

plt.legend()

# Draw the tripartite graph
#nx.draw(G, pos, with_labels=False, node_size=10, node_color=['red', 'blue', 'green'])
plt.savefig("network_plot.png", dpi=300, bbox_inches='tight')

# Show the plot
plt.show()

#%%




plt.legend()
plt.show()


#%%

#nx.draw_networkx_labels(G, pos)

plt.legend()
plt.show()


G.add_edges_from(edges_group1_group2)
G.add_edges_from(edges_group1_group3)