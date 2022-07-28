import networkx as nx
import dgl
import numpy as np
import torch
from common.predicate import PRED_DICT
from itertools import product


class KnowledgeGraph(object):
  def __init__(self, facts, predicates, dataset):
    # gen_dgl_graph(facts, predicates, dataset)
    self.dataset = dataset
    self.nxgraph, self.edge_type2idx, \
        self.ent2idx, self.idx2ent, self.rel2idx, self.idx2rel, \
        self.node2idx, self.idx2node = gen_graph(facts, predicates, dataset)
    x, y, v = zip(*sorted(self.nxgraph.edges(data=True), key=lambda t: t[:2]))
    # x, y = self.graph.edges()
    self.edge_types =  [d['edge_type'] for d in v]
    print(len(self.nxgraph.nodes()),len(self.nxgraph.edges()))
    self.graph = gen_dgl_graph(facts, predicates, dataset)
    # graph_dict = {}
    # for i,edge in enumerate(v):
    #   if ('fact',edge['edge_type'],'entity') not in graph_dict:
    #     graph_dict[('fact',edge['edge_type'],'entity')] = []
    #   graph_dict[('fact',edge['edge_type'],'entity')].append((x[i],y[i]))
    # for n, nbrdict in self.graph.adj.items():
    #   for nbr, dd in nbrdict.items():
    #     if ('fact',dd['edge_type'],'entity') not in graph_dict:
    #       graph_dict[('fact',dd['edge_type'],'entity')] = []
    #     graph_dict[('fact',dd['edge_type'],'entity')].append((n,nbr))
    # print("making dgl graph")
    # self.graph = dgl.heterograph(graph_dict)
    print(self.graph.etypes)
    print(self.graph.ntypes)
    # self.graph2 = dgl.bipartite_from_networkx(self.graph)
    # print(self.graph2.etypes)
    self.num_ents = len(self.ent2idx)
    self.num_rels = len(self.rel2idx)
    
    self.num_nodes = self.graph.num_nodes() # len(self.graph.nodes()) # 
    self.num_edges = len(self.nxgraph.edges()) # self.graph.num_edges() # 
    print(self.num_nodes, self.num_edges)
    # print(self.edge_types)
    self.edge_pairs = np.ndarray(shape=(self.num_edges, 2), dtype=np.long)
    self.edge_pairs[:, 0] = x # self.graph.srcnodes('fact')
    self.edge_pairs[:, 1] = y # self.graph.dstnodes('entity')
    
    
    # self.idx2edge = dict()
    # idx = 0
    # for x, y in self.edge_pairs:
    #   self.idx2edge[idx] = (self.idx2node[x], self.idx2node[y])
    #   idx += 1
    #   self.idx2edge[idx] = (self.idx2node[y], self.idx2node[x])
    #   idx += 1


def gen_index(facts, predicates, dataset):
  rel2idx = dict()
  idx_rel = 0
  # index sorted relationships in PRED_DICT
  for rel in sorted(predicates.keys()):
    if rel not in rel2idx:
      rel2idx[rel] = idx_rel
      idx_rel += 1
  idx2rel = dict(zip(rel2idx.values(), rel2idx.keys()))
  
  ent2idx = dict()
  idx_ent = 0
  # index sorted entities
  for type_name in sorted(dataset.const_sort_dict.keys()):
    for const in dataset.const_sort_dict[type_name]:
      ent2idx[const] = idx_ent
      idx_ent += 1
  idx2ent = dict(zip(ent2idx.values(), ent2idx.keys()))
  # index node from entities and facts
  node2idx = ent2idx.copy()
  idx_node = len(node2idx)
  for rel in sorted(facts.keys()):
    for fact in sorted(list(facts[rel])):
      val, args = fact
      if (rel, args) not in node2idx:
        node2idx[(rel, args)] = idx_node # (smoke, (A,B))
        idx_node += 1
  idx2node = dict(zip(node2idx.values(), node2idx.keys()))
  facts_idx = [*range(len(idx2ent), idx_node)] # index for facts
  return ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node, facts_idx


def gen_edge_type():
  edge_type2idx = dict()
  num_args_set = set() # # of arguments in predicate relation
  for rel in PRED_DICT:
    num_args = PRED_DICT[rel].num_args # 2 or less
    num_args_set.add(num_args)
  idx = 0
  for num_args in sorted(list(num_args_set)):
    # encode positions in binary
    for pos_code in product(['0', '1'], repeat=num_args):
      if '1' in pos_code:
        edge_type2idx[(0, ''.join(pos_code))] = idx
        idx += 1
        edge_type2idx[(1, ''.join(pos_code))] = idx
        idx += 1
  return edge_type2idx

  
def gen_graph(facts, predicates, dataset):
  """
      generate directed knowledge graph, where each edge is from subject to object
  :param facts:
      dictionary of facts
  :param predicates:
      dictionary of predicates
  :param dataset:
      dataset object
  :return:
      graph object, entity to index, index to entity, relation to index, index to relation
  """
  
  # build bipartite graph (constant nodes and hyper predicate nodes)
  g = nx.Graph()
  ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node, facts_idx = gen_index(facts, predicates, dataset)

  # for 2 arguments, this should be 01 and 10
  edge_type2idx = gen_edge_type()
  ents = [*range(0,len(idx2ent))]
  g.add_nodes_from(ents, bipartite=0)
  g.add_nodes_from(facts_idx, bipartite=1)
  # add all the nodes by index
  # for node_idx in idx2node:
  #   g.add_node(node_idx)
  
  for rel in facts.keys():
    for fact in facts[rel]:
      val, args = fact
      fact_node_idx = node2idx[(rel, args)]
      for arg in args:
        pos_code = ''.join(['%d' % (arg == v) for v in args]) #10 or 01
        g.add_edge(fact_node_idx, node2idx[arg],
                   edge_type=edge_type2idx[(val, pos_code)])
  return g, edge_type2idx, ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node


def gen_dgl_graph(facts, predicates, dataset):
  """
      generate directed knowledge graph, where each edge is from subject to object
  :param facts:
      dictionary of facts
  :param predicates:
      dictionary of predicates
  :param dataset:
      dataset object
  :return:
      graph object, entity to index, index to entity, relation to index, index to relation
  """
  
  # build bipartite graph (constant nodes and hyper predicate nodes)
  ent2idx, idx2ent, rel2idx, idx2rel, node2idx, idx2node, facts_idx = gen_index(facts, predicates, dataset)
  edge_type2idx = gen_edge_type()
  graph_dict = {}
  for rel in facts.keys():
    for fact in facts[rel]:
      val, args = fact
      fact_node_idx = node2idx[(rel, args)]
      for arg in args:
        pos_code = ''.join(['%d' % (arg == v) for v in args]) #10 or 01
        if ('fact',edge_type2idx[(val, pos_code)],'entity') not in graph_dict:
          graph_dict[('fact',edge_type2idx[(val, pos_code)],'entity')] = []
        graph_dict[('fact',edge_type2idx[(val, pos_code)],'entity')].append((fact_node_idx,node2idx[arg]))
  print("making dgl graph")
  g = dgl.heterograph(graph_dict)
  return g