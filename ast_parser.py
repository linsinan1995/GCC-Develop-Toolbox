'''
 # @ Author: Sinan Lin
 # @ Email: sinan.lin@nokia.com
 # @ Create Time: 2021-07-26 23:47:00
 # @ Modified by: Sinan Lin
 # @ Modified time: 2021-07-26 23:47:09
 # @ Description:
  gcc-10 -fdump-tree-original-raw -g -S test.c
  python3 ast_parser.py --ast /home/sinan/plct/riscv-gnu-toolchain/gcc-lec/test.c.003t.original --simplify
'''

import re
from typing import List, Optional, Tuple
import argparse
from graphviz import Digraph

class AST:
    def __init__(self, fname):
        self.tree = []
        self.fname = fname
    def get_node(self, node_idx):
        return self.tree[node_idx]
    def add_node(self, node):
        self.tree.append(node)
    def __repr__(self):
        return '\n'.join([str(tree) for tree in self.tree])
    def to_graphviz(self, filename):
        g = Digraph('G', filename=filename, format='png', node_attr={'shape': 'record'})
        refdicts = dict()
        
        for ast_node in self.tree:
            net, refdict = ast_node.to_graphviz()
            g.node(f'struct{ast_node.node_idx}', net)
            if refdict:
                refdicts[ast_node.node_idx] = refdict

        for src_idx, val in refdicts.items():
            for attr_idx, dst_idx in refdicts[src_idx].items():
                g.edge(f'struct{src_idx}:v{attr_idx}', f'struct{dst_idx}:v0')
        g.view()

def is_value_node(node, ast) -> Tuple[bool, Optional[str]]:
    if node.node_type == "integer_cst":
        return True, node.attribute['int']
    elif node.node_type == 'identifier_node':
        return True, node.attribute['strg']
    elif node.node_type == 'type_decl':
        name = node.attribute['name']
        if isinstance(name, RefNode):
            name = ast[name.node_idx - 1].attribute['strg']
        return True, name

    return False, None

class RefNode:
    # uninitialize node
    def __init__(self, idx):
        self.node_idx = idx
    def __str__(self):
        return f'node-{self.node_idx}'

class ValNode:
    # node contains literal value
    def __init__(self, idx, val):
        self.node_idx = idx
        self.val = val
    def __str__(self):
        return f'{self.val}'

class Node:
    def __init__(self, idx, category):
        self.node_idx = idx
        self.node_type = category
        self.attribute = dict()
    def add_attribute(self, attr_name, val):
        self.attribute[attr_name] = val
    def to_graphviz(self):
        refdict = dict()
        def update_refnode_dict_cb(v,i):
            if isinstance(v, RefNode):
                refdict[i] = v.node_idx
                return False
            
            return True

        keys = list(self.attribute.keys())
        values = list(self.attribute.values())

        attr_with_label = ['<k0> idx', '<k1> tree type']
        val_with_label = [f'<v0> node-{self.node_idx}', f'<v1> {self.node_type}']
        attr_with_label = '|'.join(attr_with_label + [f"<k{i+2}> {k}" for i,k in enumerate(keys)])
        val_with_label  = '|'.join(val_with_label + [f"<v{i+2}> {v}" if update_refnode_dict_cb(v,i+2) else f"<v{i+2}> " for i,v in enumerate(values)])

        net = f"{{{attr_with_label}}} | {{{val_with_label}}}"

        return net, refdict

    def __repr__(self):
        attr = ''
        for v, k in self.attribute.items():
            if isinstance(k, Node):
                # is type node
                # if len(self.node_type) > 4 and self.node_type[-4:] == 'type':
                #     k = f'{self.node_type}@{self.node_idx}'
                # elif len(self.node_type) > 3 and self.node_type[-3:] == 'cst':
                #     val = self.valu if self.node_type == 'real_cst' else self.int
                #     k = f'{val}@{self.node_idx}'
                # else:
                k = f'node-{k.node_idx}'
            attr += f', {v}:{k}'
        return f'node-{self.node_idx}, name:{self.node_type}{attr}'

def parse_attribute(node:Node, stream:str) -> Node:
    stream = stream.strip()
    while len(stream) > 0:
        # parse attribute name
        val = None
        idx = 4
        attr_name = stream[:idx]
        # align attr name (`int:`)
        if ':' in attr_name:
            idx = attr_name.find(':')
            attr_name = stream[:idx]
        # eat ':'
        stream = stream[(idx+1):].strip()
        # parse attribute value
        i = 0
        if stream[i] == '@':
            val, stream = parse_index(stream)
            val = RefNode(val)
        else:
            val = ''
            while i < len(stream) and not stream[i].isspace():
                val += stream[i]
                i += 1
            if val.isnumeric():
                try:
                    val = int(val)
                except ValueError:
                    val = float(val)
        stream = stream[i:].strip() if i < len(stream) else ''
        node.add_attribute(attr_name, val)
    return node

def parse_category(stream: str) -> int:
    stream = stream.strip()
    i = 0
    ret = ''
    while not stream[i].isspace():
        ret += stream[i]
        i += 1
    return ret, stream[i:] if i < len(stream) else ''

def parse_index(stream: str) -> int:
    stream = stream.strip()
    assert(stream[0] == '@')
    i = 1
    ret = ''
    while i < len(stream) and stream[i].isdigit():
        ret += stream[i]
        i += 1
    return int(ret), stream[i:]

def main(args=None):
    parser = argparse.ArgumentParser(description='Simple script for drawing AST.')
    parser.add_argument('--ast', help='path.', type=str, default=None)
    parser.add_argument('--out', help='path.', type=str, default='default')
    parser.add_argument('--simplify', dest='simplify', action='store_true', help='remove integer_cst node')
    parser.add_argument('--no-simplify', dest='simplify', action='store_false', help='remove integer_cst node')
    parser.set_defaults(simplify=False)

    parser = parser.parse_args(args)
    p = parser.ast
    
    assert p and "Use --ast to indicate the path of ast file"

    asts = []
    ast = None
    with open(p, 'r') as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if len(line) < 2 or line[0] == ';':
                if ';; Function ' in line:
                    if ast is not None:
                        asts.append(ast)
                    ast = AST(line.split()[2])
                continue
            if line[0] != '@':
                ast.tree[-1] = parse_attribute(ast.tree[-1], line)
            else:
                idx, line = parse_index(line)
                category, line = parse_category(line)
                node = Node(idx, category) if category else ValNode(idx)
                ast.add_node(node)
                node = parse_attribute(node, line)

        if parser.simplify:
            for node in ast.tree:                
                for k, v in node.attribute.items():
                    if isinstance(v, RefNode):
                        flag, ret = is_value_node(ast.tree[v.node_idx - 1], ast.tree)
                        if flag:
                            node.attribute[k] = ret
                        else:
                            v = ast.tree[v.node_idx - 1]
                    

            for i in range(len(ast.tree)-1, -1, -1):
                node = ast.tree[i]
                if node.node_type in ['integer_cst', 'identifier_node', 'type_decl']:
                    ast.tree.pop(i)
                elif node.node_type == 'integer_type': # remove bitsizetype
                    name = node.attribute['name']
                    if isinstance(name, RefNode):
                        name = ast[name.node_idx - 1].attribute['strg']
                    if name == 'bitsizetype':
                        ast.tree.pop(i)

        asts.append(ast)
    print(asts)

    for i, tree in enumerate(asts):
        filename = parser.out if i == 0 else f"{parser.out}-{i}"
        tree.to_graphviz(filename)

if __name__ == '__main__':
    main()