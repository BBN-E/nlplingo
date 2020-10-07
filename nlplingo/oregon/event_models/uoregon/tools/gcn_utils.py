import torch
import numpy as np

from collections import defaultdict


class Tree(object):
    """
    Reused tree object from stanfordnlp/treelstm.
    """

    def __init__(self):
        self.parent = None
        self.num_children = 0
        self.children = list()

    def add_child(self, child):
        child.parent = self
        self.num_children += 1
        self.children.append(child)

    def size(self):
        if getattr(self, '_size'):
            return self._size
        count = 1
        for i in range(self.num_children):
            count += self.children[i].size()
        self._size = count
        return self._size

    def depth(self):
        if getattr(self, '_depth'):
            return self._depth
        count = 0
        if self.num_children > 0:
            for i in range(self.num_children):
                child_depth = self.children[i].depth()
                if child_depth > count:
                    count = child_depth
            count += 1
        self._depth = count
        return self._depth

    def __iter__(self):
        yield self
        for c in self.children:
            for x in c:
                yield x


def get_head_token(head_id):
    return head_id - 1


def head_to_tree(head, tokens, actual_len, prune, subj_pos, obj_pos):
    """
    Convert a sequence of head indexes into a tree object.
    """
    tokens = tokens[:actual_len].tolist()
    head = head[:actual_len].tolist()
    root = None

    if prune < 0:
        nodes = [Tree() for _ in head]

        for i in range(len(nodes)):
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = -1  # just a filler
            if h == 0:
                root = nodes[i]
            else:
                nodes[get_head_token(head_id=h)].add_child(nodes[i])
    else:
        # find dependency path
        subj_pos = [i for i in range(actual_len) if subj_pos[i] == 0]
        obj_pos = [i for i in range(actual_len) if obj_pos[i] == 0]

        common_ancestors = None

        subj_ancestors = set(subj_pos)
        for s in subj_pos:
            h = head[s]
            tmp = [s]
            while h > 0:
                head_token = get_head_token(head_id=h)
                tmp += [head_token]
                subj_ancestors.add(head_token)
                h = head[head_token]  # get parent node

            if common_ancestors is None:
                common_ancestors = set(tmp)
            else:
                common_ancestors.intersection_update(tmp)  # apply join operation between two sets

        obj_ancestors = set(obj_pos)
        for o in obj_pos:
            h = head[o]
            tmp = [o]
            while h > 0:
                head_token = get_head_token(head_id=h)
                tmp += [head_token]
                obj_ancestors.add(head_token)
                h = head[head_token]
            common_ancestors.intersection_update(tmp)

        # find lowest common ancestor
        if len(common_ancestors) == 1:
            lca = list(common_ancestors)[0]
        else:
            child_count = {k: 0 for k in common_ancestors}
            for ca in common_ancestors:
                head_of_ca = head[ca]
                head_token_of_ca = get_head_token(head_id=head_of_ca)
                if head_of_ca > 0 and head_token_of_ca in common_ancestors:
                    child_count[head_token_of_ca] += 1

            # the LCA has no child in the CA set
            for ca in common_ancestors:
                if child_count[ca] == 0:
                    lca = ca
                    break

        path_nodes = subj_ancestors.union(obj_ancestors).difference(common_ancestors)
        path_nodes.add(lca)

        # compute distance to path_nodes
        dist = [-1 if i not in path_nodes else 0 for i in range(actual_len)]

        for i in range(actual_len):
            if dist[i] < 0:
                stack = [i]
                while stack[-1] >= 0 and stack[-1] not in path_nodes:
                    stack.append(get_head_token(head[stack[-1]]))

                if stack[-1] in path_nodes:
                    for d, j in enumerate(reversed(stack)):
                        dist[j] = d
                else:
                    for j in stack:
                        if j >= 0 and dist[j] < 0:
                            dist[j] = int(1e4)  # aka infinity

        highest_node = lca
        nodes = [Tree() if dist[i] <= prune else None for i in range(actual_len)]

        for i in range(len(nodes)):
            if nodes[i] is None:
                continue
            h = head[i]
            nodes[i].idx = i
            nodes[i].dist = dist[i]
            if h > 0 and i != highest_node:
                head_token = get_head_token(head_id=h)
                assert nodes[head_token] is not None
                nodes[head_token].add_child(nodes[i])

        root = nodes[highest_node]

    assert root is not None
    return root


def tree_to_adj(sent_len, tree, directed=True, self_loop=False):
    """
    Convert a tree object to an (numpy) adjacency matrix.
    """
    ret = np.zeros((sent_len, sent_len), dtype=np.float32)

    queue = [tree]
    idx = []
    while len(queue) > 0:
        t, queue = queue[0], queue[1:]

        idx += [t.idx]

        for c in t.children:
            ret[t.idx, c.idx] = 1
        queue += t.children

    if not directed:
        ret = ret + ret.T

    if self_loop:
        for i in idx:
            ret[i, i] = 1

    return ret


def tree_to_dist(sent_len, tree):
    ret = -1 * np.ones(sent_len, dtype=np.int64)

    for node in tree:
        ret[node.idx] = node.dist

    return ret


def get_full_adj(head_ids, token_ids, device, self_loop=True):
    head_ids = head_ids.data.cpu().numpy()
    batch_size, seq_len = token_ids.shape
    adj_list = []
    for b_id in range(batch_size):
        adj = np.zeros((seq_len, seq_len))
        for i in range(seq_len):
            adj[i][i] = 1 if self_loop else 0
            head = int(head_ids[b_id][i] - 1)
            if head >= 0:
                adj[i][head] = 1
                adj[head][i] = 1
        adj_list.append(adj)
    adj_list = np.array(adj_list)
    return torch.from_numpy(adj_list).float().to(device)


def get_pruned_adj(head_ids, retrieve_ids, subj_positions, obj_positions, pad_masks,
                   prune, device):
    with torch.no_grad():
        lengths = (pad_masks.data.cpu().numpy() == 0).astype(np.int64).sum(
            1)  # [batch size, ] actual length of each sequence in the batch
        maxlen = max(lengths)
        head_ids, retrieve_ids, subj_positions, obj_positions = head_ids.cpu().numpy(), retrieve_ids.cpu().numpy(), subj_positions.cpu().numpy(), obj_positions.cpu().numpy()
        trees = [head_to_tree(head_ids[i], retrieve_ids[i], lengths[i], prune, subj_positions[i],
                              obj_positions[i]) for i in range(len(lengths))]
        adj = [tree_to_adj(maxlen, tree, directed=False, self_loop=False).reshape(1, maxlen, maxlen) for tree in
               trees]
        adj = np.concatenate(adj, axis=0)
        adj = torch.from_numpy(adj)  # [batch size, max len, max len]
        adj = adj.float().to(device)
    return adj
