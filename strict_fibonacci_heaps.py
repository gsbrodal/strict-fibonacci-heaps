'''
    An implementation of "Strict Fibonacci Heaps", by
    Gerth Stølting Brodal, George Lagogiannis, and Robert E. Tarjan.
    Transactions on Algorithms.

    This implementation is by no means supposed to be efficient.
    Its main purpose is to support the correctness of the paper.
    A comprehensive set of assertions check for the structural
    integrity of the data structure during the execution.

    (C) Gerth Stølting Brodal, George Lagogiannis, and Robert E. Tarjan.
        November 28, 2024.
'''


import math
import random


class Heap:
    '''Strict Fibonacci heaps - A pointer based meldable heap.

    The class supports the following operations:

      - Heap() creates and returns an empty heap.
      - H.empty() returns if the heap H is empty.
      - H.find_min() returns a reference to the node in H with minimum key.
      - H.delete(x) deletes the node x from the heap H.
      - H.insert(key, value) creates in heap H a node x with item (key, value)
        and returns x.
      - H1.meld(H2) melds heaps H1 and H2, and returns the resulting heap.
      - H.delete_min() deletes the node with minimum key from the heap H.
      - H.decrease_key(x, key) decreases the key of the item in node x to key.

    node.item() returns the item (key, value) stored in a node.

    The operations FindMin, Insert, Meld and DecreaseKey take worst-case O(1)
    time, and Delete and DeleteMin worst-case O(log n) time.
    The heap implementation is purely pointer based, except for O(1) sized
    lists used to hold the list of reductions to apply.
    '''

    def __init__(self):
        '''Initialize a new empty heap.'''

        self._active = True
        self._size = 0
        self._root = None
        self._rank_list = None
        self._fix_passive = None
        self._fix_free_multiple = None
        self._fix_free_single = None
        self._fix_loss_zero = None
        self._fix_loss_one_multiple = None
        self._fix_loss_one_single = None
        self._fix_loss_two = None

    def empty(self):
        '''Return if heap is empty.'''

        assert self._active

        return self._size == 0

    def find_min(self):
        '''Return the node with the item with smallest key.'''

        assert self._active
        assert self._size >= 1

        return self._root

    def delete(self, node):
        '''Delete the node from this heap.'''

        assert self._active
        assert node.passive() or node.heap() is self

        self.decrease_key(node, Node._MINUS_INFINITY)
        self.delete_min()

    def insert(self, key, value):
        '''Insert new (key, value) item into heap.'''

        assert self._active
        assert key is not None

        new_node = Node(self, key, value)  # new free rank 0 node
        if self._root is None:
            self._root = new_node
        else:
            self._root = self.link(self._root, new_node)
            self.apply_reductions(3 * [self.reduce_free] +
                                  2 * [self.reduce_root])
        return new_node

    def meld(self, other):
        '''Meld this heap with another heap. Returns the resulting heap.'''

        assert self._active and other._active

        if self._size < other._size:
            small, large = self, other
        else:
            small, large = other, self
        small._active = False  # The main novelty in the paper
        large._size += small._size
        if small._size > 0:
            # Retire small heap record
            small_root = small._root
            small._size = 0
            small._root = None
            small._rank_list = None
            small.fix_list_retire()
            # Insert small's fix-list as passive at front of large's fix-list
            large_head = large.fix_list_head()
            small_head = small_root
            small_tail = small_root._prev
            small_head._prev = large_head._prev
            small_tail._next = large_head
            small_tail._next._prev = small_tail
            small_head._prev._next = small_head
            large._fix_passive = small_head
            # Link roots and apply reductions
            large._root = large.link(large._root, small_root)
            large.apply_reductions([large.reduce_free, large.reduce_root])

        return large

    def delete_min(self):
        '''Delete the minimum (the root) from a non-empty heap.'''

        assert self._active
        assert self._size >= 1

        root = self._root
        if self._size == 1:
            assert root._left_child is None
            self._root = None
            root.retire(self)
        else:
            new_root = min(root.children())  # uses Node.__lt__
            if new_root.fixed():
                new_root.fixed2free()
            new_root.cut(self)
            while root._left_child is not None:
                child = root._left_child
                if child.fixed():
                    child.fixed2free()
                child.cut(self)
                self.link(new_root, child)
            self._root = new_root
            root.retire(self)
            self.reduce_passive()
            while self.reduce_loss():
                pass  # repeat until no loss reductions possible
            while self.reduce_free() or self.reduce_root():
                pass  # repeat until no free or root reduction can be applied

    def decrease_key(self, node, key):
        '''Replace key of node with a smaller key (None = -infinity).'''

        assert self._active
        assert node.passive() or node.heap() is self
        assert key is Node._MINUS_INFINITY or key <= node._key

        node._key = key
        parent = node._parent
        if parent is None or parent < node:
            return
        if node.fixed():
            node.fixed2free()
        node.cut(self)
        self._root = self.link(node, self._root)
        self.reduce_loss()
        self.apply_reductions(6 * [self.reduce_free] + 4 * [self.reduce_root])

    ##################################################################
    #                           Reductions
    ##################################################################

    def link(self, x, y):
        '''Link two nodes x and y. Return the winner (parent of the other).'''

        assert self._active
        assert x is not None
        assert y is not None
        assert x is not y
        assert x.passive() or x.heap() is self
        assert y.passive() or y.heap() is self

        if x < y:  # key comparison
            x.add_child(y)
            return x
        else:
            y.add_child(x)
            return y

    def apply_reductions(self, reductions):
        '''Apply reductions to the extend possible.'''

        while reductions:
            failed = []
            for reduction in reductions:
                if not reduction():
                    failed.append(reduction)
            if len(failed) == len(reductions):
                return  # cannot apply any further reductions
            reductions = failed  # at least one reduction succeeded

    def reduce_passive(self):
        '''Make three passive nodes free nodes (or all if less passive).'''

        assert self._active

        for _ in range(3):
            if self._fix_passive is None:
                return
            node = self._fix_passive
            node.passive2free(self)

    def reduce_free(self):
        '''Apply a free node reduction. Returns if successful.'''

        assert self._active

        x = self._fix_free_multiple
        if x is None:
            return False
        y = x._next
        if x > y:
            x, y = y, x

        assert x is not y
        assert x.free() and y.free()
        assert x._rank is y._rank
        assert x.heap() is self
        assert y.heap() is self
        assert x < y

        y.cut(self)  # y is not root
        y.free2fixed()
        x.add_child(y)
        z = x._left_child._left
        if z.passive():
            z.cut(self)
            self._root.add_child(z)
        return True

    def reduce_root(self):
        '''Apply a root degree reduction.'''

        assert self._active

        root = self._root
        if root is None or root._left_child is None:
            return False  # no root or children

        z = root._left_child._left  # rightmost child
        y = z._left
        x = y._left

        if z is y or z is x or not x.passive():
            return False  # not three passive children

        assert x.passive()
        assert y.passive()
        assert z.passive()
        assert x is not y is not z is not x

        x.cut(self)
        y.cut(self)
        z.cut(self)
        x.passive2free(self)
        y.passive2free(self)
        z.passive2free(self)
        if y < x:
            x, y = y, x
        if z < y:
            y, z = z, y
            if y < x:
                x, y = y, x

        assert x < y < z

        y.free2fixed()
        z.free2fixed()
        root.add_child(x)
        x.add_child(y)
        y.add_child(z)

        return True

    def reduce_loss(self):
        '''Apply single or two node loss reductions.'''

        return self.reduce_loss_one() or self.reduce_loss_two()

    def reduce_loss_one(self):
        '''Apply two node loss reduction (loss = 1), return if successful.'''

        assert self._active

        x = self._fix_loss_one_multiple
        if x is None:
            return False
        y = x._next
        if y < x:
            x, y = y, x

        assert x.fixed() and y.fixed()
        assert x._loss == 1
        assert y._loss == 1
        assert x.heap() is self
        assert y.heap() is self
        assert x._rank is y._rank
        assert x < y

        y.cut(self)  # y is not root since fixed
        x.fix_list_remove(self)
        y.fix_list_remove(self)
        x._loss -= 1
        y._loss -= 1
        x.fix_list_add(self)
        y.fix_list_add(self)
        x.add_child(y)
        z = x._left_child._left
        if z.passive():
            z.cut(self)
            self._root.add_child(z)

        return True

    def reduce_loss_two(self):
        '''Apply one node loss reduction (loss >= 2), return if successful.'''

        assert self._active

        x = self._fix_loss_two
        if x is None:
            return False

        assert x.fixed()
        assert x._loss >= 2
        assert x.heap() is self

        x.fixed2free()

        return True

    ##################################################################
    #                           Rank list
    ##################################################################

    def rank_zero(self):
        '''Get rank record for rank zero. Create if it does not exist.'''

        rank = self._rank_list

        assert self._active
        assert rank is None or rank._rank == 0

        if rank is None:
            self._rank_list = rank = Rank(self, 0)
        rank.increase_reference_count()

        return rank

    ##################################################################
    #                           Fix-list
    ##################################################################

    # Fix-list groups

    PASSIVE           = 0
    FREE_MULTIPLE     = 1
    FREE_SINGLE       = 2
    LOSS_ZERO         = 3
    LOSS_ONE_MULTIPLE = 4
    LOSS_ONE_SINGLE   = 5
    LOSS_TWO          = 6

    def fix_list_retire(self):
        '''Set all references to fix-list to None.'''

        self._fix_passive           = None
        self._fix_free_multiple     = None
        self._fix_free_single       = None
        self._fix_loss_zero         = None
        self._fix_loss_one_multiple = None
        self._fix_loss_one_single   = None
        self._fix_loss_two          = None

    def fix_list_head(self):
        '''Return first node on fix-list. Returns None if empty.'''

        return self.fix_list_insertion_point(Heap.PASSIVE)

    def fix_list_insertion_point(self, group):
        '''Find insertion point starting at fix-list group.'''

        points = (
            self._fix_passive,
            self._fix_free_multiple,
            self._fix_free_single,
            self._fix_loss_zero,
            self._fix_loss_one_multiple,
            self._fix_loss_one_single,
            self._fix_loss_two
        )
        for point in points[group:] + points[:group]:
            if point is not None:
                return point
        return None

    def fix_list_insert(self, group, node):
        '''Insert active node at front of fix-list group.'''

        assert node.active()

        succ = self.fix_list_insertion_point(group)
        node.fix_list_insert_before(succ)

        if   group == Heap.FREE_MULTIPLE    : self._fix_free_multiple     = node
        elif group == Heap.FREE_SINGLE      : self._fix_free_single       = node
        elif group == Heap.LOSS_ZERO        : self._fix_loss_zero         = node
        elif group == Heap.LOSS_ONE_MULTIPLE: self._fix_loss_one_multiple = node
        elif group == Heap.LOSS_ONE_SINGLE  : self._fix_loss_one_single   = node
        elif group == Heap.LOSS_TWO         : self._fix_loss_two          = node

    ##################################################################
    #                      Validation methods
    ##################################################################

    def validate_rank_list(self):
        '''Validate pointers and values in rank-lise.'''

        heap = self
        rank = heap._rank_list
        if heap._active and heap._size == 0:
            assert rank is None
        else:
            assert rank is not None
            assert rank._prev is None
            while rank is not None:
                assert rank._reference_count > 0
                assert rank._heap is heap
                assert rank._rank is not None
                free = rank._free
                if free is not None:
                    assert free.free()
                    assert free._rank is rank
                loss_one = rank._loss_one
                if loss_one is not None:
                    assert loss_one.fixed()
                    assert loss_one._loss == 1
                    assert loss_one._rank is rank
                if rank._next is not None:
                    assert rank._next._prev is rank
                    assert rank._next._rank > rank._rank
                rank = rank._next

    def validate_fix_list(self):
        '''Validate fix-list of active heap.'''

        heap = self
        assert self._active
        if heap._size == 0:
            assert heap._fix_passive is None
            assert heap._fix_free_multiple is None
            assert heap._fix_free_single is None
            assert heap._fix_loss_zero is None
            assert heap._fix_loss_one_multiple is None
            assert heap._fix_loss_one_single is None
            assert heap._fix_loss_two is None
        else:
            head = heap.fix_list_head()
            assert head is not None
            # Verify cyclic _next and _prev links
            node = head
            assert node._next._prev is node
            node = node._next
            while node is not head:
                assert node._next._prev is node
                node = node._next
            # Verify links from heap record to fix list and fix list grouping
            node = head
            if heap._fix_passive is not None:
                assert heap._fix_passive is node
                assert node.passive()
                node = node._next
                while node is not head and node.passive():
                    node = node._next
            free_ranks = set()
            if heap._fix_free_multiple is not None:
                assert node is heap._fix_free_multiple
                assert node is not node._next
                assert node._next is not head
                assert node.free()
                assert node._next.free()
                assert node._rank is node._next._rank
                assert node._rank._free is node
                free_ranks.add(node._rank._rank)
                node = node._next
                while (node is not head and
                       node.free() and
                       node._rank is node._prev._rank):
                    node = node._next
                while (node is not head and
                       node.free() and
                       node._next is not head and
                       node._next.free() and
                       node._rank is node._next._rank):
                    assert node._rank._rank not in free_ranks
                    free_ranks.add(node._rank._rank)
                    node = node._next
                    while (node is not head and
                           node.free() and
                           node._rank is node._prev._rank):
                        node = node._next
            if heap._fix_free_single is not None:
                assert heap._fix_free_single is node
                assert node.free()
                assert node._rank._rank not in free_ranks
                assert (heap._fix_free_multiple is not None or
                        node._rank._free is node)
                free_ranks.add(node._rank._rank)
                node = node._next
                while node is not head and node.free():
                    assert node._rank._rank not in free_ranks
                    free_ranks.add(node._rank._rank)
                    node = node._next
            if heap._fix_loss_zero is not None:
                assert node is heap._fix_loss_zero
                assert node.fixed()
                assert node._loss == 0
                node = node._next
                while node is not head and node.fixed() and node._loss == 0:
                    node = node._next
            loss_one_ranks = set()
            if heap._fix_loss_one_multiple is not None:
                assert node is heap._fix_loss_one_multiple
                assert node is not node._next
                assert node._next is not head
                assert node.fixed()
                assert node._next.fixed()
                assert node._loss == 1
                assert node._next._loss == 1
                assert node._rank is node._next._rank
                assert node._rank._loss_one is node
                loss_one_ranks.add(node._rank._rank)
                node = node._next
                while (node is not head and
                       node.fixed() and
                       node._loss == 1 and
                       node._rank is node._prev._rank):
                    node = node._next
                while (node is not head and
                       node.fixed() and
                       node._loss == 1 and
                       node._next is not head and
                       node._next.fixed() and
                       node._next._loss == 1 and
                       node._rank is node._next._rank):
                    # more ranks with multiple loss one
                    assert node._rank not in loss_one_ranks
                    loss_one_ranks.add(node._rank._rank)
                    node = node._next
                    while (node is not head and
                           node.fixed() and
                           node._loss == 1 and
                           node._rank is node._prev._rank):
                        node = node._next
            if heap._fix_loss_one_single is not None:
                assert node is heap._fix_loss_one_single
                assert node.fixed()
                assert node._loss == 1
                assert node._rank._rank not in loss_one_ranks
                assert (heap._fix_loss_one_multiple is not None or
                        node._rank._loss_one is node)
                loss_one_ranks.add(node._rank._rank)
                node = node._next
                while node is not head and node.fixed() and node._loss == 1:
                    assert node._rank._rank not in loss_one_ranks
                    loss_one_ranks.add(node._rank._rank)
                    node = node._next
            if heap._fix_loss_two is not None:
                assert node is heap._fix_loss_two
                assert node.fixed()
                assert node._loss >= 2
                node = node._next
                while node is not head and node.fixed() and node._loss >= 2:
                    node = node._next
            assert node is head  # processed whole fix-list

    def validate(self):
        '''Validate all heap structure and invariants.'''

        def validate_tree(node, parent=None):
            '''Recursive validate tree nodes.'''

            # Validate parent pointer
            assert node._parent is parent
            # Validate heap order
            if parent is not None:
                assert parent < node
            # Validate sibling pointers (cyclic linked list)
            assert node is node._right._left
            assert node is node._left._right
            # Validate passive heap records
            if node.passive():
                assert node.heap()._size == 0
            # Validate active node heap records all point to this heap
            if node.active():
                assert node.heap() is heap
            # Invariant I1 - fixed nodes have active parent
            if node.fixed():
                assert parent is not None
                assert parent.active()
            # Verify children right to left
            fixed = 0
            allow_passive = True
            for child in reversed(list(node.children())):
                if child.active():
                    # Invariant I1 - active children left of passive children
                    if child.fixed():
                        fixed += 1
                        #  I1 - i'th rightmost fixed child rank + loss >= i - 1
                        assert child._rank._rank + child._loss >= fixed - 1
                    allow_passive = False
                else:
                    assert child.passive()
                    assert allow_passive
            if node.active():
                assert node._rank._rank == fixed  # rank = # fixed children
            # Invariant I4
            degree = len(list(node.children()))
            assert degree <= Delta if node.active() else degree <= Delta - 1
            # All active ranks <= R
            if node.active():
                assert node._rank._rank <= R
            # recurse
            for child in node.children():
                validate_tree(child, node)

        heap = self
        root = heap._root
        assert heap._active
        heap.validate_rank_list()
        heap.validate_fix_list()
        if heap._size == 0:
            assert root is None
        else:
            assert root is not None
            size = sum(1 for node in root.all_nodes())
            assert size == heap._size
            passive = sum(node.passive() for node in root.all_nodes())
            R = 5 / 4 * math.log(size, 2) + 6
            Delta = 5 / 2 * math.log(3 * size - passive, 2) + 14
            # Invariant I2
            free = sum(node.free() for node in root.all_nodes())
            assert free <= R + 1
            # Invariant I3
            loss = sum(node._loss for node in root.all_nodes() if node.fixed())
            assert loss <= R + 1
            validate_tree(root)

    ##################################################################
    #                        Save heap as LaTeX
    ##################################################################

    def latex(self, filename='heap_figure.tex', show_keys=False):
        '''Save heap as a LaTeX figure using the forest package.'''

        heap = self
        root = heap._root

        assert heap._active
        assert root is not None

        def traverse(node, indent=0):
            '''Convert subtree rooted at node to Latex with indentation.'''

            key = str(node._key) if show_keys else ''
            if node.passive():
                txt = r'\PASSIVE{' + key + '}, passive'
            elif node.free():
                txt = r'\ACTIVE{' + str(node._rank._rank)
                txt += r'}{' + key + '}, free'
            else:
                txt = r'\ACTIVE{' + str(node._rank._rank) + '/'
                txt += str(node._loss) + r'}{' + key + '}, fixed'
            if node._left_child is None:
                txt = ' ' * indent + '[ ' + txt + ' ]\n'
            else:
                txt = ' ' * indent + '[ ' + txt + '\n'
                for child in node.children():
                    txt += traverse(child, indent + 2)
                txt += ' ' * indent + ']\n'
            return txt

        txt = r'''\documentclass[margin=15pt]{standalone}
\usepackage{forest}
\begin{document}
\forestset{forest circles/.style={
    for tree={math content, draw, circle,
      inner sep=0pt, outer sep=0cm, anchor=center,
      l=25pt, s sep=20pt, edge=dashed},
    passive/.style={fill=black, minimum size=4pt},
    fixed/.style={fill=black!15, font=\scriptsize, edge=solid, minimum size=16pt, inner sep=0pt},
    free/.style={minimum size=16pt, inner sep=0pt,font=\scriptsize}
  }
}
\newcommand{\ACTIVE}[2]{\makebox[0cm][c]{#1}\rlap{\hspace{1.5em}\tiny #2}}
\newcommand{\PASSIVE}[1]{\rlap{\hspace{0.5em}\tiny #1}}
\begin{forest}
  forest circles,
''' + traverse(root) + r'''\end{forest}
\end{document}
'''
        with open(filename, 'w') as file:
            print(txt, file=file)


######################################################################
#                           Node records
######################################################################


class Node:
    '''A node storing item node.item() = (key, value).'''

    _MINUS_INFINITY = object()

    def __init__(self, heap, key, value):
        '''Create a free node with rank zero for this heap.'''

        assert heap._active

        heap._size += 1
        # item information
        self._key = key
        self._value = value
        # tree structure
        self._parent = None
        self._left_child = None
        self._right = self  # no sibling
        self._left = self  # no sibling
        # state
        self._free = True  # True = free, False = fixed
        self._loss = None  # int when fixed, None when free
        self._rank = heap.rank_zero()
        # fix list
        self._prev = self
        self._next = self
        self.fix_list_add(heap)

    def retire(self, heap):
        '''Make this single node disconnected from the heap.'''

        assert not self.retired()
        assert heap._active
        assert self.passive() or self.heap() is heap

        assert self._parent is None
        assert self._left_child is None
        assert self._left is self._right is self

        rank = self._rank
        self.fix_list_remove(heap)
        heap._size -= 1
        self._rank = None
        rank.decrease_reference_count()

    #######################################################
    # Methods for accessing the state of a node

    def item(self):
        '''Return the item (key, value) stored in node.'''

        assert not self.retired()

        return (self._key, self._value)

    def retired(self):
        '''Return if this not has been retired.'''

        return self._rank is None

    def heap(self):
        '''Return the heap record for this node (can be a passive).'''

        assert not self.retired()
        assert self._rank._heap is not None

        return self._rank._heap

    def active(self):
        '''Return if the node is active.'''

        assert not self.retired()

        return self.heap()._active

    def passive(self):
        '''Return if the node is passive.'''

        assert not self.retired()

        return not self.active()

    def free(self):
        '''Return if the node is free.'''

        assert not self.retired()

        return self.active() and self._free

    def fixed(self):
        '''Return if the node is fixed.'''

        assert not self.retired()

        return self.active() and not self._free

    def __lt__(self, other):
        '''Compare nodes by key (None = -infinity). Ensures dinstinct keys.'''

        assert self is not other

        return (self._key is Node._MINUS_INFINITY or
                (other._key is not Node._MINUS_INFINITY and
                 (self._key < other._key or
                  (self._key == other._key and id(self) < id(other)))))

    def children(self):
        '''Generator to return all children of node.'''

        child = self._left_child
        if child is not None:
            yield child
            while child._right is not self._left_child:
                child = child._right
                yield child

    def all_nodes(self):
        '''Generator to yield all nodes in subtree rooted at node.'''

        yield self
        for child in self.children():
            yield from child.all_nodes()

    def height(self):
        '''Return height of subtree rooted at node.'''

        return 1 + max((child.height() for child in self.children()), default=0)

    #######################################################
    # Methods for modifying the state of a node

    def change_rank(self, new_rank):
        '''Change rank of active node to new_rank.'''

        heap = self.heap()
        assert self.active()
        assert heap is new_rank._heap

        new_rank.increase_reference_count()
        self.fix_list_remove(heap)
        self._rank.decrease_reference_count()
        self._rank = new_rank
        self.fix_list_add(heap)

    def increase_rank(self):
        '''Increase rank of node by one.'''

        self.change_rank(self._rank.next())

    def decrease_rank(self):
        '''Decrease rank of node by one.'''

        assert self.active()
        assert self._rank._rank > 0

        self.change_rank(self._rank.prev())

    def add_child(self, child):
        '''Add child as rightmost child if passive, otherwise leftmost.'''

        assert self < child
        assert child._parent is None
        assert child._left is child
        assert child._right is child

        child._parent = self
        if self._left_child is None:
            # First and only child
            self._left_child = child
        else:
            # update left-right (new child before leftmost child)
            child._right = self._left_child
            child._left = child._right._left
            child._right._left = child
            child._left._right = child
            # Make child leftmost (active) or rightmost (passive)
            if not child.passive():
                self._left_child = child
        if self.active() and child.fixed():
            self.increase_rank()

    def cut(self, heap):
        '''Remove link to parent in this heap.'''

        parent = self._parent
        right = self._right
        left = self._left

        assert not self.retired()
        assert heap._active
        assert self.passive() or self.heap() is heap
        assert parent is not None

        # update parent
        self._parent = None
        if parent._left_child is self:
            if right is not self:
                parent._left_child = right
            else:
                parent._left_child = None
        # update left-right
        if right is not self:
            left._right = right
            right._left = left
            self._right = self
            self._left = self
        # update parent loss
        if self.fixed() and parent.active():
            parent.decrease_rank()
            if parent.fixed():
                parent.fix_list_remove(heap)
                parent._loss += 1
                parent.fix_list_add(heap)

    def free2fixed(self):
        '''Make this free node fixed.'''

        assert not self.retired()
        assert self.free()

        heap = self.heap()
        self.fix_list_remove(heap)
        self._free = False
        self._loss = 0
        self.fix_list_add(heap)

    def fixed2free(self):
        '''Make this free node fixed.'''

        assert not self.retired()
        assert self.fixed()

        parent = self._parent
        assert parent is not None
        assert parent.active()

        heap = self.heap()

        self.fix_list_remove(heap)
        self._free = True
        self._loss = None
        self.fix_list_add(heap)
        parent.decrease_rank()
        if parent.fixed():
            parent.fix_list_remove(heap)
            parent._loss +=1
            parent.fix_list_add(heap)

    def passive2free(self, heap):
        '''Make this passive node a free node of rank zero.'''

        assert not self.retired()
        assert self.passive()
        assert heap._active
        assert all(not child.fixed() for child in self.children())

        self.fix_list_remove(heap)
        self._rank.decrease_reference_count()
        self._rank = heap.rank_zero()
        self._free = True
        self._loss = None
        parent = self._parent
        if parent is not None:
            # make child leftmost child
            self.cut(heap)
            parent.add_child(self)
        self.fix_list_add(heap)

    #######################################################
    # Update fix-list

    def fix_list_unlink(self):
        '''Unlink this node from fix-list.'''

        self._prev._next = self._next
        self._next._prev = self._prev
        self._prev = self
        self._next = self

    def fix_list_insert_before(self, succ):
        '''Insert this node in fix-list before succ.'''

        assert self is self._next
        assert self is self._prev
        if succ is not None:
            self._next = succ
            self._prev = succ._prev
            self._next._prev = self
            self._prev._next = self

    def fix_list_insert_after(self, prev):
        '''Insert this node in fix-list after prev.'''

        assert self is self._next
        assert self is self._prev

        self._prev = prev
        self._next = prev._next
        self._next._prev = self
        self._prev._next = self

    def fix_list_same_group(self, other):
        '''Return if this node and other node are in same "multiple" group.'''

        return ((self.free() and other.free() and self._rank is other._rank) or
                (self.fixed() and other.fixed() and
                 self._rank is other._rank and
                 self._loss == 1 and other._loss == 1))

    def fix_list_group(self, head):
        '''Identify cardinality (<= 3) and first node of "multiple" group.'''

        count = 1
        first = last = self
        while (count < 3 and first is not head and
               self.fix_list_same_group(first._prev)):
            first = first._prev
            count += 1
        while (count < 3 and last._next is not head and
               self.fix_list_same_group(last._next)):
            last = last._next
            count += 1

        return count, first

    def fix_list_add(self, heap):
        '''Add active node to the fix_list of the heap.'''

        assert self.active()
        assert self.heap() is heap

        head = heap.fix_list_head()
        if self.free():
            first = self._rank._free
            if first is None:
                self._rank._free = self
                heap.fix_list_insert(Heap.FREE_SINGLE, self)
            else:
                count, _ = first.fix_list_group(head)
                if count >= 2:
                    self.fix_list_insert_after(first)
                else:
                    assert count == 1
                    if heap._fix_free_single is first:
                        if first._next is not head and first._next.free():
                            heap._fix_free_single = first._next
                        else:
                            heap._fix_free_single = None
                    first.fix_list_unlink()
                    heap.fix_list_insert(Heap.FREE_MULTIPLE, self)
                    heap.fix_list_insert(Heap.FREE_MULTIPLE, first)
                    assert first._next is self
                    assert self._rank._free is first
        else:
            assert self.fixed()
            if self._loss == 0:
                heap.fix_list_insert(Heap.LOSS_ZERO, self)
            elif self._loss >= 2:
                heap.fix_list_insert(Heap.LOSS_TWO, self)
            else:
                assert self._loss == 1
                first = self._rank._loss_one
                if first is None:
                    self._rank._loss_one = self
                    heap.fix_list_insert(Heap.LOSS_ONE_SINGLE, self)
                else:
                    count, _ = first.fix_list_group(head)
                    if count >= 2:
                        self.fix_list_insert_after(first)
                    else:
                        assert count == 1
                        if heap._fix_loss_one_single is first:
                            if (first._next is not head and
                                first._next.fixed() and
                                first._next._loss == 1):
                                heap._fix_loss_one_single = first._next
                            else:
                                heap._fix_loss_one_single = None
                        first.fix_list_unlink()
                        heap.fix_list_insert(Heap.LOSS_ONE_MULTIPLE, self)
                        heap.fix_list_insert(Heap.LOSS_ONE_MULTIPLE, first)
                        assert first._next is self

    def fix_list_remove(self, heap):
        '''Remove node from the fix_list of the heap.'''

        assert self.passive() or self.heap() is heap

        succ = self._next
        head = heap.fix_list_head()

        # Update rank-list _free
        if self.active() and self._rank._free is self:
            assert self.free()
            if (succ is not head and
                succ.free() and
                succ._rank is self._rank):
                self._rank._free = succ
            else:
                self._rank._free = None

        # Update rank-list _loss_one
        if self.active() and self._rank._loss_one is self:
            assert self.fixed() and self._loss == 1
            if (succ is not head and
                succ.fixed() and succ._loss == 1 and
                succ._rank is self._rank):
                self._rank._loss_one = succ
            else:
                self._rank._loss_one = None

        if heap._size == 1:
            assert head is self
            assert self._next is self
            heap.fix_list_retire()
        else: # at least two nodes in fix-list
            if self.passive():
                if self is heap._fix_passive:
                    if succ is not head and succ.passive():
                        heap._fix_passive = succ
                    else:
                        heap._fix_passive = None
            elif self.free():
                count, first = self.fix_list_group(head)
                if count == 1:
                    if heap._fix_free_single is self:
                        if succ is not head and succ.free():
                            heap._fix_free_single = succ
                        else:
                            heap._fix_free_single = None
                elif count >= 3:
                    if heap._fix_free_multiple is self:
                        heap._fix_free_multiple = succ
                else:
                    assert count == 2
                    other = first if first is not self else succ
                    if heap._fix_free_multiple is first:
                        first = first._next._next
                        if (first is not head and
                            first.free() and
                            first._next is not head and
                            first._next.free() and
                            first._rank is first._next._rank):
                            heap._fix_free_multiple = first
                        else:
                            heap._fix_free_multiple = None
                    other.fix_list_unlink()
                    heap.fix_list_insert(Heap.FREE_SINGLE, other)
            else:
                assert self.fixed()
                if self._loss == 0:
                    if heap._fix_loss_zero is self:
                        if succ is not head and succ.fixed() and succ._loss == 0:
                            heap._fix_loss_zero = succ
                        else:
                            heap._fix_loss_zero = None
                elif self._loss >= 2:
                    if heap._fix_loss_two is self:
                        if succ is not head and succ.fixed() and succ._loss >= 2:
                            heap._fix_loss_two = succ
                        else:
                            heap._fix_loss_two = None
                else:
                    assert self._loss == 1
                    count, first = self.fix_list_group(head)
                    if count == 1:
                        if heap._fix_loss_one_single is self:
                            if succ is not head and succ.fixed() and succ._loss == 1:
                                heap._fix_loss_one_single = succ
                            else:
                                heap._fix_loss_one_single = None
                    elif count >= 3:
                        if heap._fix_loss_one_multiple is self:
                            heap._fix_loss_one_multiple = succ
                    else:
                        assert count == 2
                        other = first if first is not self else succ
                        if heap._fix_loss_one_multiple is first:
                            first = first._next._next
                            if (first is not head and
                                first.fixed() and
                                first._loss == 1 and
                                first._next is not head and
                                first._next.fixed() and
                                first._next._loss == 1 and
                                first._rank is first._next._rank):
                                heap._fix_loss_one_multiple = first
                            else:
                                heap._fix_loss_one_multiple = None
                        other.fix_list_unlink()
                        heap.fix_list_insert(Heap.LOSS_ONE_SINGLE, other)
            # remove node from fix-list
            self.fix_list_unlink()


######################################################################
#                         Rank records
######################################################################


class Rank:
    '''Class to represent a node in the rank list for a heap record.'''

    def __init__(self, heap, rank):
        '''Create a new rank node for a given integer rank.'''

        assert heap._active
        assert rank >= 0

        self._next = None
        self._prev = None
        self._heap = heap
        self._rank = rank
        self._free = None
        self._loss_one = None
        self._reference_count = 0

    def retire(self):
        '''Remove rank node from heap and rank list.'''

        rank = self
        heap = rank._heap

        assert not rank.retired()
        assert rank._reference_count == 0
        assert not heap._active or rank._free is None
        assert not heap._active or rank._loss_one is None

        if heap._rank_list is rank:  # First rank on rank_list
            heap._rank_list = rank._next
        if rank._next is not None:
            rank._next._prev = rank._prev
        if rank._prev is not None:
            rank._prev._next = rank._next
        rank._next = None
        rank._prev = None
        rank._heap = None
        rank._rank = None
        rank._free = None
        rank._loss_one = None

    def retired(self):
        '''Return if this rank node is retired.'''

        return self._heap is None

    def decrease_reference_count(self):
        '''Decrease reference count to rank node. Retire if it becomes 0.'''

        assert not self.retired()
        assert self._reference_count > 0

        self._reference_count -= 1
        if self._reference_count == 0:
            self.retire()

    def increase_reference_count(self):
        '''Increase reference count to rank node.'''

        assert not self.retired()

        self._reference_count += 1

    def next(self):
        '''Return rank + 1. Create it if it does not exist.'''

        assert not self.retired()

        if self._next is None or self._next._rank > self._rank + 1:
            Rank(self._heap, self._rank + 1).insert_after(self)

        return self._next

    def prev(self):
        '''Return rank - 1. Create it if it does not exist.'''

        assert not self.retired()
        assert self._rank > 0
        assert self._prev is not None  # rank 0 exists, if rank > 0 exists

        if self._prev is None or self._prev._rank < self._rank - 1:
            Rank(self._heap, self._rank - 1).insert_after(self._prev)

        return self._prev

    def insert_after(self, previous_rank):
        '''Insert this rank after previous_rank.'''

        assert not self.retired()
        assert previous_rank is not None
        assert self._rank > previous_rank._rank

        self._prev = previous_rank
        self._next = previous_rank._next
        previous_rank._next = self
        if self._next is not None:
            self._next._prev = self


######################################################################
#                          Test methods
######################################################################


def swap(L, i, j):
    '''Swap entries L[i] and L[j].'''

    L[i], L[j] = L[j], L[i]


def pop_random(L):
    '''Remove a random element from L (by swapping with last element).'''

    swap(L, -1, random.randint(0, len(L) - 1))
    return L.pop()


def random_items(n, distinct_keys=False):
    '''Returns a list of n random (key, value) items (value=42).'''

    if distinct_keys:
        return [(key, 42) for key in random.sample(range(1, 3 * n), n)]
    else:
        return [(random.randint(1, n), 42) for _ in range(n)]


def delete_all(heap):
    '''Create sorted list with all items from heap by calling n x delete_min.'''

    sorted_sequence = []
    while not heap.empty():
        node = heap.find_min()
        heap.validate()
        sorted_sequence.append(node.item())
        heap.delete_min()
        heap.validate()
    heap.validate()
    return sorted_sequence


def test_sorting_insert(n):
    '''Sort using n x insert and n x delete_min.'''

    items = random_items(n)
    heap = Heap()
    heap.validate()
    # Create heap with n items
    for key, value in items:
        heap.insert(key, value)
        heap.validate()
    assert delete_all(heap) == sorted(items)


def test_sorting_meld(n):
    '''Sort using (n - 1) x meld in random order and n x delete_min.'''

    items = random_items(n)
    heaps = []
    # Create n heaps with one item
    for key, value in items:
        heap = Heap()
        heap.validate()
        heap.insert(key, value)
        heap.validate()
        heaps.append(heap)
    # Repeatedly meld two random heaps until one heap remains
    while len(heaps) >= 2:
        heap1 = pop_random(heaps)
        heap2 = pop_random(heaps)
        heap = heap1.meld(heap2)
        heap.validate()
        heaps.append(heap)
    heap = heaps.pop()
    assert delete_all(heap) == sorted(items)


def test_sorting_decreasekey(n):
    '''Sort using n x decrease_key and n x delete_min.'''

    items = random_items(n)
    heap = Heap()
    heap.validate()
    nodes = []
    # Create heap with n items each with the same large key
    for key, value in items:
        node = heap.insert(n + 1, value)
        heap.validate()
        nodes.append((node, key))
    # Decrease keys to the items real value
    random.shuffle(nodes)
    for node, key in nodes:
        heap.decrease_key(node, key)
        heap.validate()
    assert delete_all(heap) == sorted(items)


def test_sorting_sample(n, delete_probability=0.5):
    '''Sort n items with a sample removed using delete.'''

    items = random_items(n)
    heap = Heap()
    heap.validate()
    nodes = []
    # Create heap with n nodes
    for key, value in items:
        node = heap.insert(key, value)
        heap.validate()
        nodes.append((node, (key, value)))
    random.shuffle(nodes)
    # Remove sample
    items = []  # not deleted items
    for node, item in nodes:
        if random.random() < delete_probability:
            heap.delete(node)
            heap.validate()
        else:
            items.append(item)
    assert delete_all(heap) == sorted(items)


def test_sorting(n, repeats):
    '''Run all tests for increasing values of n, n increases with 50%.'''

    print('Sorting n =', n, end=' ', flush=True)
    for _ in range(1, repeats + 1):
        print('.', end='', flush=True)
        test_sorting_insert(n)
        test_sorting_meld(n)
        test_sorting_decreasekey(n)
        test_sorting_sample(n)
    print()


def test_random_operations(n):
    '''Test a random sequence of n heap operations.'''

    print(n, 'random heap operations ', end='', flush=True)
    heaps = []
    for iteration in range(1, n + 1):
        if iteration % 100 == 0:
            print('.', end='', flush=True)
        p = random.random()
        if len(heaps) == 0 or p < 0.05:  # new heap
            heap = Heap()
            heaps.append((heap, []))
            heap.validate()
        elif p < 0.1:  # meld
            if len(heaps) >= 2:
                heap1, S1 = pop_random(heaps)
                heap2, S2 = pop_random(heaps)
                heap = heap1.meld(heap2)
                heaps.append((heap, S1 + S2))
                heap.validate()
        elif p < 0.5:  # decrease_key
            heap, S = random.choice(heaps)
            if not heap.empty():
                node = random.choice(list(heap._root.all_nodes()))
                key = node._key
                new_key = random.randint(key - 25, key - 1)
                heap.decrease_key(node, new_key)
                S.remove(key)
                S.append(new_key)
                heap.validate()
        elif p < 0.8:  # insert
            heap, S = random.choice(heaps)
            key = random.randint(1, 100)
            heap.insert(key, 42)
            S.append(key)
            heap.validate()
        else:  # delete_min
            heap, S = random.choice(heaps)
            if not heap.empty():
                key, value = heap.find_min().item()
                assert key == min(S)
                S.remove(key)
                heap.delete_min()
                heap.validate()
        # Validate content of the heaps
        for heap, S in heaps:
            if heap.empty():
                assert len(S) == 0
            else:
                nodes = list(heap._root.all_nodes())
                assert sorted(S) == sorted(node._key for node in nodes)
    print(' final heap sizes:', *sorted(heap._size for heap, S in heaps))


######################################################################
#         Generation of figure illustrating a typical heap
######################################################################


def random_heap(size):
    '''Create a random heap using insert, meld and decrease_key.'''

    heaps = []
    for key, value in random_items(size, distinct_keys=True):
        heap = Heap()
        heap.insert(key, value)
        heaps.append(heap)
    while len(heaps) > 1:
        heap1 = pop_random(heaps)
        heap2 = pop_random(heaps)
        heap = heap1.meld(heap2)
        heaps.append(heap)
    heap = heaps.pop()
    for _ in range(size):
        nodes = list(heap._root.all_nodes())
        node = random.choice(nodes)
        key, value = node.item()
        new_key = random.randint(1, key)
        if new_key not in {node._key for node in heap._root.all_nodes()}:
            heap.decrease_key(node, new_key)

    return heap


def generate_figure(tex_file='heap-figure.tex'):
    '''Create latex document with figure showing a heap.

    The generated heap satisfies the following requirements:

      - The root is passive.
      - The root has at least 4 children.
      - The height is at most 6.
      - At least two fixed nodes have positive loss.
      - Not all free nodes are children of the root.
    '''

    print('Trying to create an illustrative heap ', end='', flush=True)
    while True:
        heap = random_heap(30)
        root = heap._root
        nodes = list(root.all_nodes())
        root_degree = len(list(root.children()))

        if (root.active()
          or root_degree <= 3
          or root.height() > 6
          or sum(node.fixed() and node._loss > 0 for node in nodes) < 2
          or sum(node._parent is not root for node in nodes if node.free()) < 2
        ):
            print('.', end='', flush=True)
            continue
        break
    print(' saving', tex_file)
    heap.latex(tex_file, show_keys=True)


######################################################################
#                               Main
######################################################################


if __name__ == '__main__':
    test_sorting(1, 10)
    test_sorting(10, 100)
    test_sorting(100, 10)
    test_random_operations(10000)
    generate_figure()
