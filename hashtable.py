class LinkedElement:

    def __init__(self, key, value):
        self.key = key
        self.val = value
        self.prev = None
        self.next = None

    def set_next(self, n):
        self.next = n

    def set_prev(self, p):
        self.prev = p

    def enext(self):
        return self.next

    def eprev(self):
        return self.prev

    def is_key(self, k):
        return self.key == k

    def value(self):
        return self.val

    def __str__(self):
        return f'{self.key}: {self.val}'


class LinkedList:

    def __init__(self):
        self.start = None
        self.end = None

    def get(self, elem=None, by_key=False, **kwargs):
        curr = self.start
        if by_key:
            while not curr.is_key(kwargs['key']):
                if curr.enext() is None:
                    curr = None
                    break
                curr = curr.enext()
        else:
            while curr != elem:
                if curr.enext() is None:
                    curr = None
                    break
                curr = curr.enext()
        return curr

    def add(self, elem):
        if self.start is None:
            self.start = elem
            self.end = elem

        else:
            self.end.set_next(elem)
            elem.set_prev(self.end)
            self.end = elem

    def remove(self, elem=None, by_key=False, **kwargs):
        key = None
        if 'key' in kwargs:
            key = kwargs['key']
        curr = self.get(elem, by_key, key=key)

        if curr is not None:

            if curr == self.start:
                nexter = curr.enext()
                nexter.set_prev(None)
                self.start = nexter

            elif curr == self.end:
                prevter = curr.eprev()
                prevter.set_next(None)
                self.end = prevter

            else:
                curr.eprev().set_next(curr.enext())

    def print(self):
        curr = self.start
        info = ''
        while curr:
            info += f'{str(curr)}\t'
            curr = curr.enext()
        return info


class HashTable:

    def __init__(self, len):
        self.len = len
        self.elements = [LinkedList() for _ in range(len)]

    def hash(self, key):
        return key % self.len

    def add(self, key, value):
        elem = LinkedElement(key, value)
        pos = self.hash(key)
        self.elements[pos].add(elem)

    def get_by_key(self, key):
        pos = self.hash(key)
        elem = self.elements[pos].get(by_key=True, key=key)
        if elem:
            return elem
        else:
            return None

    def print(self):
        for i in self.elements:
            print(f'CELL: {i.print()}')

    def remove(self, key):
        to_remove = self.get_by_key(key)
        if to_remove:
            pos = self.hash(key)
            self.elements[pos].remove(to_remove)


hash = HashTable(3)
hash.add(3, 'ff')
hash.add(9, 'ffs')
hash.add(1, 'fsdf')
hash.add(2, 'fsdf')
hash.print()









