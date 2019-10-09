import random


class Memory:

    def __init__(self, maxSize: int):
        self.datas = {}
        self.maxSize = maxSize
        self.nbItems = 0

    def append(self, dict: dict):
        for k in dict.keys():
            if k not in self.datas:
                self.datas[k] = [None] * self.nbItems

        for k in self.datas:
            if k in dict:
                self.datas[k].append(dict[k])
            else:
                self.datas[k].append(None)

        self.nbItems += 1
        while self.nbItems > self.maxSize:
            self.nbItems -= 1
            index = random.randint(0, self.nbItems)
            for k in self.datas:
                self.datas[k].pop(index)

    def getItem(self, index: int):
        item = {}
        for k in self.datas.keys():
            item[k] = self.datas[k][index]
        return item

    def getCol(self, col: str):
        return self.datas[col]

    def __getitem__(self, key: any):
        if type(key) is int:
            return self.getItem(key)
        elif type(key) is str:
            return self.getCol(key)
        else:
            return None
            
    def __getattr__(self, name):
        return self.getCol(name)

    def reduce(self, count: int):
        indexes = random.sample(range(self.nbItems), count)
        ret = Memory(count)
        for k in self.datas.keys():
            ret.datas[k] = [self.datas[k][i] for i in indexes]
        ret.nbItems = count
        return ret

    def __iter__(self):
        self._pos = 0
        return self

    def __next__(self):
        if self._pos < self.nbItems:
            ret = self.getItem(self._pos)
            self._pos += 1
            return ret
        else:
            raise StopIteration


if __name__ == "__main__":
    mem = Memory(20)
    keys = ["a", "b", "c", "d", "e", "f", "g", "h"]
    for i in range(30):
        item = {}
        for j in range(8):
            item[keys[random.randint(0, 7)]] = random.random()
        print(item)
        mem.append(item)

    print("---------------------------------------")
    for i in mem:
        print(i)

    print("---------------------------------------")
    for i in mem.reduce(10):
        print(i)

    print("---------------------------------------")
    print(mem[0])
    print(mem["a"])
