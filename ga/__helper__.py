import hashlib


def hash_str(string):
    hasher = hashlib.md5()
    hasher.update(bytes(string, 'ascii'))
    return hasher.hexdigest()

def hash_file(filepath):
    #http://pythoncentral.io/hashing-files-with-python/
    blocksize=65536
    hasher = hashlib.md5()
    with open(filepath,'rb') as fh:
        buf = fh.read(blocksize)
        while len(buf)>0:
            hasher.update(buf)
            buf = fh.read(blocksize)
    return hasher.hexdigest()


class Command:
    def __init__(self, command, data, generation=0):
        self.command = command
        self.data = data
        self.generation = generation
        self.id = self._id()

    def __str__(self):
        return "<command {} with data {}>".format(self.command, self.data)

    def _id(self):
        return hash_str(self.__str__())

    def getFold(self):
        if self.command == "evaluate":
            return self.data[2]
        else:
            return -1
