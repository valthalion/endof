"""
Read TSPLIB benchmark files and transform them into the cost matrix needed for
ENDOF (Endof New Distributed Optimization Framework)

This file is distributed under the MIT license (http://opensource.org/licenses/MIT)
"""

class parsetsp(object):
    """Read a TSP problem specification from TSPLIB and build the cost matrix"""

    def __init__(self, inputfile):
        self.name, self.cm = self.parse(inputfile)


    def parse(self, inputfile):
        with open(inputfile, 'r') as f:
            line = f.readline()
            name = line.split()[-1]
            # Skip 6 rows, get size
            for _ in range(6):
                line = f.readline()
                if line.startswith("DIMENSION"):
                    size = int(line.split()[-1])
            cm = []
            numbers = []
            line = f.readline()
            while not line.startswith("EOF"):
                numbers.extend(int(n) for n in line.split())
                if len(numbers) >= size:
                    cm.append(numbers[:size])
                    numbers = numbers[size:]
                line = f.readline()
        return name, cm
