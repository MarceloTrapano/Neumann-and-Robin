from Neumann import Neumann
from numpy import cos, sin, exp
import numpy as np

def main() -> None:
    f =     lambda x: 12*x
    a = [0, 2]
    b = [1, 1]
    neumann = Neumann(f, 2)

    neumann.setAB(a, b)
    neumann.addSolution(lambda x: 2*x**3+2*x-3)
    neumann.setLabel(r"$y''= 12x$")
    neumann.setLeft()
    lista = np.array([])
    for n in [10,100,500,1_000,5_000,10_000]:
        neumann.setN(n)
        lista = np.append(lista, neumann.getError())
    print(lista)
if __name__ == '__main__':
    main()