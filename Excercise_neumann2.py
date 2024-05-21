from Neumann import Neumann
from numpy import cos, sin, exp, pi
import numpy as np

def main() -> None:
    f = lambda x: exp(x)*sin(x)
    n : int = 50
    a : list[float]= [0, 0]
    b : list[float]= [pi, exp(1)]
    neumann : Neumann = Neumann(f, 2)

    neumann.setAB(a, b)
    neumann.addSolution(lambda x: 1/2*(1+2*exp(1)*x-exp(pi)*x-exp(x)*cos(x)))
    neumann.setLabel(r"$y''= e^x\sin x$")
    neumann.setN(n)
    neumann.setRight()
    lista = np.array([])
    for n in [10,100,500,1_000,5_000,10_000]:
        neumann.setN(n)
        lista = np.append(lista, neumann.getError())
    print(lista)

if __name__ == '__main__':
    main()