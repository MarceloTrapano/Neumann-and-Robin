import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from ThomasSolve import ThomasSolve
import copy

class Neumann:
    '''
    Solves second order differencial equations with Neumann boundary conditions, given by equation y'' = f(x).
    '''
    def __init__(self, f, way : int = 2) -> None:
        '''
        Initiation of problem given by equation: y'' = f(x).
        
        Args:
            f (lambda function): function f
            way (int): pick way with which you want to solve the problem (from 0 to 2, defaults to 2)'''
        self.f = f
        self.way : int = way

        self.solution = None
        self.label : str = ""
        
        self.sol_exist : bool = False
        self.set_n : bool = False
        self.set_ab : bool = False

        self.derivative : list[bool] = [False, False]

        self.n = 1
        self.a = 0
        self.b = 2
        self.val_a = 0
        self.val_b = 0
        self.h = 0

        self.x : np.ndarray
        self.unknown_x : np.ndarray
    def setLeft(self) -> None:
        '''Consider left side as derivative.'''
        self.derivative : list[bool]= [True, False]
        self.setH()

    def setRight(self) -> None:
        '''Consider right side as derivative.'''
        self.derivative : list[bool]= [False, True]
        self.setH()
    def setN(self, n : int) -> None:
        '''
        Sets the number of points.
        
        Args:
            n (int): number of points'''
        self.n = n
        self.set_n = True
        self.setH()

    def setLabel(self, label : str) -> None:
        '''
        Sets given equation in LaTeX to write it on plot.
        
        Args:
            label (str): label of graph'''
        self.label = label

    def setAB(self, a : list, b : list) -> None:
        '''
        Sets boundary conditions.
        
        Args:
            a (list): Two element list containing
            value of left boundary argument and value.
            b (list): Two element list containing
            value of right boundary argument and value.'''
        self.a = a[0]
        self.b = b[0]
        self.val_a = a[1]
        self.val_b = b[1]
        self.set_ab = True
        self.setH()
    
    def setH(self) -> None:
        '''Sets distance between points'''
        self.h = (self.b - self.a)/(self.n+1)
        self.x = np.linspace(self.a, self.b, self.n+2)
        if not self.derivative[0]:
            self.unknown_x = self.x[1:self.n+2]
        elif not self.derivative[1]:
            self.unknown_x = self.x[0:self.n+1]

    def addSolution(self, solution) -> None:
        '''Add solution to the problem'''
        self.solution = solution
        self.sol_exist = True
    def solve(self) -> float:
        '''
        Solves the problem.
        
        Returns:
            float: solution depending on solve type variable.'''
        if not all([self.set_ab, self.set_n]) or not any(self.derivative):
            raise ValueError("Cannot solve equation without setting parameters")
        v1 : np.ndarray = (-2)*np.ones(self.n+1)
        v2 : np.ndarray = np.ones(self.n)
        v3 : np.ndarray = v2.copy()
        B : np.ndarray = (self.h**2)*np.array(list(map(self.f, self.unknown_x)))
                
        match self.way:
            case 0:
                if self.derivative[0]:
                    v1[0] = -1
                    B[0] = self.h*self.val_a
                    B[-1] = B[-1] - self.val_b
                else:
                    v1[-1] = 1
                    v3[-1] = -1
                    B[-1] = self.h*self.val_b
                    B[0] = B[0] - self.val_a

                sol = ThomasSolve(v3,v1,v2,B)
                return sol
            case 1:
                if self.derivative[0]:
                    v1[0] = -1
                    B[0] = 1/2*self.h**2*self.f(self.x[0]) + self.h*self.val_a
                    B[-1] = B[-1] - self.val_b
                else:
                    v1[-1] = -1
                    v3[-1] = 1
                    B[-1] = 1/2*self.h**2*self.f(self.x[-1]) - self.h*self.val_b
                    B[0] = B[0] - self.val_a
                sol = ThomasSolve(v3,v1,v2,B)
                return sol
            case 2:
                A = np.diag(v1)+np.diag(v2,1)+np.diag(v2,-1)
                if self.derivative[0]:
                    A[0][0] = -3; A[0][1] = 4; A[0][2] = -1
                    B[0] = 2*self.h*self.val_a
                    B[-1] = B[-1] - self.val_b
                else:
                    A[-1][-1] = 3; A[-1][-2] = -4; A[-1][-3] = 1
                    B[-1] = 2*self.h*self.val_b
                    B[0] = B[0] - self.val_a
                
                sol = LA.solve(A, B)
                return sol
            case _:
                raise ValueError("Nie ma takiej opcji")
    def plt_config(self) -> None:
        '''
        Configures plot.
        '''
        A = 6
        plt.rc('figure', figsize=[46.82 * .5**(.5 * A), 33.11 * .5**(.5 * A)])
        plt.rc('text', usetex=True)
        plt.rc('text.latex', preamble=r'\usepackage{polski}')
        plt.rc('font', family='serif')
    def getError(self) -> float:
        if not self.sol_exist:
            raise ValueError("Cannot get error without solution")
        unknown_sol = self.solve()
        if not self.derivative[0]:
            approx_sol = np.insert(unknown_sol, 0, self.val_a)
        else:
            approx_sol = np.insert(unknown_sol, len(unknown_sol), self.val_b)
        real_sol = list(map(self.solution, self.x))
        error_sol = np.abs(approx_sol-real_sol)
        error_norm = LA.norm(error_sol, np.inf)
        return error_norm
    def show(self) -> str:
        '''
        Plots the solution.
        
        Returns:
            str: Error norm if can.'''
        self.plt_config()
        unknown_sol = self.solve()
        if not self.derivative[0]:
            approx_sol = np.insert(unknown_sol, 0, self.val_a)
        else:
            approx_sol = np.insert(unknown_sol, len(unknown_sol), self.val_b)
        if self.sol_exist:
            real_sol = list(map(self.solution, self.x))
            error_sol = np.abs(approx_sol-real_sol)
            error_norm = LA.norm(error_sol, np.inf)
            plt.subplot(211)
            plt.grid(True)
            domain = np.linspace(self.a, self.b, 1000)
            plt.plot(domain ,list(map(self.solution, domain)), color='b', label=r"Rzeczywista funkcja")
            if self.n < 30:
                plt.scatter(self.x ,approx_sol, color='g', label=r"Aproksymacja")
            else:
                plt.plot(self.x ,approx_sol, color='g', label=r"Aproksymacja")
            plt.title(self.label)
            plt.ylabel(r"Wartości funkcji $f(x)$")
            plt.legend()
            plt.subplot(212)
            plt.grid(True)
            plt.plot(self.x ,error_sol, color='r', label=r"Błąd")
            plt.title(r"Wykres błędu")
            plt.xlabel(r"Wartości $x$")
            plt.ylabel(r"Wartość błędu")
            plt.subplots_adjust(hspace=0.4)
            plt.show()
            self.result = approx_sol
            return error_norm
        plt.grid(True)
        plt.plot(self.x ,approx_sol, color='g', label=r"Aproksymacja")
        plt.title(self.label)
        plt.xlabel(r"Wartości $x$")
        plt.ylabel(r"Wartości funkcji $f(x)$")
        plt.subplots_adjust(hspace=0.4)
        plt.show()
        self.result = approx_sol
        return ""
    
        
    def __str__(self) -> str:
        '''
        String representation of result. Shows result in plot with simple print function.
        '''
        solution = self.show()
        if self.sol_exist:
            return f"Rozwiązanie przybliżone:\n {self.result} \n Bład: {solution}"
        return f"Rozwiązanie przybliżone:\n {self.result}"
    
def main() -> None:
    f =     lambda x: 12*x
    n = 10
    a = [0, 2]
    b = [1, 1]
    neumann = Neumann(f, 2)

    neumann.setAB(a, b)
    neumann.addSolution(lambda x: 2*x**3+2*x-3)
    neumann.setLabel(r"$y''= 12x$")
    neumann.setN(n)
    neumann.setLeft()
    print(neumann)

if __name__ == '__main__':
    main()
    