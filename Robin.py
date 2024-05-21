import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from ThomasSolve import ThomasSolve
class Robin:
    '''
    Solves second order differencial equations with Robin boundary conditions.
    '''
    def __init__(self, alpha, beta, gamma, f) -> None:
        '''
        Initiation of problem given by equation: alpha*y''+beta*y'+gamma*y = f(x).
        
        Args:
            alpha (lambda function): alpha coefficient
            beta (lambda function): beta coefficient
            gamma (lambda function): gamma coefficient
            f (lambda function): function f'''
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.f = f

        self.solution = None
        self.label : str = ""
        
        self.sol_exist : bool = False
        self.set_n : bool = False
        self.set_ab : bool = False

        self.n : int = 1
        self.a : float = 0
        self.b : float = 2
        self.val_a : float = 0
        self.val_b : float = 0
        self.der_a : float = 0
        self.der_b : float = 0
        self.coef_a : float = 0
        self.coef_b : float = 0
        self.h : float = 0

        self.x : np.ndarray
        self.unknown_x : np.ndarray
    def setN(self, n : int) -> np.ndarray:
        '''
        Sets the number of points.
        
        Args:
            n (int): number of points'''
        self.n = n
        self.set_n = True
        self.setH()
        return self.x

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
            a (list[float]): Four element list containing value of left boundary argument,
            value and coefficients next to derivative and y expresion.
            b (list[float]): Four element list containing value of right boundary argument,
            value and coefficients next to derivative and y expresion.'''
        self.a = a[0]
        self.b = b[0]
        self.val_a = a[1]
        self.val_b = b[1]
        self.der_a = a[2]
        self.der_b = b[2]
        self.coef_a = a[3]
        self.coef_b = b[3]
        self.set_ab = True
        self.setH()
    
    def setH(self) -> None:
        '''Sets distance between points'''
        self.h = (self.b - self.a)/(self.n+1)
        self.x = np.linspace(self.a, self.b, self.n+2)
        self.unknown_x = self.x[1:self.n+2]

    def addSolution(self, solution) -> None:
        '''Add solution to the problem'''
        self.solution = solution
        self.sol_exist = True
    def solve(self) -> float:
        '''
        Solves the problem.
        
        Returns:
            float: solution depending on solve type variable.'''
        if not all([self.set_ab, self.set_n]):
            raise ValueError("Cannot solve equation without setting parameters")
        alpha = np.array(list(map(self.alpha, self.x)))
        beta = np.array(list(map(self.beta, self.x)))
        v1 : np.ndarray = -2*alpha + self.h**2*np.array(list(map(self.gamma, self.x)))
        v2 : np.ndarray = alpha+1/2*self.h*beta
        v3 : np.ndarray = alpha-1/2*self.h*beta
        v2 = v2[0:-1]
        v3 = v3[1:len(v1)+1]
        B : np.ndarray = (self.h**2)*np.array(list(map(self.f, self.x)))
        v1[0] = self.coef_a*self.h - self.der_a
        v2[0] = self.der_a
        v1[-1] = self.coef_b*self.h + self.der_b
        v3[-1] = -self.der_b
        B[0] = self.val_a*self.h
        B[-1] = self.val_b*self.h
        sol : np.ndarray = ThomasSolve(v3,v1,v2,B)
        return sol
        
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
        approx_sol = self.solve()
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
        approx_sol = self.solve()
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
        if self.n < 30:
                plt.scatter(self.x ,approx_sol, color='g', label=r"Aproksymacja")
        else:
                plt.plot(self.x ,approx_sol, color='g', label=r"Aproksymacja")
        plt.title(self.label)
        plt.xlabel(r"Wartości $x$")
        plt.ylabel(r"Wartości funkcji $f(x)$")
        plt.legend()
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
    alpha =     lambda x: 1
    beta =      lambda x: -x
    gamma =     lambda x: 1
    f =         lambda x: np.exp(x)*(-x**2+x+2)
    n = 50
    a = [0, 0, 0, 1]
    b = [1, 2*np.exp(1), 1, 0]
    robin = Robin(alpha, beta, gamma, f)

    robin.setAB(a, b)
    robin.setLabel(r"$y''+x^2y'-xy= x$")
    robin.addSolution(lambda x: x*np.exp(x))
    robin.setN(n)

    print(robin)

if __name__ == '__main__':
    main()
    